"""Run the whole digest: profile -> collect -> rank -> write -> (email).

Usage:
    python main.py              # dry run: builds digest.html / digest.txt, no email
    python main.py --send       # also emails the digest
    python main.py --new-profile  # redo the interview even if profile.json exists
"""

import argparse
import asyncio
import os
import smtplib
import sys
import time
from pathlib import Path

from interview import run_interview
from collectorOrchestrator import run_pipeline
from ranker import rank
from RonBurgundy import write_digest, render, send_email

from profile import Profile

PROFILE_FILE = Path("profile.json")


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    sys.exit(1)


def load_or_create_profile(force_new: bool) -> Profile:
    """Use profile.json if it exists. Only run the interview if it doesn't."""
    if PROFILE_FILE.exists() and not force_new:
        try:
            profile = Profile.model_validate_json(PROFILE_FILE.read_text(encoding="utf-8"))
        except ValueError as error:  # covers invalid JSON, bad fields and bad encoding
            fail(f"{PROFILE_FILE} is not valid ({error}). Delete it or run with --new-profile.")
        print(f"Using existing profile ({len(profile.topics)} topics: {', '.join(profile.topics)})")
        return profile

    if not sys.stdin.isatty():
        fail("No profile.json yet and this isn't an interactive session. "
             "Run `python main.py` yourself once to do the interview.")

    print("No profile found. Starting the interview.\n")
    profile = run_interview()
    if profile is None:
        fail("The interview did not produce a valid profile.")
    PROFILE_FILE.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nSaved {PROFILE_FILE}")
    return profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--send", action="store_true", help="actually email the digest")
    parser.add_argument("--new-profile", action="store_true", help="redo the interview")
    args = parser.parse_args()

    os.chdir(Path(__file__).resolve().parent)

    started = time.perf_counter()
    timings: dict[str, float] = {}

    profile = load_or_create_profile(args.new_profile)

    # 2. Collect
    print("\n=== Collecting stories ===")
    t = time.perf_counter()
    bundle = asyncio.run(run_pipeline(profile))
    timings["collect"] = time.perf_counter() - t
    Path("collected.json").write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    total_collected = sum(len(r.stories) for r in bundle.results)
    if total_collected == 0:
        fail("No stories were collected for any topic.")

    # 3. Rank
    print("\n=== Ranking ===")
    t = time.perf_counter()
    ranked = rank(profile, bundle)
    timings["rank"] = time.perf_counter() - t
    if ranked is None or not ranked.selected:
        fail("Ranking failed or selected no stories.")
    Path("ranked.json").write_text(ranked.model_dump_json(indent=2), encoding="utf-8")

    # 4. Write
    print("\n=== Writing the digest ===")
    t = time.perf_counter()
    digest = write_digest(profile, ranked.selected)
    timings["write"] = time.perf_counter() - t
    if digest is None:
        fail("The writer could not produce a valid digest.")
    text, html_doc = render(digest, ranked.selected)
    Path("digest.txt").write_text(text, encoding="utf-8")
    Path("digest.html").write_text(html_doc, encoding="utf-8")

    # 5. Email (only with --send)
    emailed = False
    if args.send:
        print("\n=== Sending email ===")
        try:
            send_email(digest.subject, text, html_doc)
            emailed = True
        except (smtplib.SMTPException, OSError) as error:
            fail(f"Email failed: {error}. The digest was still saved to digest.html.")

    # Summary
    print("\n=== Done ===")
    print(f"Subject:   {digest.subject}")
    print(f"Stories:   {len(ranked.selected)} selected from {total_collected} collected")
    if bundle.failed_topics:
        print(f"Warning:   no stories for: {', '.join(bundle.failed_topics)}")
    print("Timings:   " + ", ".join(f"{k} {v:.0f}s" for k, v in timings.items())
          + f", total {time.perf_counter() - started:.0f}s")
    print("Files:     digest.html, digest.txt")
    print("Emailed:   yes" if emailed else "Emailed:   no (dry run, use --send)")


if __name__ == "__main__":
    main()