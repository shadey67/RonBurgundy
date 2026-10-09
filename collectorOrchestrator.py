"""Run one collector per topic in parallel, then merge the results"""

import asyncio
import math
import time

from pydantic import BaseModel
from collector import collect, CollectionResult
from profile import Profile

TOPIC_TIMEOUT = 180
MAX_CONCURRENCY = 3
ATTEMPTS = 2
class CollectedBundle(BaseModel):
    results: list[CollectionResult]
    failed_topics: list[str]

async def run_topic(
        topic: str,
        profile: Profile,
        stories_wanted: int,
        semaphore: asyncio.Semaphore,
        timings: dict[str, float]
) -> CollectionResult | None:
    async with semaphore:
        for attempt in range(1, ATTEMPTS + 1):
            start = time.perf_counter()
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(collect, topic, profile, stories_wanted),
                    timeout = TOPIC_TIMEOUT
                )
            except asyncio.TimeoutError:
                print(f"[pipeline:{topic}] timed out (attempt {attempt}/{ATTEMPTS})")
                result = None
            except Exception as error:
                print(f"[pipeline:{topic}] error: {error} (attempt {attempt}/{ATTEMPTS})")
                result = None

            timings[topic] = timings.get(topic, 0.0) + (time.perf_counter() - start)
            if result is not None:
                return result
        return None

async def run_pipeline(profile: Profile) -> CollectedBundle:
    # Ask for a little extra per topic, because the ranker will drop duplicates later.
    stories_wanted = max(2, math.ceil(profile.max_stories / len(profile.topics)) + 1)

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    timings: dict[str, float] = {}
    started = time.perf_counter()

    # gather() starts all the topic tasks together and returns results in topic order.
    results = await asyncio.gather(
        *(run_topic(t, profile, stories_wanted, semaphore, timings) for t in profile.topics)
    )

    wall_time = time.perf_counter() - started
    sequential_estimate = sum(timings.values())
    print(f"\nFinished in {wall_time:.0f}s "
          f"(running one after another would have taken about {sequential_estimate:.0f}s)")

    ok = [r for r in results if r is not None]
    failed = [t for t, r in zip(profile.topics, results) if r is None]
    if failed:
        print(f"Failed topics: {', '.join(failed)}")
    return CollectedBundle(results=ok, failed_topics=failed)


if __name__ == "__main__":
    with open("profile.json") as f:
        profile = Profile.model_validate_json(f.read())

    bundle = asyncio.run(run_pipeline(profile))

    with open("collected.json", "w") as f:
        f.write(bundle.model_dump_json(indent=2))

    total = sum(len(r.stories) for r in bundle.results)
    print(f"Saved {total} stories across {len(bundle.results)} topics to collected.json")
