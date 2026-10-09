"""Interview the user and save their news preferences as a validated profile"""

from dotenv import load_dotenv
from anthropic import Anthropic
from pydantic import ValidationError
from profile import Profile

load_dotenv()
client = Anthropic()

MODEL = "claude-haiku-5-5"
MAX_TURNS = 12
MAX_VALIDATION_TRIES = 3

SYSTEM_PROMPT = """
You are a discovery agent for personalised daily news digest.
You must interview the user to learn:
- the news topics they're interested in (min 1, max 8)
- any news sources they prefer or want to avoid
- the tone they prefer: neutral, casual or analytical
- how many stories they want per day (min 3, max 15)

Ask one or two short questions at a time. Don't guess missing details.
Once you have everything, call the save_profile tool exactly once.
"""

SAVE_PROFILE_TOOL = {
    "name": "save_profile",
    "description": "Save the user's news preferences once all details are known.",
    "input_schema": Profile.model_json_schema(),
}

def run_interview() -> Profile | None:
    messages = [{
        "role": "user",
        "content": "Hi, I'd like to set up my daily news digest"
    }]
    failed_validations = 0

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[SAVE_PROFILE_TOOL],
            messages=messages
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_call = next((b for b in response.content if b.type == "tool_use"), None)

        if tool_call is None:
            for block in response.content:
                if block.type == "text":
                    print(f"\nClaude: {block.text}")
            answer = input("\nYou: ").strip()
            messages.append({"role": "user", "content": answer or "(no answer)"})
            continue

        try:
            return Profile.model_validate(tool_call.input)
        except ValidationError as error:
            failed_validations += 1
            if failed_validation >= MAX_VALIDATION_TRIES:
                print("Could not get a valid profile. Giving up.")
                return None

            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": f"Validation failed, please fix and call again:\n{error}",
                    "is_error": True,
                }],
            })

    print("Interview took too long. Stopping.")

    return None

if __name__ == "__main__":
    profile = run_interview()
    if profile:
        with open("profile.json", "w") as f:
            f.write(profile.model_dump_json(indent=2))
        print("\nSaved profile.json:")
        print(profile.model_dump_json(indent=2))
