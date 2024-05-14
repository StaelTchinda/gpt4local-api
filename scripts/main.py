# Add current path to sys.path
import sys
from pathlib import Path
from typing import Literal, Text

from gpt4local.g4l.typing import Messages
PROJECT_ROOT_PATH = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT_PATH))

MODELS_PATH = PROJECT_ROOT_PATH / 'gpt4local' / 'models'

import argparse

from gpt4local.g4l.local import LocalEngine

EXISTING_MODELS = [
  "mistral-7b-instruct-v0.2.Q4_K_M",
  "orca-mini-3b-gguf2-q4_0",
]

LllmRole = Literal["user", "system", "assistant", "tool", "function"]

def parse_args():
    parser = argparse.ArgumentParser("Local LLM Engine")
    parser.add_argument('--model', type=str, default='mistral-7b-instruct-v0.2.Q4_K_M', help='Model name')
    parser.add_argument('--role', type=str, default='user', help='Role', choices=LllmRole.values)
    return parser.parse_args()


EXIT_COMMAND: Text = ":exit"


def main():
    args = parse_args()

    engine = LocalEngine(
        gpu_layers = -1,  # use all GPU layers
        cores      = 0    # use all CPU cores
    )
    input_message: Text = ""

    messages: Messages = []

    while input_message != EXIT_COMMAND:
        input_message = input("You: ")
        if input_message == EXIT_COMMAND:
            break

        messages.append({"role": args.role, "content": input_message})

        response = engine.chat.completions.create(
            model    = args.model,
            messages = messages,
            stream   = True
        )

        full_system_response: Text = ""

        for token in response:
            if token.choices[0].finish_reason == 'stop':
                print()
                break
            elif token.choices[0].delta.content is None:
                raise ValueError("Unexpected None content")
            print(token.choices[0].delta.content, end='')
            full_system_response += token.choices[0].delta.content

        messages.append({"role": "system", "content": full_system_response})


if __name__ == '__main__':
    main()
