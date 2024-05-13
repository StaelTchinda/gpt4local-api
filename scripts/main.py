# Add current path to sys.path
import sys
from pathlib import Path
from typing import Optional, Text
PROJECT_ROOT_PATH = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT_PATH))

MODELS_PATH = PROJECT_ROOT_PATH / 'gpt4local' / 'models'

import argparse

from gpt4local.g4l.local import LocalEngine



def parse_args():
    parser = argparse.ArgumentParser("Local LLM Engine")
    parser.add_argument('--model', type=str, default='orca-mini-3b-gguf2-q4_0', help='Model name')
    parser.add_argument('--role', type=str, default='user', help='Role')
    return parser.parse_args()


EXIT_COMMAND: Text = ":exit"


def main():
    args = parse_args()

    engine = LocalEngine(
        gpu_layers = -1,  # use all GPU layers
        cores      = 0    # use all CPU cores
    )
    input_message: Text = ""

    while input_message != EXIT_COMMAND:
        input_message = input("You: ")
        if input_message == EXIT_COMMAND:
            break

        response = engine.chat.completions.create(
            model    = args.model,
            messages = [{"role": args.role, "content": input_message}],
            stream   = True
        )


        for token in response:
            if token.choices[0].finish_reason == 'stop':
                print()
                break
            elif token.choices[0].delta.content is None:
                raise ValueError("Unexpected None content")
            print(token.choices[0].delta.content, end='')


if __name__ == '__main__':
    main()
