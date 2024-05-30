"""
  File    : p5tokenize.py
  Brief   : A command-line tool for testing the powered T5 tokenizer.
  Author  : Martin Rizzo | <martinrizzo@gmail.com>
  Date    : May 2, 2024
  Repo    : https://github.com/martin-rizzo/ComfyUI-PixArt
  License : MIT
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                             Powered T5 Encoder
       An enhanced T5 encoder with integrated weighted prompt support

     Copyright (c) 2024 Martin Rizzo

     Permission is hereby granted, free of charge, to any person obtaining
     a copy of this software and associated documentation files (the
     "Software"), to deal in the Software without restriction, including
     without limitation the rights to use, copy, modify, merge, publish,
     distribute, sublicense, and/or sell copies of the Software, and to
     permit persons to whom the Software is furnished to do so, subject to
     the following conditions:

     The above copyright notice and this permission notice shall be
     included in all copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
     EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
     CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
     TORT OR OTHERWISE, ARISING FROM,OUT OF OR IN CONNECTION WITH THE
     SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
"""
import time
import argparse
from src.t5 import T5Tokenizer


TEST_CASES = [
    ("Normal text",
     "[(16612, '1.00', 1), (1499, '1.00', 2), (1, '1.00', 0)]"
    ),
    ("An (important) word",
     "[(389, '1.00', 1), (359, '1.10', 2), (1448, '1.00', 3), (1, '1.00', 0)]"
    ),
    ("The (unnecessary)(parens)",
     "[(37, '1.00', 1), (12592, '1.10', 2), (1893, '1.10', 2), (35, '1.10', 2), (7, '1.10', 2), (1, '1.00', 0)]"
    ),
    ("\[Escaped\] \(text\)",
     "[(784, '1.00', 1), (427, '1.00', 1), (12002, '1.00', 1), (26, '1.00', 1), (908, '1.00', 1), (41, '1.00', 2), (6327, '1.00', 2), (61, '1.00', 2), (1, '1.00', 0)]"
    ),
    ("(A great (((house:1.2)) [on] a (hill:0.5) under an intense) (((sun))) \(score:8\).",
     "[(71, '1.00', 1), (248, '1.00', 2), (629, '1.45', 3), (30, '1.00', 4), (3, '1.10', 5), (9, '1.10', 5), (9956, '0.55', 6), (365, '1.10', 7), (46, '1.10', 8), (6258, '1.10', 9), (1997, '1.33', 10), (41, '1.00', 11), (7, '1.00', 11), (9022, '1.00', 11), (10, '1.00', 11), (13520, '1.00', 11), (5, '1.00', 11), (1, '1.00', 0)]"
    )
]

DEFAULT_PROMPT_TO_TEST_TIME = """
A ((photo)) of a [small] flying ((hovercraft:1.1) with a pilot inside),
hovering over the [((rugged)) surface of Mars].
[The ((craft:1.2)), made of [reflective metals], (gleams under the (sunlight))].
(In the background, a tunnel [[entrance]] adds a sense of (mystery:1.15)).
(The image, with its [selective focus], captures ((detailed textures)),
and (an atmospheric color palette that emphasizes the Martian landscape)).
"""
DEFAULT_REPEAT_COUNT = 1000



def convert_to_string(tokens_weights: list, weight_format: str = "{:.2f}") -> str:
    """
    Convert a list of token-weight pairs (and optional word IDs) to a string.

    Args:
        tokens_weights (list): List of tuples with tokens, weights, and optional word IDs.
        weight_format  (str) : Format for weights (default is "{:.2f}").

    Returns:
        str: A string representation of the provided list.
    """
    if not tokens_weights:
        return tokens_weights
    if len(tokens_weights[0]) == 3:
        tokens_weights = [(token, weight_format.format(weight), word_id) for token, weight, word_id in tokens_weights]
    else:
        tokens_weights = [(token, weight_format.format(weight)) for token, weight in tokens_weights]
    return str(tokens_weights)



def test(tokenizer:T5Tokenizer) -> bool:
    """
    Test the 'tokenize_with_weights' function using predefined test cases.
    Prints a summary of test results, including details of passed and failed cases.

    Args:
        tokenizer (T5Tokenizer): The tokenizer to be tested.

    Returns:
        bool: True if all test cases pass, False otherwise.
    """
    passed_cases = []
    failed_cases = []

    for prompt, expected_output in TEST_CASES:
        actual_output = tokenizer.tokenize_with_weights(prompt, include_word_ids=True)[0]
        actual_output = convert_to_string( actual_output, weight_format="{:.2f}" )

        if actual_output == expected_output:
            passed_cases.append(prompt)
        else:
            failed_cases.append((prompt, expected_output, actual_output))

    print("\nTEST RESULTS\n============")
    print("\n Passed Cases:")
    if not passed_cases:
        print("  None")
    else:
        for prompt in passed_cases:
            print(f'  - "{prompt}"')

    print("\n Failed Cases:")
    if not failed_cases:
        print("  None")
    else:
        for prompt, expected, actual in failed_cases:
            print(f'  - Prompt "{prompt}"')
            print(f'    Expected: {expected}')
            print(f'    Actual  : {actual}')

    print(f"\nSummary: {len(passed_cases)} passed, {len(failed_cases)} failed\n")
    return len(failed_cases) == 0



def test_time(tokenizer  : T5Tokenizer,
              prompt     : str = None,
              repetitions: int = None
              ) -> float:
    """
    Test the time taken by the tokenizer to process a prompt multiple times.

    Args:
        tokenizer   (T5Tokenizer): The tokenizer to be tested.
        prompt      (str): The text prompt to be tokenized.
        repetitions (int): The number of times to tokenize the prompt.

    Returns:
        float, float: The elapsed time in seconds to tokenize the prompt.
    """
    if prompt is None:
        prompt = DEFAULT_PROMPT_TO_TEST_TIME
    if repetitions is None:
        repetitions = DEFAULT_REPEAT_COUNT

    cleaned_prompt = prompt.strip()

    # tokenize the prompt multiple times and measure the elapsed time
    start_time = time.time()
    for _ in range(repetitions):
        tokenizer.tokenize_with_weights(cleaned_prompt)
    default_elapsed_time = time.time() - start_time
    start_time = time.time()
    for _ in range(repetitions):
        tokenizer.tokenize_with_weights(cleaned_prompt, include_word_ids=True)
    word_id_elapsed_time = time.time() - start_time

    # print the prompt, timing results and reference times
    print(f'\nPROMPT:\n=======\n{cleaned_prompt}\n')
    print(f' Tokenizing {repetitions} times took')
    print(f'   default       : {default_elapsed_time} seconds.')
    print(f'   with word-ids : {word_id_elapsed_time} seconds.\n')
    #
    #  REFERENCE      T5Tokenizer     T5TokenizerFast
    #   default          ~0.9s             ~0.7s
    #   with word-ids    ~1.8s             ~1.4s
    #
    return default_elapsed_time, word_id_elapsed_time



#===========================================================================#
#////////////////////////////////// MAIN ///////////////////////////////////#
#===========================================================================#

def main():
    parser = argparse.ArgumentParser(
        description='Tool for testing the powered T5 tokenizer.',
        add_help=False
        )
    parser.add_argument('prompt', nargs='*', help='The text prompt. Can be a single parameter or multiple.')
    parser.add_argument('-w', '--wordid', action='store_true', help='Display the word ID for each token.')
    parser.add_argument('-t', '--test', action='store_true', help='Run tests for the tokenize_with_weights function.')
    parser.add_argument(      '--time', action='store_true', help='Measure the time taken for the tokenize_with_weights function.')
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this help message and exit.')
    parser.add_argument('--version',  action='version', version='%(prog)s 0.1')

    args      = parser.parse_args()
    tokenizer = T5Tokenizer.from_pretrained(legacy=True)
    prompt    = ' '.join(args.prompt) if args.prompt else ''

    if args.test:
        test(tokenizer)
    elif args.time:
        prompt = prompt if prompt.strip() else None
        test_time(tokenizer, prompt)
    else:
        tokens_weights = tokenizer.tokenize_with_weights(prompt,
                                                         padding          = False,
                                                         padding_max_size = 0,
                                                         include_word_ids = args.wordid
                                                         )[0]
        tokens_weights = convert_to_string( tokens_weights )
        print( tokens_weights )

if __name__ == '__main__':
    main()
