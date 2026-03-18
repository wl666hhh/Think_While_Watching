import re
from typing import Union, List, Tuple

def seperate_source_target(text: Union[str, List[str]]) -> Tuple[str, str]:

    if isinstance(text, list):
        if len(text) == 0:
            raise ValueError("Input list is empty.")
        text = text[0]

    if not isinstance(text, str):
        raise ValueError(f"Input must be a string or list of strings, got: {type(text)}")

    pattern = (
        r"(<\|im_start\|\>system.*?<\|im_end\|\>.*?"
        r"<\|im_start\|\>user.*?<\|im_end\|\>\n)"
        r"(<\|im_start\|\>assistant.*)"
    )

    match = re.search(pattern, text, flags=re.S)

    if match:
        source_text = match.group(1)
        target_text = match.group(2)
        return source_text, target_text
    else:
        raise ValueError(
            f"Text format incorrect. Expected format:\n"
            f"<|im_start|>system<|im_end|>\\n<|im_start|>user<|im_end|>\\n<|im_start|>assistant<|im_end|>\\n\n"
            f"Actual text (first 100 chars): {text[:100]}..."
        )
