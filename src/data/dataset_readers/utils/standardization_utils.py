import re
import sys


whitespaces = re.findall(r"\s", u"".join(chr(c) for c in range(sys.maxunicode + 1)), re.UNICODE)
empty_chars = ["\u200b", "\ufeff", "\u2061"]  # zero width space, byte order mark


def standardize_text_simple(text, output_offset=False):
    for whitespace in whitespaces:
        if whitespace == "\n" or whitespace == "\t":
            continue
        text = text.replace(whitespace, " ")

    for empty_char in empty_chars:
        text = text.replace(empty_char, " ")

    stripped_text = text.strip()
    offset = len(stripped_text) - len(text.rstrip())
    return (stripped_text, offset) if output_offset else stripped_text
