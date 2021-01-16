from typing import List, Tuple, Callable, Optional, NamedTuple
import unicodedata

from allennlp_models.rc.dataset_readers.utils import STRIPPED_CHARACTERS

from src.data.types import Span


def get_sequence_boundaries(special_tokens_mask: List[int]):
    """
    Returns the token index boundaries of a sequence that was encoded together with other sequences,
    by using special_tokens_mask.
    """
    boundaries = []
    start_index = None
    special_sequence = True
    for i, value in enumerate(special_tokens_mask):
        if value == 0:
            if special_sequence is True:
                start_index = i
                special_sequence = False
        elif value == 1:
            if special_sequence is False:
                boundaries.append(Span(start_index, (i - 1)))
                special_sequence = True
    if special_sequence is False:
        boundaries.append(Span(start_index, len(special_tokens_mask) - 1))
    return boundaries


def get_token_answer_span(
    offset_mapping: List[Tuple[int, int]],
    special_tokens_mask: List[int],
    answer: str,
    answer_start_offset: int,
    sequence_index: int,
):
    answer_end_offset = answer_start_offset + len(answer)

    sequence_range = get_sequence_boundaries(special_tokens_mask)[sequence_index]

    if (
        answer_start_offset < offset_mapping[sequence_range.start][0]
        or offset_mapping[sequence_range.end][1] < answer_end_offset
    ):
        return None

    answer_token_indices = []
    for i, offset in enumerate(offset_mapping):
        if i < sequence_range.start or i > sequence_range.end:
            continue
        is_start = offset[0] <= answer_start_offset and answer_start_offset < offset[1]
        is_mid = answer_start_offset <= offset[0] and offset[1] <= answer_end_offset
        is_end = offset[0] < answer_end_offset and answer_end_offset <= offset[1]

        if is_start or is_mid or is_end:
            answer_token_indices.append(i)

    token_answer_span = Span(answer_token_indices[0], answer_token_indices[-1])
    return token_answer_span


def find_all(substr, text):
    matches = []
    start = 0
    while True:
        start = text.find(substr, start)
        if start == -1:
            break
        matches.append(start)
        start += 1
    return matches


def run_strip_accents(text):
    """
    From tokenization_bert.py by huggingface/transformers.
    Strips accents from a piece of text.
    """
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


class TokensGroup(NamedTuple):
    token_indices: List[int]
    is_punctuation: bool


def group_tokens_by_whole_words(
    origin_texts: List[str],
    offset_mapping: List[Tuple[int, int]],
    special_tokens_mask: List[int],
    is_drop_directed=False,
) -> List[TokensGroup]:
    # This is a very rough heuristic - Assume that other than space, words are separated by STRIPPED_CHARACTERS,
    # Construct the string offset-by-offset from the original text,
    # and stop each time a separation indicator is encountered to construct a whole word.
    whole_word_separators = "".join([STRIPPED_CHARACTERS, "\u2013", "\u2014", "\n", "\t"])

    sequence_ranges = get_sequence_boundaries(special_tokens_mask)

    tokens_groups: List[TokensGroup] = [None for i in range(len(offset_mapping))]
    tokens_group_indices = []
    for i, (start_offset, end_offset) in enumerate(offset_mapping):
        sequence_index = None
        for j, sequence_range in enumerate(sequence_ranges):
            if i >= sequence_range.start and i <= sequence_range.end:
                sequence_index = j
                break
        if sequence_index is None:
            for index in tokens_group_indices:
                tokens_groups[index] = TokensGroup(tokens_group_indices, False)
            tokens_group_indices = []
            continue

        origin_text = origin_texts[sequence_index]

        token_text = origin_text[start_offset:end_offset]

        is_first_in_sequence = i == sequence_ranges[sequence_index].start

        is_after_space = False
        if i > 0:
            is_after_space = offset_mapping[i - 1][1] != start_offset

        is_group_separator = any([char in whole_word_separators for char in token_text])

        is_ordinal_numeral = False
        if len(tokens_group_indices) > 0:
            last_ingested_char = origin_text[offset_mapping[tokens_group_indices[-1]][1] - 1]
            is_ordinal_numeral = (
                token_text
                in [
                    "st",
                    "nd",
                    "rd",
                    "th",
                ]
                and last_ingested_char in [str(n) for n in range(0, 9)]
            )

        if (
            is_first_in_sequence
            or is_after_space
            or is_group_separator
            or (is_ordinal_numeral and is_drop_directed)
        ):
            for index in tokens_group_indices:
                tokens_groups[index] = TokensGroup(tokens_group_indices, False)
            tokens_group_indices = []

            if is_group_separator:
                tokens_groups[i] = TokensGroup([i], True)
            else:
                tokens_group_indices.append(i)
        else:
            tokens_group_indices.append(i)

    for index in tokens_group_indices:
        tokens_groups[index] = TokensGroup(tokens_group_indices, False)
    tokens_group_indices = []

    return tokens_groups


def find_valid_spans(
    text: str,
    answer_text: str,
    offset_mapping: List[Tuple[int, int]],
    special_tokens_mask,
    get_token_answer_span_partial: Callable,
    tokens_groups: List[TokensGroup],
    answer_start_offset: Optional[int] = None,
) -> Tuple[str, List[Span]]:
    text = (
        text.lower() if answer_start_offset is None else text
    )  # .lower() can change the length of the text

    answer_texts = []
    if answer_start_offset is None:
        option1 = answer_text.lower()
        answer_texts.append(option1)
        option2 = option1.strip(STRIPPED_CHARACTERS)
        if option2 not in answer_texts:
            answer_texts.append(option2)
        option3 = run_strip_accents(option1)
        if option3 not in answer_texts:
            answer_texts.append(option3)
        option4 = run_strip_accents(option2)
        if option4 not in answer_texts:
            answer_texts.append(option4)
    else:
        answer_texts.append(answer_text)

    valid_spans = []

    known_token_answer_span = None
    if answer_start_offset is not None:
        known_token_answer_span = get_token_answer_span_partial(
            answer=answer_text,
            answer_start_offset=answer_start_offset,
            offset_mapping=offset_mapping,
            special_tokens_mask=special_tokens_mask,
        )
        if known_token_answer_span is None:
            return valid_spans
        else:
            known_answer = text[
                offset_mapping[known_token_answer_span.start][0] : offset_mapping[
                    known_token_answer_span.end
                ][1]
            ]
            if known_answer in answer_texts:
                valid_spans.append(known_token_answer_span)
                return valid_spans

    for answer_text_option in answer_texts:
        start_offsets = find_all(answer_text_option, text)

        if answer_start_offset is not None:
            start_offsets = sorted(
                start_offsets,
                key=lambda start_offset: abs(answer_start_offset - start_offset),
            )

        for start_offset in start_offsets:
            token_answer_span = get_token_answer_span_partial(
                answer=answer_text_option,
                answer_start_offset=start_offset,
                offset_mapping=offset_mapping,
                special_tokens_mask=special_tokens_mask,
            )
            if token_answer_span is None:
                continue

            if known_token_answer_span is not None:
                if token_answer_span != known_token_answer_span:
                    diff_threshold = 10
                    if (
                        abs(token_answer_span.start - known_token_answer_span.start)
                        >= diff_threshold
                        or abs(token_answer_span.end - known_token_answer_span.end)
                        >= diff_threshold
                    ):
                        continue
                valid_spans.append(token_answer_span)
                break
            else:
                # Whole-word matching
                text_first_token_index = tokens_groups[token_answer_span.start].token_indices[0]
                text_last_token_index = tokens_groups[token_answer_span.end].token_indices[-1]
                if (
                    text_first_token_index != token_answer_span.start
                    or text_last_token_index != token_answer_span.end
                ):
                    continue

                # Text matching
                stripped_answer_text = answer_text_option.strip(STRIPPED_CHARACTERS)
                token_start_offset, token_end_offset = (
                    offset_mapping[token_answer_span.start][0],
                    offset_mapping[token_answer_span.end][1],
                )
                stripped_text_part = (
                    text[token_start_offset:token_end_offset].lower().strip(STRIPPED_CHARACTERS)
                )
                if stripped_answer_text != stripped_text_part:
                    continue

                valid_spans.append(token_answer_span)

        if len(valid_spans) > 0 and known_token_answer_span is not None:
            break

    return valid_spans
