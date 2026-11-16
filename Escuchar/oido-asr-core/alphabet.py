"""Vocabulario explícito del reconocedor propio. El índice 0 es CTC blank."""
VOCABULARY = " abcdefghijklmnñopqrstuvwxyzáéíóúü"
BLANK = 0
TO_INDEX = {character: index + 1 for index, character in enumerate(VOCABULARY)}
FROM_INDEX = {index: character for character, index in TO_INDEX.items()}


def encode(text: str) -> list[int]:
    return [TO_INDEX[char] for char in text.lower() if char in TO_INDEX]


def decode_ctc(indices) -> str:
    result, previous = [], BLANK
    for index in indices:
        index = int(index)
        if index != BLANK and index != previous:
            result.append(FROM_INDEX.get(index, ""))
        previous = index
    return "".join(result).strip()
