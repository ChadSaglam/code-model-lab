"""Behaviour tests for glm53_flash.tokenizer.ByteTokenizer.

Every training example and every model answer passes through this byte tokenizer, so its
boundaries matter: a silent off-by-one in the byte offset or a mishandled special token would
corrupt the whole corpus. The existing suite (test_lab.py) only checks one ASCII round trip;
these tests pin the byte-offset mapping, the bos/eos flag combinations, decode's stop-at-eos
and skip-the-specials rules, multi-byte UTF-8, invalid-byte replacement, and the empty string.
No torch and no model are touched.
"""
from __future__ import annotations

from glm53_flash import ByteTokenizer

tokenizer = ByteTokenizer()


# ---------------------------------------------------------------- the id layout

def test_the_reserved_ids_and_byte_offset_are_fixed():
    # four control ids below the byte window, then the byte slots start at the offset.
    assert (ByteTokenizer.pad_id, ByteTokenizer.bos_id,
            ByteTokenizer.eos_id, ByteTokenizer.sep_id) == (0, 1, 2, 3)
    assert ByteTokenizer.byte_offset == 4
    assert ByteTokenizer.vocab_size == 260


# ---------------------------------------------------------------- encode: the byte-offset mapping

def test_each_byte_is_shifted_by_the_byte_offset():
    # "A" is byte 65, "B" is 66; both land at value + byte_offset (4), no special tokens.
    assert tokenizer.encode("AB") == [69, 70]


def test_the_null_byte_maps_to_the_first_byte_slot():
    # byte 0 is the lowest byte value; it must map to byte_offset, not collide with a special id.
    assert tokenizer.encode("\x00") == [ByteTokenizer.byte_offset]


def test_every_encoded_token_sits_above_the_special_ids():
    tokens = tokenizer.encode("def f(x):\n    return x + 1\n")
    assert all(token >= ByteTokenizer.byte_offset for token in tokens)
    assert max(token for token in tokens) < ByteTokenizer.vocab_size


# ---------------------------------------------------------------- encode: the bos/eos flags

def test_no_flags_emits_only_byte_tokens():
    assert tokenizer.encode("A") == [69]


def test_bos_is_inserted_at_the_front():
    assert tokenizer.encode("A", bos=True) == [ByteTokenizer.bos_id, 69]


def test_eos_is_appended_at_the_end():
    assert tokenizer.encode("A", eos=True) == [69, ByteTokenizer.eos_id]


def test_both_flags_wrap_the_content():
    assert tokenizer.encode("A", bos=True, eos=True) == [
        ByteTokenizer.bos_id, 69, ByteTokenizer.eos_id]


def test_empty_string_without_flags_is_empty():
    assert tokenizer.encode("") == []


def test_empty_string_with_both_flags_is_just_the_markers():
    assert tokenizer.encode("", bos=True, eos=True) == [
        ByteTokenizer.bos_id, ByteTokenizer.eos_id]


# ---------------------------------------------------------------- encode: multi-byte UTF-8

def test_a_multibyte_character_becomes_one_token_per_utf8_byte():
    # "€" is U+20AC -> UTF-8 bytes 0xE2 0x82 0xAC; each byte is offset independently.
    assert tokenizer.encode("€") == [0xE2 + 4, 0x82 + 4, 0xAC + 4]


def test_token_count_matches_the_utf8_byte_length_not_the_character_count():
    text = "café"  # 'é' is two UTF-8 bytes, so five bytes for four characters
    assert len(tokenizer.encode(text)) == len(text.encode("utf-8")) == 5


# ---------------------------------------------------------------- decode: stop at eos

def test_decode_stops_at_the_first_eos_and_drops_the_rest():
    # 69,70 decode to "AB"; the eos marker ends the string and the trailing 99 is never read.
    assert tokenizer.decode([69, 70, ByteTokenizer.eos_id, 99]) == "AB"


def test_decode_of_an_eos_first_sequence_is_empty():
    assert tokenizer.decode([ByteTokenizer.eos_id, 69, 70]) == ""


# ---------------------------------------------------------------- decode: skip the specials

def test_decode_skips_pad_bos_and_sep_tokens():
    # pad (0), bos (1) and sep (3) are below the byte offset and carry no byte; only 'A' remains.
    assert tokenizer.decode(
        [ByteTokenizer.pad_id, ByteTokenizer.bos_id, ByteTokenizer.sep_id, 69]) == "A"


def test_a_skipped_special_in_the_middle_does_not_break_the_run():
    assert tokenizer.decode([69, ByteTokenizer.pad_id, 70]) == "AB"


def test_decode_of_an_empty_list_is_the_empty_string():
    assert tokenizer.decode([]) == ""


# ---------------------------------------------------------------- decode: invalid bytes

def test_an_invalid_utf8_byte_becomes_the_replacement_character_not_a_crash():
    # token 132 is byte 0x80, a lone continuation byte that is not valid UTF-8 on its own.
    assert tokenizer.decode([4 + 0x80]) == "�"


# ---------------------------------------------------------------- round trips

def test_ascii_round_trip_with_markers_returns_the_original():
    text = "def f(x):\n    return x + 1\n"
    assert tokenizer.decode(tokenizer.encode(text, bos=True, eos=True)) == text


def test_multibyte_round_trip_with_markers_returns_the_original():
    text = "café € ηθ\n"
    assert tokenizer.decode(tokenizer.encode(text, bos=True, eos=True)) == text


def test_empty_string_round_trips_through_the_markers():
    assert tokenizer.decode(tokenizer.encode("", bos=True, eos=True)) == ""


def test_the_leading_bos_marker_is_not_decoded_back_into_text():
    assert tokenizer.decode(tokenizer.encode("hi", bos=True)) == "hi"


# ---------------------------------------------------------------- the vocabulary bound

def test_vocab_size_covers_the_specials_and_every_byte_value():
    # four special ids plus all 256 byte values; this is the model's output dimension.
    assert ByteTokenizer.vocab_size == ByteTokenizer.byte_offset + 256


def test_every_byte_value_maps_to_a_distinct_id_inside_the_vocab():
    # all 256 byte values get their own slot and the top one still fits the vocab.
    ids = [ByteTokenizer.byte_offset + value for value in range(256)]
    assert len(set(ids)) == 256
    assert min(ids) == ByteTokenizer.byte_offset
    assert max(ids) == ByteTokenizer.vocab_size - 1
