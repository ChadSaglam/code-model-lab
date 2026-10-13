"""Behaviour tests for local_models' answer parsing and friendly error mapping.

extract_function pulls a function out of a chat-style reply (thinking blocks, markdown
fences, surrounding prose, inconsistent indentation); friendly turns a verifier's internal
message into words a learner understands. Both run no model and never reach Ollama.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lab"))


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "lab" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


local_models = load("local_models")


# ---------------------------------------------------------------- extract_function

def test_thinking_block_is_stripped_before_the_function():
    reply = "<think>maybe def wrong(): pass</think>def foo():\n    return 2\n"
    assert local_models.extract_function(reply, "foo") == "def foo():\n    return 2\n"


def test_thinking_block_is_stripped_before_a_fenced_function():
    # the <think> section names a different, wrong body; it must not leak past the fence either.
    reply = ("<think>I will write def foo(a, b): return 999</think>\n"
             "```python\ndef foo(a, b):\n    return a + b\n```")
    assert local_models.extract_function(reply, "foo") == "def foo(a, b):\n    return a + b\n"


def test_fenced_block_that_holds_the_target_is_chosen_over_others():
    reply = ("```python\ndef other():\n    return 0\n```\n"
             "and the real one:\n"
             "```python\ndef target(x):\n    return x + 1\n```")
    assert local_models.extract_function(reply, "target") == "def target(x):\n    return x + 1\n"


def test_first_fence_is_used_when_none_names_the_target():
    reply = "```\ndef helper():\n    return 1\n```"
    assert local_models.extract_function(reply, "missing") == "def helper():\n    return 1\n"


def test_first_fence_with_only_prose_is_returned_as_is():
    # when no fence names the target and the first holds no def at all, its prose is handed back.
    reply = "```python\njust some prose, no function here\n```"
    assert local_models.extract_function(reply, "missing") == "just some prose, no function here\n"


def test_bare_fence_without_a_language_tag_is_read():
    # a bare ``` fence that holds the target reads back to the exact body, like a ```python one.
    reply = "```\ndef foo(a, b):\n    return a + b\n```"
    assert local_models.extract_function(reply, "foo") == "def foo(a, b):\n    return a + b\n"


def test_language_hint_on_the_fence_is_not_treated_as_code():
    # ```py as well as ```python must open a block without leaking the hint into the body.
    reply = "```py\ndef foo():\n    return 7\n```"
    assert local_models.extract_function(reply, "foo") == "def foo():\n    return 7\n"


def test_prose_without_a_definition_is_returned_trimmed():
    assert local_models.extract_function("  just prose, no code here  ", "foo") == "just prose, no code here\n"


def test_an_indented_definition_is_dedented_to_column_zero():
    # a model that answers inside a markdown list indents the whole function.
    reply = "    def foo(x):\n        return x\n    print('after')"
    assert local_models.extract_function(reply, "foo") == "def foo(x):\n    return x\n"


def test_a_statement_back_at_the_def_level_ends_the_function():
    reply = "def foo(x):\n    y = x\n    return y\nprint(foo(1))"
    assert local_models.extract_function(reply, "foo") == "def foo(x):\n    y = x\n    return y\n"


def test_a_second_top_level_function_ends_the_first():
    # a trailing sibling def (after a blank line) must not be folded into the first function.
    reply = "def foo(x):\n    return x\n\ndef other(y):\n    return y\n"
    assert local_models.extract_function(reply, "foo") == "def foo(x):\n    return x\n"


def test_blank_lines_inside_the_body_are_kept():
    reply = "def foo(x):\n    a = x\n\n    return a\ntrailing"
    assert local_models.extract_function(reply, "foo") == "def foo(x):\n    a = x\n\n    return a\n"


def test_windows_line_endings_are_normalised():
    assert local_models.extract_function("def foo():\r\n    return 1\r\n", "foo") == "def foo():\n    return 1\n"


def test_empty_answer_returns_a_single_newline():
    assert local_models.extract_function("", "foo") == "\n"


def test_result_always_ends_with_a_single_newline():
    for reply in ("def foo():\n    return 1", "no def at all", "```\ndef foo():\n    return 1\n```"):
        extracted = local_models.extract_function(reply, "foo")
        assert extracted.endswith("\n") and not extracted.endswith("\n\n")


# ---------------------------------------------------------------- friendly

def test_specific_loop_message_wins_over_the_generic_one():
    # "disallowed syntax: For" precedes the catch-all "disallowed syntax" in the table.
    assert local_models.friendly("disallowed syntax: For at line 2") == (
        "uses a for loop (loops aren't allowed; try sum(... for ...))")


def test_while_import_and_recursion_each_get_their_own_wording():
    assert "while loop" in local_models.friendly("disallowed syntax: While")
    assert "imports a module" in local_models.friendly("disallowed syntax: Import os")
    assert "recursion isn't allowed" in local_models.friendly("recursion detected")


def test_unknown_syntax_falls_back_to_the_generic_line():
    assert local_models.friendly("disallowed syntax: Lambda") == "uses Python syntax the tests don't allow"


def test_unsafe_call_and_attribute_access_are_explained():
    assert local_models.friendly("unsafe call to range").startswith("calls a function that isn't allowed")
    assert ".something" in local_models.friendly("attribute access on list")


def test_an_unrecognised_message_is_returned_unchanged():
    assert local_models.friendly("segmentation fault in cosmic rays") == "segmentation fault in cosmic rays"
