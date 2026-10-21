"""Load BFCLv3 multi-turn-base first turns as (prompt, gold answer) examples.

Data comes from the GTTA repo (see data/bfcl_v3/SOURCE_COMMIT), including its own
val/test split: we tune on val (40) and report on test (160).
"""
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from . import prompts

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "bfcl_v3"

# BFCL constants/eval_config.py: MULTI_TURN_FUNC_DOC_FILE_MAPPING
CLASS_TO_DOC = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}

SPLIT_FILES = {
    "val": "BFCL_v3_multi_turn_base_val.json",
    "test": "BFCL_v3_multi_turn_base_test.json",
}


@dataclass
class Example:
    id: str
    split: str
    involved_classes: list
    user_request: str
    gold_calls: list
    prompt: str          # full chat text, ends with "<|im_start|>assistant\n"
    answer: str          # gold calls formatted as "[call1, call2]"

    def to_dict(self):
        return asdict(self)


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_functions(classes):
    functions = []
    for cls in classes:
        functions.extend(_read_jsonl(DATA_DIR / "multi_turn_func_doc" / CLASS_TO_DOC[cls]))
    return functions


def _dynamics_prompt(functions, request, task, dynamics):
    """DG variant: their ENV_DYNAMICS_SYSTEM_PROMPT with rules for the task's environments (see sa/dg.py)."""
    from . import dg  # local import: dg pulls in the vendored BFCL code
    system = dg.system_message(functions, "dynamics", dg.dynamics_string(task, dynamics))
    return prompts.format_chat([system, {"role": "user", "content": request}])


def load_split(split, dynamics=None):
    """dynamics: optional {class name: [rules]} → prompts use the DG system prompt with those rules."""
    tasks = _read_jsonl(DATA_DIR / SPLIT_FILES[split])
    answers = {a["id"]: a for a in _read_jsonl(DATA_DIR / "possible_answer" / SPLIT_FILES[split])}
    examples = []
    for task in tasks:
        first_turn = task["question"][0]
        assert len(first_turn) == 1 and first_turn[0]["role"] == "user", task["id"]
        request = first_turn[0]["content"]
        calls = answers[task["id"]]["ground_truth"][0]
        functions = load_functions(task["involved_classes"])
        examples.append(Example(
            id=task["id"],
            split=split,
            involved_classes=task["involved_classes"],
            user_request=request,
            gold_calls=calls,
            prompt=(prompts.build_prompt(functions, request) if dynamics is None
                    else _dynamics_prompt(functions, request, task, dynamics)),
            answer=prompts.format_answer(calls),
        ))
    return examples


def load_all():
    return load_split("val") + load_split("test")
