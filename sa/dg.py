"""Dynamics Grounding (DG) on BFCLv3, faithful to GTTA's released BFCL code and prompts.

Pieces (all driven by a batched `generate(prompts) -> texts` function, e.g. vLLM):
1. Exploration: their 80 goals run against BFCL's executable simulators with EXPLORATION_SYSTEM_PROMPT
   (one call at a time; stop on "###STOP" after step 0; max 20 steps).          (base_handler.py)
2. Extraction: each exploration log → JSON rules with prompts/env_dynamics.txt.  (extract_dynamics.ipynb)
3. Filtering: per environment, all rules → filtered rules with filter_env_dynamics.txt.
4. Evaluation: multi-turn BFCL tasks with DEFAULT_SYSTEM_PROMPT (none) or ENV_DYNAMICS_SYSTEM_PROMPT (DG),
   scored with BFCL's multi_turn_checker.                                       (eval_runner.py)
Original models: GPT-4.1 (explore/extract), GPT-4o (extract in notebook), o3 (filter). Ours: the task model itself.
"""
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = ROOT / "third_party"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

from bfcl.constants.env_dynamics_prompt import ENV_DYNAMICS_SYSTEM_PROMPT  # noqa: E402
from bfcl.constants.exploration_prompts import EXPLORATION_SYSTEM_PROMPT  # noqa: E402
from bfcl.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker  # noqa: E402
from bfcl.eval_checker.multi_turn_eval.multi_turn_utils import (  # noqa: E402
    execute_multi_turn_func_call, is_empty_execute_response)
from bfcl.parsing import default_decode_execute_prompting  # noqa: E402

from . import data, prompts  # noqa: E402

DG_DIR = data.DATA_DIR / "dg"
MAX_STEPS = 20  # BFCL MAXIMUM_STEP_LIMIT
CLASS_TO_ENV = {
    "TravelAPI": "travel_booking", "GorillaFileSystem": "gorilla_file_system", "TradingBot": "trading_bot",
    "MessageAPI": "message_api", "TwitterAPI": "posting_api", "MathAPI": "math_api",
    "TicketAPI": "ticket_api", "VehicleControlAPI": "vehicle_control",
}
ENV_TO_CLASS = {v: k for k, v in CLASS_TO_ENV.items()}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_chat(messages):
    """BFCL QwenHandler prompting format (roles: system, user, assistant, tool)."""
    text = ""
    for m in messages:
        text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    return text + "<|im_start|>assistant\n"


def system_message(functions, kind, dynamics_str=""):
    funcs = prompts.preprocess_functions(functions)
    if kind == "default":
        content = prompts.DEFAULT_SYSTEM_PROMPT.format(functions=funcs)
    elif kind == "explore":
        content = EXPLORATION_SYSTEM_PROMPT.format(functions=funcs)
    elif kind == "dynamics":
        content = ENV_DYNAMICS_SYSTEM_PROMPT.format(functions=funcs, environmental_dynamics=dynamics_str)
    else:
        raise ValueError(kind)
    return {"role": "system", "content": content}


def dynamics_string(entry, dynamics):
    """Their base_handler: rules of every environment in initial_config, one JSON object per line."""
    out = ""
    for env in entry["initial_config"].keys():
        if env in dynamics:
            out += "\n".join(json.dumps(item) for item in dynamics[env])
    return out


class Episode:
    """One BFCL task run step by step; mirrors base_handler.inference_multi_turn_prompting(_exploration)."""

    def __init__(self, entry, system, namespace, explore=False):
        self.entry = entry
        self.explore = explore
        self.namespace = namespace          # BFCL keeps simulator instances in globals by this name
        self.messages = [system]
        self.turns = entry["question"]
        self.turn_idx = -1
        self.results = []                   # per turn: list of raw model responses (one per step)
        self.log = []                       # per turn: list of steps {assistant, tool}
        self.force_quit = False
        self.done = False
        self._next_turn()

    def _next_turn(self):
        self.turn_idx += 1
        if self.turn_idx >= len(self.turns) or self.force_quit:
            self.done = True
            return
        self.messages.extend(self.turns[self.turn_idx])
        self.results.append([])
        self.log.append([])
        self.step = 0

    def prompt(self):
        return format_chat(self.messages)

    @staticmethod
    def _first_valid_call_line(text):
        """Exploration only: the prompt asks for one call at a time, but Qwen often writes several
        "[call]" lines or "[call]\\n###STOP". Execute the first line that BFCL can decode.
        (Deviation from GTTA, whose GPT-4.1 explorer presumably followed the format; see decisions.md.)"""
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("["):
                try:
                    if not is_empty_execute_response(default_decode_execute_prompting(line)):
                        return line
                except Exception:  # noqa: BLE001
                    continue
        return text

    def consume(self, text):
        """Process one model response; returns nothing, advances state."""
        if self.explore and self.step > 0 and "###STOP" in text:
            self._next_turn()
            return
        if self.explore:
            text = self._first_valid_call_line(text)
        self.messages.append({"role": "assistant", "content": text})
        self.results[-1].append(text)
        step_log = {"assistant": text}
        self.log[-1].append(step_log)
        try:
            decoded = default_decode_execute_prompting(text)
            if is_empty_execute_response(decoded):
                self._next_turn()
                return
        except Exception as e:  # noqa: BLE001  (BFCL: any decode error ends the turn)
            step_log["decode_error"] = str(e)
            self._next_turn()
            return
        results, _ = execute_multi_turn_func_call(
            decoded, self.entry["initial_config"], self.entry["involved_classes"],
            self.namespace, self.entry["id"], long_context=False, is_evaL_run=False)
        step_log["tool"] = results
        for res, call in zip(results, decoded):
            self.messages.append({"role": "tool", "name": call, "content": res})
        self.step += 1
        if self.step > MAX_STEPS:
            self.force_quit = True
            self._next_turn()


def run_episodes(generate, episodes, on_round=None):
    """Step-synchronous batching: every active episode advances one step per round."""
    rnd = 0
    while True:
        active = [e for e in episodes if not e.done]
        if not active:
            return episodes
        texts = generate([e.prompt() for e in active])
        for e, t in zip(active, texts):
            e.consume(t)
        rnd += 1
        if on_round:
            on_round(rnd, len(active))


# ---------- exploration → rules ----------
def exploration_episodes(namespace):
    eps = []
    for entry in read_jsonl(DG_DIR / "multi_turn_exploration.json"):
        functions = data.load_functions(entry["involved_classes"])
        eps.append(Episode(entry, system_message(functions, "explore"), namespace, explore=True))
    return eps


def format_interaction_log(ep):
    """extract_dynamics.ipynb format_interaction_log: goal + per-step assistant text and first tool result."""
    env = CLASS_TO_ENV[ep.entry["involved_classes"][0]]
    goal = ep.entry["question"][0][0]["content"].replace("Goal: ", "").strip()
    out = "Environment:\n" + env + "\nGoal:\n" + goal + "\nInteraction Log:\n"
    for i, st in enumerate(ep.log[0] if ep.log else [], 1):
        out += f"Step {i}:\n"
        if "tool" not in st:
            continue  # erroneous step
        out += f"Assistant: {st['assistant']}\n"
        out += f"Environment: {st['tool'][0] if st['tool'] else ''}\n"
    return out


def extraction_prompt(ep):
    system = (DG_DIR / "env_dynamics.txt").read_text()
    return format_chat([{"role": "system", "content": system},
                        {"role": "user", "content": format_interaction_log(ep)}])


def parse_json_array(text):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    try:
        val = json.loads(m.group(0) if m else text)
    except Exception:  # noqa: BLE001
        return None
    return val if isinstance(val, list) else None


def filter_prompt(env, rules):
    system = (DG_DIR / "filter_env_dynamics.txt").read_text()
    return format_chat([{"role": "system", "content": system},
                        {"role": "user", "content": f"Environment: {env}\nDynamics: {json.dumps(rules)}"}])


# ---------- evaluation ----------
def eval_entries(split):
    tasks = read_jsonl(data.DATA_DIR / data.SPLIT_FILES[split])
    answers = {a["id"]: a for a in read_jsonl(data.DATA_DIR / "possible_answer" / data.SPLIT_FILES[split])}
    return [(t, answers[t["id"]]["ground_truth"]) for t in tasks]


def eval_episodes(entries, namespace, dynamics=None):
    eps = []
    for entry, _ in entries:
        functions = data.load_functions(entry["involved_classes"])
        if dynamics is None:
            system = system_message(functions, "default")
        else:
            system = system_message(functions, "dynamics", dynamics_string(entry, dynamics))
        eps.append(Episode(copy.deepcopy(entry), system, namespace))
    return eps


def score(ep, ground_truth, namespace):
    """eval_runner.multi_turn_runner: decode each step, drop empty/undecodable, then multi_turn_checker."""
    if len(ep.results) != len(ground_truth):
        return {"valid": False, "error_type": "multi_turn:force_terminated"}
    decoded_turns = []
    for turn in ep.results:
        dec = []
        for text in turn:
            try:
                d = default_decode_execute_prompting(text)
                if not is_empty_execute_response(d):
                    dec.append(d)
            except Exception:  # noqa: BLE001
                continue
        decoded_turns.append(dec)
    entry = ep.entry
    res = multi_turn_checker(decoded_turns, ground_truth, entry, entry["id"].rsplit("_", 1)[0], namespace)
    return {"valid": bool(res.get("valid")), "error_type": res.get("error_type")}
