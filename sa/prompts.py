"""BFCLv3 prompt format, copied from the GTTA repo (commit 24ba304, BFCLv3/.../bfcl).

Sources:
- constants/default_prompts.py: DEFAULT_SYSTEM_PROMPT (also saved in data/bfcl_v3/default_prompts.py)
- model_handler/utils.py: system_prompt_pre_processing_chat_model, func_doc_language_specific_pre_processing
- model_handler/local_inference/qwen.py: QwenHandler._format_prompt (prompting mode, not tool-call mode)
"""
import copy

DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.

At each turn, you should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task.
"""

DEFAULT_SYSTEM_PROMPT = (
    DEFAULT_SYSTEM_PROMPT_WITHOUT_FUNC_DOC
    + """
Here is a list of functions in JSON format that you can invoke.\n{functions}\n
"""
)

PYTHON_HINT = " Note that the provided function is in Python 3 syntax."

# Qwen chat markers, as written by BFCL's QwenHandler.
USER_START = "<|im_start|>user"
TURN_END = "<|im_end|>"


def preprocess_functions(functions):
    """BFCL's func_doc_language_specific_pre_processing for Python categories."""
    functions = copy.deepcopy(functions)
    for item in functions:
        item["description"] = item["description"] + PYTHON_HINT
    return functions


def system_prompt(functions):
    """BFCL inserts the function list with str.format, i.e. Python's repr of a list of dicts."""
    return DEFAULT_SYSTEM_PROMPT.format(functions=preprocess_functions(functions))


def format_chat(messages):
    """BFCL QwenHandler._format_prompt: plain im_start/im_end markers + open assistant turn."""
    text = ""
    for message in messages:
        text += f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>\n"
    return text + "<|im_start|>assistant\n"


def build_prompt(functions, user_request):
    """First-turn prompt: system (instructions + functions) + the user's first request."""
    return format_chat([
        {"role": "system", "content": system_prompt(functions)},
        {"role": "user", "content": user_request},
    ])


def format_answer(calls):
    """Gold calls in the format the system prompt asks for: [call1, call2, ...]."""
    return "[" + ", ".join(calls) + "]"
