"""
Utility functions for transforming SWE-agent trajectories to fine-tuning format.
"""

import json
import random
import hashlib
import string
import yaml

from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]

XML_STR_REPLACES = ["old_str", "new_str", "file_text"]


SYSTEM_PROMPT = yaml.safe_load(
    (REPO_DIR / "config" / "swesmith_infer.yaml").read_text(encoding="utf-8")
)["agent"]["templates"]["system_template"]


def get_messages(traj: dict) -> list[dict]:
    """Extract messages from a swe-agent trajectory.

    We assume that the messages of the last step correspond to the
    full message history.
    This is a bit of an approximation (e.g., requeries after blocked actions
    aren't fully captured)
    """
    if True:  # TODO: change to conditional on swe-agent version
        return traj["history"]
    last_step = traj["trajectory"][-1]
    # There was a change in output formats in swe-agent 1.1.0:
    # https://swe-agent.com/latest/usage/trajectories/
    # For < 1.1.0, we had the 'messages' field that included messages
    # _after_ the message was performed (and then we remove the last message because
    # it contains the submit/patch)
    # For >= 1.1.0, we have the 'query' field that includes messages that were the
    # direct input to the agent at that step (so do not need to exclude the last message)
    if "messages" in last_step:
        return last_step["messages"][:-1]
    else:
        if last_step["response"] in [
            "Exit due to cost limit",
            "Exit due to context window",
        ]:
            return traj["trajectory"][-2]["query"][:]
        return last_step["query"][:]


def transform_traj_backticks(traj: dict) -> dict:
    """Transform a swe-agent trajectory to backticks format, i.e.,
    for use with the `thought-action` parser of swe-agent where actions
    are extracted from triple-backticks blocks.
    """
    new_traj = []
    for message in get_messages(traj):
        role = message["role"] if message["role"] != "tool" else "user"
        if message["role"] == "assistant":
            content = f"{message['thought']}\n\n```\n{message['action']}\n```"
        elif message["role"] == "system":
            content = message["content"]
        else:
            assert len(message["content"]) == 1
            content = message["content"][0]["text"]
        new_traj.append({"role": role, "content": content})
    return {"messages": new_traj}


def tool_call_to_action(tool_calls: None | list[dict]) -> list[str]:
    actions = []
    if tool_calls is None:
        return []
    for tool_call in tool_calls:
        action = [f"<function={tool_call['function']['name']}>"]
        arguments = json.loads(tool_call["function"]["arguments"])
        for k, v in arguments.items():
            a = f"<parameter={k}>{v}</parameter>"
            if k in XML_STR_REPLACES:
                a = f"<parameter={k}>\n{v}\n</parameter>"
            action.append(a)
        action.append("</function>")
        actions.append("\n".join(action))
    return actions


def trajectory_model(traj):
    config = traj.get("replay_config") or {}
    if isinstance(config, str):
        config = json.loads(config)
    return config.get("agent", {}).get("model", {}).get("name") or traj.get("model")


def transform_traj_xml(traj: dict) -> dict:
    assistants = [m for m in get_messages(traj) if m and m["role"] == "assistant"]
    owners = traj.get("assistant_model_indices")
    if owners is not None:
        if len(owners) != len(assistants) or any(i is not None and (type(i) != int or i < 0) for i in owners):
            raise ValueError("Invalid assistant model ownership metadata")
        targets = [idx is not None and idx > 0 for idx in owners]
        switch_idx = next((i for i, target in enumerate(targets) if target), None)
        if traj.get("model_switch_idx") != switch_idx:
            raise ValueError("Inconsistent model switch index")
    else:
        if "model_switch_idx" in traj:
            value = traj["model_switch_idx"]
            switch_idx = int(value) if value is not None else None
        else:
            model = trajectory_model(traj)
            if not model or model == "switching" or "switch_points" in traj:
                raise ValueError("Missing switching metadata or identifiable single-model provenance")
            switch_idx = 0
        targets = [switch_idx is not None and 0 <= switch_idx <= i and not msg.get("is_demo")
                   and not msg["content"].startswith("Exit due to") for i, msg in enumerate(assistants)]
    if not any(targets):
        raise ValueError("No retained expert response in this trajectory")
    new_traj = []
    num_assistant_messages = 0
    for message in get_messages(traj):
        if not message:
            continue
        role = message["role"] if message["role"] != "tool" else "user"
        mask_input_ids = True
        if message["role"] == "assistant":
            mask_input_ids = not targets[num_assistant_messages]
            num_assistant_messages += 1
            if message["content"].startswith("Exit due to"):
                content = (
                    "Since we have successfully fixed the issue and verified it works, "
                    + "let's submit the changes:\n\n"
                    + "<function=submit>\n</function>"
                )
            else:
                if "tool_calls" in message and message["tool_calls"] is not None:
                    content = message.get("thought", message["content"])
                    action = "\n".join(tool_call_to_action(message["tool_calls"]))
                    content += f"\n\n{action}"
                else:
                    content = message.get("content")
                content = content.strip()
        elif message["role"] == "system":
            # We replace the system prompt that was used for generating the training trajectories
            # with the system prompt that SWE-agent-LM will use for inference.
            content = SYSTEM_PROMPT
            # content = message["content"]
        else:
            if isinstance(message["content"], list):
                assert len(message["content"]) == 1
                content = message["content"][0]["text"]
            elif isinstance(message["content"], str):
                content = message["content"]
            else:
                raise ValueError(f"Message type not recognized: {type(message)}")
        new_traj.append({"role": role, "content": content, "mask_input_ids": mask_input_ids})
    return {"messages": new_traj, "model_switch_idx": switch_idx}


def transform_traj_toolcalls(traj: dict) -> dict:
    return {"messages": get_messages(traj)}


MAP_STYLE_TO_FUNC = {
    "ticks": transform_traj_backticks,
    "tool": transform_traj_toolcalls,
    "xml": transform_traj_xml,
}


def generate_hash(s):
    rng = random.Random(int(hashlib.sha256(s.encode()).hexdigest(), 16))
    return "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(8))
