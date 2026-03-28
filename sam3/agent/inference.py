# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import hashlib
import json
import os
import re

from sam3.agent.agent_core import agent_inference


def _safe_filename_component(text: str, max_length: int = 120) -> str:
    """Create a filesystem-safe, length-bounded filename component."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    if not normalized:
        normalized = "prompt"
    if len(normalized) <= max_length:
        return normalized

    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    truncated = normalized[: max_length - len(digest) - 1].rstrip("._-")
    return f"{truncated}_{digest}"


def run_single_image_inference(
    image_path,
    text_prompt,
    llm_config,
    send_generate_request,
    call_sam_service,
    output_dir="agent_output",
    debug=False,
):
    """Run inference on a single image with provided prompt"""

    llm_name = llm_config["name"]

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate output file names
    image_basename = os.path.splitext(os.path.basename(image_path))[0]
    prompt_for_filename = _safe_filename_component(text_prompt)

    base_filename = f"{image_basename}_{prompt_for_filename}_agent_{llm_name}"
    output_json_path = os.path.join(output_dir, f"{base_filename}_pred.json")
    output_image_path = os.path.join(output_dir, f"{base_filename}_pred.png")
    agent_history_path = os.path.join(output_dir, f"{base_filename}_history.json")

    # Check if output already exists and skip
    if os.path.exists(output_json_path):
        print(f"Output JSON {output_json_path} already exists. Skipping.")
        return output_image_path if os.path.exists(output_image_path) else output_json_path

    print(f"{'-' * 30} Starting SAM 3 Agent Session... {'-' * 30} ")
    agent_history, final_output_dict, rendered_final_output = agent_inference(
        image_path,
        text_prompt,
        send_generate_request=send_generate_request,
        call_sam_service=call_sam_service,
        output_dir=output_dir,
        debug=debug,
    )
    print(f"{'-' * 30} End of SAM 3 Agent Session... {'-' * 30} ")

    final_output_dict["text_prompt"] = text_prompt
    final_output_dict["image_path"] = image_path

    # Save outputs
    json.dump(final_output_dict, open(output_json_path, "w"), indent=4)
    json.dump(agent_history, open(agent_history_path, "w"), indent=4)
    rendered_final_output.save(output_image_path)

    print(f"\n✅ Successfully processed single image!")
    print(f"Output JSON: {output_json_path}")
    print(f"Output Image: {output_image_path}")
    print(f"Agent History: {agent_history_path}")
    return output_image_path
