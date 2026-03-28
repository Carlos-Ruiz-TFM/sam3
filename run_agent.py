import os
import torch
from functools import partial

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.inference_mode().__enter__()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.agent.client_llm import send_generate_request as send_generate_request_orig
from sam3.agent.client_sam3 import call_sam_service as call_sam_service_orig
from sam3.agent.inference import run_single_image_inference

print("Loading SAM 3 model and processor...")
sam3_root = os.path.dirname(sam3.__file__)
bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

model = build_sam3_image_model(bpe_path=bpe_path, backend="ultralytics") # requires sam3.pt
processor = model
processor.set_confidence_threshold(0.5)

print("Configuring local MLLM...")

LLM_SERVER_URL = "http://192.168.0.27:1234/v1" 

llm_config = {
    "provider": "openai",
    "model": "qwen3-vl-8b", 
    "api_key": "lm-studio",
    "base_url": LLM_SERVER_URL,
    "name": "lmstudio_qwen3_8b_local"
}

send_generate_request = partial(
    send_generate_request_orig, 
    server_url=LLM_SERVER_URL, 
    model=llm_config["model"], 
    api_key=llm_config["api_key"]
)

call_sam_service = partial(
    call_sam_service_orig, 
    sam3_processor=processor
)

def main():
    image_path = os.path.abspath("assets/images/carlos.jpg")
    prompt = "Entire sidewalk including all steppable elements. Sewers must be included." # Tu instrucción compleja
    
    print(f"\nStarting agent reasoning for: '{prompt}'")
    
    output_image_path = run_single_image_inference(
        image_path,
        prompt,
        llm_config,
        send_generate_request,
        call_sam_service,
        debug=True,
        output_dir="agent_output"
    )

    if output_image_path:
        print(f"\n✅ Success! The agent has finished. Mask saved to: {output_image_path}")
    else:
        print("\nThe agent could not find the object or got stuck.")

if __name__ == "__main__":
    main()