from gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from transformers import AutoTokenizer
import json
import glob
from safetensors import safe_open
from typing import Tuple
import os

def load_hf_model(model_path: str, device: str) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
    assert tokenizer.padding_side == "right", "The tokenizer padding side must be right"

    # Find all the *.safetensors files in the model directory
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))

    # ... and load them one by one in the tensors dictionary
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    
    # Load the model configuration
    with open(os.path.join(model_path, "config.json"), "r") as f:
        model_config = json.load(f)
        config = PaliGemmaConfig(**model_config)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)

    # Load the sate dict of the model
    model.load_state_dict(tensors, strict=False)

    # Tie the weights of the token and position embeddings
    model.tie_weights()

    return (model, tokenizer)

    

