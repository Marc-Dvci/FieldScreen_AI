import sys
import os
import json
import torch
from safetensors.torch import load_file
import gguf
import re

def convert(lora_path, outfile):
    print(f"Converting {lora_path} to {outfile}")
    
    # Load config
    with open(os.path.join(lora_path, "adapter_config.json"), "r") as f:
        config = json.load(f)
    
    alpha = float(config.get("lora_alpha", 16))
    print(f"Source Alpha: {alpha}")

    # Load tensors
    state_dict = load_file(os.path.join(lora_path, "adapter_model.safetensors"), device="cpu")
    
    # Initialize GGUF writer
    gguf_writer = gguf.GGUFWriter(outfile, "gemma3") # Arch name usually gemma or llama, but for adapter it matters?
    # Actually, adapter type is independent of arch string in header?
    # convert_lora_to_gguf.py sets:
    # self.gguf_writer.add_type(gguf.GGUFType.ADAPTER)
    # self.gguf_writer.add_string(gguf.Keys.Adapter.TYPE, "lora")
    # self.gguf_writer.add_float32(gguf.Keys.Adapter.LORA_ALPHA, self.lora_alpha)
    
    # Note: older gguf package might use different constants?
    # But installed gguf 0.17 has GGUFWriter.
    
    # We need to add minimal KV for adapter
    # Check if add_type exists in installed gguf
    # If not, maybe it's automatic?
    # But usually we need to set specific KV pairs.
    
    # In older gguf, maybe we just set "general.type" = "adapter"? No.
    # Let's check if we can add custom KV.
    
    gguf_writer.add_string("general.type", "adapter")
    gguf_writer.add_string("adapter.type", "lora")
    gguf_writer.add_float32("adapter.lora.alpha", alpha)
    
    # Mapping
    # Standard Llama/Gemma mapping
    # model.layers.X -> blk.X
    # self_attn.q_proj -> attn_q
    # self_attn.k_proj -> attn_k
    # self_attn.v_proj -> attn_v
    # self_attn.o_proj -> attn_output
    # mlp.gate_proj -> ffn_gate
    # mlp.up_proj -> ffn_up
    # mlp.down_proj -> ffn_down
    # embed_tokens -> token_embd
    # norm -> output_norm
    # lm_head -> output
    
    skipped_vision = 0
    converted = 0

    for name, tensor in state_dict.items():
        if "base_model.model." in name:
            new_name = name.replace("base_model.model.", "")
        else:
            new_name = name

        # Skip vision tower tensors — llama.cpp LoRA doesn't support
        # vision encoder adapters yet. These tensors are only useful
        # for bf16/HuggingFace inference (not GGUF).
        if "vision_tower" in new_name:
            skipped_vision += 1
            if skipped_vision <= 4:
                print(f"Skipping vision tower tensor: {name}")
            elif skipped_vision == 5:
                print(f"  ... (suppressing further vision tower messages)")
            continue

        # Clean up suffix
        suffix = ""
        if "lora_A.weight" in new_name:
            suffix = ".lora_a"
            new_name = new_name.replace(".lora_A.weight", "")
        elif "lora_B.weight" in new_name:
            suffix = ".lora_b"
            new_name = new_name.replace(".lora_B.weight", "")
        elif "lora_embedding_A" in new_name:
            suffix = ".lora_a"
            new_name = new_name.replace(".lora_embedding_A", "")
        elif "lora_embedding_B" in new_name:
            suffix = ".lora_b"
            new_name = new_name.replace(".lora_embedding_B", "")
        else:
            # Maybe bias or norm?
            # Skip for now if unknown 
            print(f"Skipping unknown tensor type: {name}")
            continue

        # Map layer
        # Typical path after base_model.model. removal:
        #   model.language_model.model.layers.0.self_attn.q_proj
        # Strip prefixes until we reach "layers", "embed_tokens", etc.
        parts = new_name.split(".")

        # Strip leading "model." and "language_model." prefixes (may occur multiple times)
        while parts and parts[0] in ("model", "language_model"):
            parts = parts[1:]

        # Handle layers
        if parts[0] == "layers":
            # layers.0.self_attn.q_proj
            idx = parts[1]
            block = f"blk.{idx}"
            
            # Map component
            comp = parts[2:] # self_attn, q_proj
            
            if comp[0] == "self_attn":
                if comp[1] == "q_proj": short = "attn_q"
                elif comp[1] == "k_proj": short = "attn_k"
                elif comp[1] == "v_proj": short = "attn_v"
                elif comp[1] == "o_proj": short = "attn_output"
                else: short = f"attn_{comp[1]}"
            elif comp[0] == "mlp":
                if comp[1] == "gate_proj": short = "ffn_gate"
                elif comp[1] == "up_proj": short = "ffn_up"
                elif comp[1] == "down_proj": short = "ffn_down"
                else: short = f"ffn_{comp[1]}"
            else:
                 # Post attention norm?
                 if comp[0] == "input_layernorm": short = "attn_norm"
                 elif comp[0] == "post_attention_layernorm": short = "ffn_norm"
                 else: short = "_".join(comp)
                 
            final_name = f"{block}.{short}.weight{suffix}"
            
        elif parts[0] == "embed_tokens":
             final_name = f"token_embd.weight{suffix}"
        elif parts[0] == "norm":
             final_name = f"output_norm.weight{suffix}"
        elif parts[0] == "lm_head":
             final_name = f"output.weight{suffix}"
        else:
             print(f"Warning: Could not map {new_name}")
             final_name = new_name + ".weight" + suffix

        # Convert simple types
        data = tensor.to(torch.float32).numpy()

        print(f"Writing {name} -> {final_name} | {data.shape}")
        gguf_writer.add_tensor(final_name, data)
        converted += 1

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    print(f"\nDone. Converted {converted} tensors to GGUF.")
    if skipped_vision > 0:
        print(f"Skipped {skipped_vision} vision tower tensors (not supported in GGUF LoRA).")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python simple_lora_converter.py <lora_dir> <outfile>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
