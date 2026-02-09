import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from typing import Union, Optional

from cs336_basics.BPE_Tokenizer import BPE_Tokenizer
from cs336_basics.Transformer.transformer_lm import TransformerLM
from cs336_basics.Trainer.Checkpointing import load_checkpoint_config, load_checkpoint_model


class Generator:
    def __init__(
        self,
        tokenizer: BPE_Tokenizer,
        model: Union[TransformerLM, str],
        device: Optional[torch.device] = None,
        strict: bool = True,
    ):
        self.tokenizer = tokenizer

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # ------------------------------------------------------------
        # Case 1: user passed checkpoint path
        # ------------------------------------------------------------
        if isinstance(model, (str, os.PathLike)):
            ckpt_path = str(model)

            config = load_checkpoint_config(ckpt_path)

            self.model = TransformerLM(
                vocab_size = config["vocab_size"],
                context_length = config["context_length"],
                num_layers = config["num_layers"],
                d_model = config["d_model"],
                num_heads = config["num_heads"],
                d_ff = config.get("d_ff", None),
                rope_theta = config.get("rope_theta", None),
                eps = config.get("eps", 1e-5),
                device = self.device,
            )

            load_checkpoint_model(
                src = ckpt_path,
                model = self.model,
                strict = strict,
            )

        # ------------------------------------------------------------
        # Case 2: user passed model object
        # ------------------------------------------------------------
        elif isinstance(model, TransformerLM):
            self.model = model.to(self.device)

        else:
            raise TypeError(f"model must be TransformerLM or checkpoint path str, got {type(model)}")

        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
    ) -> str:
        # encode prompt -> List[int]
        prompt_ids = self.tokenizer.encode(prompt)

        # generate token ids
        generated_ids = self.model.generate(
            prompt_token_ids = prompt_ids,
            max_new_tokens = max_new_tokens,
            temperature = temperature,
            top_p = top_p,
            eos_token_id = eos_token_id,
        )

        # decode back to string
        return self.tokenizer.decode(generated_ids)
    

if __name__ == "__main__":

    # tokenizer = BPE_Tokenizer.from_files(
    #     vocab_path = "cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-train-vocab.pkl",
    #     merges_path = "cs336_basics/BPE_Tokenizer/tests/TinyStoriesV2-GPT4-train-merges.pkl",
    #     special_tokens = ["<|endoftext|>"]
    # )
    # generator = Generator(
    #     tokenizer = tokenizer,
    #     model = "scripts/checkpoints/train_on_gpu_TinyStories_20260209_011538.pt"
    # )

    # prompt = "Once upon a time, there was a man called Snape. He was a professor at a magic school."
    # generated_text = generator.generate(
    #     prompt = prompt, 
    #     max_new_tokens = 256, 
    #     temperature = 0.8, 
    #     top_p = 0.95,
    #     eos_token_id = tokenizer.special_token_bytes[b'<|endoftext|>']
    # )
    # print(f"TinyStories:\n{generated_text}\n")

    tokenizer = BPE_Tokenizer.from_files(
        vocab_path = "cs336_basics/BPE_Tokenizer/tests/owt_train-vocab.pkl",
        merges_path = "cs336_basics/BPE_Tokenizer/tests/owt_train-merges.pkl",
        special_tokens = ["<|endoftext|>"]
    )
    generator = Generator(
        tokenizer = tokenizer,
        model = "scripts/checkpoints/train_on_gpu_OpenWebText_20260209_085954.pt"
    )

    prompt = "I have a dream, that one day this nation will"
    generated_text = generator.generate(
        prompt = prompt, 
        max_new_tokens = 128, 
        temperature = 0.9, 
        top_p = 0.9,
        eos_token_id = tokenizer.special_token_bytes[b'<|endoftext|>']
    )
    print(f"OpenWebText:\n{generated_text}\n")