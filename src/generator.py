class Generator:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        max_input_len: int = 3840,
    ):
        from pathlib import Path

        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = Path(model_name)
        if not model_path.is_dir() or not (model_path / "config.json").is_file():
            raise FileNotFoundError(f"Local model not found at {model_path}. Download it with ModelScope first.")
        model_name = str(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
        )
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.max_input_len = int(max_input_len)

    def prepare_prompt_parts(self, prefix: str, context: str, suffix: str, max_tokens: int | None = None):
        """Apply Qwen's chat template while preserving evidence boundaries.

        ReDeEP must analyse exactly the prompt used during generation.  The
        returned prefix/context/suffix therefore describe the formatted chat
        prompt, rather than the raw user instruction.
        """
        from .prompt_utils import fit_prompt_parts

        prefix, context, suffix = str(prefix), str(context), str(suffix)
        raw_prompt = prefix + context + suffix
        formatted_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": raw_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        context_offset = formatted_prompt.find(context)
        if context and context_offset < 0:
            raise ValueError("Qwen chat template did not preserve the evidence text")
        if context:
            formatted_prefix = formatted_prompt[:context_offset]
            formatted_suffix = formatted_prompt[context_offset + len(context) :]
        else:
            formatted_prefix, formatted_suffix = formatted_prompt, ""
        parts = fit_prompt_parts(
            self.tokenizer,
            formatted_prefix,
            context,
            formatted_suffix,
            self.max_input_len if max_tokens is None else int(max_tokens),
        )
        parts["chat_template"] = True
        return parts

    def generate_with_token_ids(self, prompt: str):
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_len,
        ).to(self.model.get_input_embeddings().weight.device)
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        import torch

        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        generated = output[0][inputs["input_ids"].shape[1] :]
        from .prompt_utils import normalize_generated_token_ids

        token_ids = normalize_generated_token_ids(
            generated,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id,
        )
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        return text, token_ids

    def generate_with_token_ids_batch(self, prompts):
        """Generate several prompts in one forward/generation batch.

        Batching is only a throughput optimization: each prompt remains an
        independent sample and the caller still invokes this once per hop.
        """
        import torch

        prompts = list(prompts)
        if not prompts:
            return []
        old_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_len,
        ).to(self.model.get_input_embeddings().weight.device)
        generation_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generation_kwargs["temperature"] = self.temperature
        with torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        input_width = inputs["input_ids"].shape[1]
        from .prompt_utils import normalize_generated_token_ids

        result = []
        for row in output:
            generated = row[input_width:]
            token_ids = normalize_generated_token_ids(
                generated, self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
            )
            result.append((self.tokenizer.decode(token_ids, skip_special_tokens=True).strip(), token_ids))
        self.tokenizer.padding_side = old_padding_side
        return result

    def generate(self, prompt: str) -> str:
        return self.generate_with_token_ids(prompt)[0]
