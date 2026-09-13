class Generator:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        max_input_len: int = 3840,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        self.max_new_tokens, self.temperature = max_new_tokens, temperature
        self.max_input_len = int(max_input_len)

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

    def generate(self, prompt: str) -> str:
        return self.generate_with_token_ids(prompt)[0]
