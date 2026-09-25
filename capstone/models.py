"""Generation backends for local Hugging Face and OpenAI API models."""

from typing import Dict, List, Optional


class BaseModel:
    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        raise NotImplementedError

    def metadata(self) -> Dict:
        """Return resolved runtime details needed to reproduce a model run."""
        return {}


class HFModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        load_in_4bit: bool = False,
        revision: Optional[str] = None,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_name = model_name
        self.requested_revision = revision
        self.load_in_4bit = load_in_4bit
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"device_map": "auto", "torch_dtype": "auto"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, **kwargs
        )
        self.model.eval()

    def metadata(self) -> Dict:
        devices = sorted({str(parameter.device) for parameter in self.model.parameters()})
        return {
            "backend": "hf",
            "model_name": self.model_name,
            "requested_revision": self.requested_revision,
            "resolved_commit_hash": getattr(self.model.config, "_commit_hash", None),
            "model_dtype": str(getattr(self.model, "dtype", "unknown")),
            "devices": devices,
            "load_in_4bit": self.load_in_4bit,
            "tokenizer_class": type(self.tokenizer).__name__,
        }

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        template_kwargs = {}
        if "qwen3" in getattr(self.model.config, "_name_or_path", "").lower():
            template_kwargs["enable_thinking"] = False

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **template_kwargs,
        )
        input_device = self.model.get_input_embeddings().weight.device
        inputs = {key: value.to(input_device) for key, value in inputs.items()}

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_length = inputs["input_ids"].shape[1]
        generated = output[0][prompt_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


class OpenAIModel(BaseModel):
    def __init__(self, model_name: str, temperature: Optional[float] = None):
        from openai import OpenAI

        self.client = OpenAI()
        self.model_name = model_name
        self.temperature = temperature

    def metadata(self) -> Dict:
        return {
            "backend": "openai",
            "model_name": self.model_name,
            "temperature": self.temperature,
            "note": "Hosted model snapshots are controlled by the API provider.",
        }

    def generate(self, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
        request = {
            "model": self.model_name,
            "input": messages,
            "max_output_tokens": max_new_tokens,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        response = self.client.responses.create(**request)
        return response.output_text.strip()


def build_model(
    backend: str,
    model_name: str,
    load_in_4bit: bool = False,
    revision: Optional[str] = None,
    openai_temperature: Optional[float] = None,
) -> BaseModel:
    if backend == "hf":
        return HFModel(
            model_name, load_in_4bit=load_in_4bit, revision=revision
        )
    if backend == "openai":
        return OpenAIModel(model_name, temperature=openai_temperature)
    raise ValueError(f"Unknown backend: {backend}")
