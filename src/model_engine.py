import math
import os
import re
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

from .config import cfg


class ModelLossCallback(TrainerCallback):
    """Callback to track the final training loss across the run."""

    def __init__(self):
        super().__init__()
        self.final_loss: float = float("inf")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.final_loss = logs["loss"]


class LLMEngine:
    """
    Wraps Llama-3-8B-Instruct with LoRA for fine-tuning and inference.
    """

    def __init__(self):
        print(f"[System] Loading Tokenizer: {cfg.MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID)

        # Add a dedicated pad token instead of reusing eos_token.
        # Reusing eos_token causes the model to learn to ignore the end-of-
        # sequence signal, which hurts generation quality at inference time.
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self._model_needs_resize = True  # flag consumed by load_model

    # ------------------------------------------------------------------ #
    # Model loading
    # ------------------------------------------------------------------ #
    def load_model(
        self,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
    ) -> AutoModelForCausalLM:
        """Load the base model and wrap it with a fresh LoRA adapter."""
        print(f"[System] Loading Model Weights: {cfg.MODEL_ID}")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Resize embeddings if we added a pad token
        if self._model_needs_resize:
            model.resize_token_embeddings(len(self.tokenizer))

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=cfg.TARGET_MODULES,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def run_training(
        self,
        prompt_list: List[str],
        learning_rate: float = cfg.LEARNING_RATE,
        batch_size: int = cfg.BATCH_SIZE,
        lora_r: int = cfg.LORA_R,
        lora_alpha: int = cfg.LORA_ALPHA,
        lora_dropout: float = cfg.LORA_DROPOUT,
        trial_number: Optional[int] = None,
    ) -> float:
        """
        Fine-tune the model on *prompt_list* using the HuggingFace Trainer.

        Returns the final training loss (or 999.0 on failure).
        """
        model = self.load_model(lora_r, lora_alpha, lora_dropout)

        # Tokenise ----------------------------------------------------------
        dataset = Dataset.from_dict({"text": prompt_list})
        tokenised = dataset.map(
            lambda x: self.tokenizer(
                x["text"],
                truncation=True,
                max_length=cfg.MAX_SEQ_LENGTH,
            ),
            batched=True,
        )

        # Output directory per trial so checkpoints are not overwritten -----
        if trial_number is not None:
            trial_output = os.path.join(cfg.OUTPUT_DIR, f"trial_{trial_number}")
        else:
            trial_output = cfg.OUTPUT_DIR
        os.makedirs(trial_output, exist_ok=True)

        args = TrainingArguments(
            output_dir=trial_output,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION,
            num_train_epochs=cfg.NUM_EPOCHS,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            seed=cfg.RANDOM_SEED,
        )

        loss_cb = ModelLossCallback()

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenised,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            callbacks=[loss_cb],
        )

        print(">>> Starting LoRA Fine-Tuning...")
        trainer.train()
        print(">>> Training Complete. Saving adapter...")
        trainer.save_model(trial_output)

        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()

        final = loss_cb.final_loss
        return final if not math.isnan(final) else 999.0

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def run_inference(
        self,
        prompt_list: List[str],
        class_names: List[str],
        adapter_path: str,
        lora_r: int = cfg.LORA_R,
        lora_alpha: int = cfg.LORA_ALPHA,
        lora_dropout: float = cfg.LORA_DROPOUT,
        max_new_tokens: int = 256,
    ) -> List[str]:
        """
        Generate predictions for a list of (label-free) prompts.

        The method loads the base model, attaches the saved LoRA adapter,
        generates text, and parses each response to extract the predicted
        class label from ``class_names``.

        Returns a list of predicted class-name strings (one per prompt).
        """
        print(f"[Inference] Loading base model and adapter from {adapter_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        base_model.resize_token_embeddings(len(self.tokenizer))
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()

        predictions: List[str] = []
        for prompt in prompt_list:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=cfg.MAX_SEQ_LENGTH,
            ).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                )

            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            pred_label = self._parse_prediction(generated, class_names)
            predictions.append(pred_label)

        del model, base_model
        torch.cuda.empty_cache()
        return predictions

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_prediction(text: str, class_names: List[str]) -> str:
        """
        Extract the predicted class label from generated text.

        Strategy:
        1. Look for "[Diagnosis]: <label>" pattern.
        2. Fallback: return the first class name that appears in the text.
        3. If nothing matches return "UNKNOWN".
        """
        # Try structured pattern first
        match = re.search(r"\[Diagnosis\]\s*:\s*(.+)", text)
        if match:
            snippet = match.group(1).strip()
            for name in class_names:
                if name.lower() in snippet.lower():
                    return name

        # Fallback: first mention of any class name (case-insensitive)
        text_lower = text.lower()
        for name in class_names:
            if name.lower() in text_lower:
                return name

        return "UNKNOWN"
