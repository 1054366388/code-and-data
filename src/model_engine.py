import torch
import math
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from .config import cfg


class ModelLossCallback(TrainerCallback):
    """Callback to extract the last evaluation or training loss."""
    def __init__(self):
        super().__init__()
        self.final_loss = float('inf')

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            self.final_loss = logs["loss"]

class LLMEngine:
    def __init__(self):
        print(f"[System] Loading Tokenizer: {cfg.MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self, lora_r, lora_alpha, lora_dropout):
        print(f"[System] Loading Model Weights: {cfg.MODEL_ID}")
        # Load Base Model
        model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16
        )

        # Apply LoRA configurations dynamically
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=cfg.TARGET_MODULES
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    def run_training(self, prompt_list, learning_rate=cfg.LEARNING_RATE, 
                     batch_size=cfg.BATCH_SIZE, lora_r=cfg.LORA_R, 
                     lora_alpha=cfg.LORA_ALPHA, lora_dropout=cfg.LORA_DROPOUT):
        """
        Executes standard Hugging Face Trainer loop with dynamic hyperparameters and returns the final loss.
        """
        model = self.load_model(lora_r, lora_alpha, lora_dropout)

        # Prepare Dataset
        dataset = Dataset.from_dict({"text": prompt_list})
        tokenized_data = dataset.map(
            lambda x: self.tokenizer(x["text"], truncation=True, max_length=512),
            batched=True
        )

        args = TrainingArguments(
            output_dir=cfg.OUTPUT_DIR,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION,
            num_train_epochs=cfg.NUM_EPOCHS,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch",
            seed=cfg.RANDOM_SEED
        )

        loss_callback = ModelLossCallback()

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_data,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False),
            callbacks=[loss_callback]
        )

        print(">>> Starting LoRA Fine-Tuning...")
        trainer.train()
        print(">>> Training Complete. Saving adapter...")
        trainer.save_model(cfg.OUTPUT_DIR)
        
        # Free up memory
        del model
        torch.cuda.empty_cache()

        # Return final loss or a fallback large number if NaN
        return loss_callback.final_loss if not math.isnan(loss_callback.final_loss) else 999.0