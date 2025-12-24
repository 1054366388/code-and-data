import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from .config import cfg


class LLMEngine:
    def __init__(self):
        print(f"[System] Loading Tokenizer: {cfg.MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model(self):
        print(f"[System] Loading Model Weights: {cfg.MODEL_ID}")
        # Load Base Model
        model = AutoModelForCausalLM.from_pretrained(
            cfg.MODEL_ID,
            device_map="auto",
            torch_dtype=torch.float16
        )

        # Apply LoRA (Eq. 6)
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=cfg.LORA_R,
            lora_alpha=cfg.LORA_ALPHA,
            lora_dropout=cfg.LORA_DROPOUT,
            target_modules=cfg.TARGET_MODULES
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        return model

    def run_training(self, prompt_list):
        """
        Executes standard Hugging Face Trainer loop.
        """
        model = self.load_model()

        # Prepare Dataset
        dataset = Dataset.from_dict({"text": prompt_list})
        tokenized_data = dataset.map(
            lambda x: self.tokenizer(x["text"], truncation=True, max_length=512),
            batched=True
        )

        args = TrainingArguments(
            output_dir=cfg.OUTPUT_DIR,
            per_device_train_batch_size=cfg.BATCH_SIZE,
            gradient_accumulation_steps=cfg.GRADIENT_ACCUMULATION,
            num_train_epochs=cfg.NUM_EPOCHS,
            learning_rate=cfg.LEARNING_RATE,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch"
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=tokenized_data,
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        )

        print(">>> Starting LoRA Fine-Tuning...")
        trainer.train()
        print(">>> Training Complete. Saving adapter...")
        trainer.save_model(cfg.OUTPUT_DIR)