import torch
from dataclasses import dataclass
import numpy as np
from torch.nn.functional import cross_entropy
import pytorch_lightning as pl
from transformers import GPTJForCausalLM, LlamaForCausalLM, AdamW
from bitsandbytes.optim import AdamW8bit

from autograd_4bit import load_gptj_model_4bit_low_ram, load_llama_model_4bit_low_ram
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)

@dataclass
class FinetunerConfig():
    target_lora_modules: list
    lr: float = 1e-3
    batch_size: int = 1
    warmup_steps: int = 10
    num_epochs: int = 1
    adapter_dim: int = 2
    classification: bool = False
    lora_r: int = 2
    lora_alpha: int = 16
    lora_dropout: float = 0


class CommonFinetuner(pl.LightningModule):
    def __init__(self, model_name, fine_tuning_config, train_dataset, val_dataset=None):
        super().__init__()

        self.model_name = model_name
        self.config = fine_tuning_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.validation_step_outputs = []
        self.model = None
        self.squeeze_input = True

    def load(self):
        pass

    def forward(self, input_ids, attention_masks):
        assert self.model != None, 'Load model with load() method.'
        return self.model.forward(
                            input_ids=input_ids,
                            attention_mask=attention_masks
                            )

    def common_step(self, batch, batch_idx):
        input_ids, attention_masks, loss_mask = batch
        roll_dim = 2
        if self.squeeze_input:
            input_ids = input_ids.squeeze(-2)
            attention_masks = attention_masks.squeeze(-2)
            loss_mask = loss_mask.squeeze(-2)
            roll_dim = 1
        out = self(
                    input_ids=input_ids,
                    attention_masks=attention_masks
                    )
        logits = out.logits[loss_mask.roll(shifts=-1, dims=roll_dim)]
        completion_tok_ids = input_ids[loss_mask]
        loss = cross_entropy(logits, completion_tok_ids)
        preds = None
        if self.config.classification:
            preds = torch.argmax(logits, dim=1)
        return loss, preds, completion_tok_ids


    def training_step(self, batch, batch_idx):
        loss, preds, trues = self.common_step(batch, batch_idx)
        self.log('train_loss', loss)
        if self.config.classification:
            acc = round((preds == trues).sum().cpu().item() / preds.shape[0], 2)
            self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.common_step(batch, batch_idx)
        self.log('val_loss', loss)
        if self.config.classification:
            trues = torch.sum(preds == labels).cpu()
            total = len(labels)
            self.validation_step_outputs.append((loss, trues, total))
            return loss, trues, total
        self.validation_step_outputs.append((loss, None, None))
        return loss, None, None
    
    def on_validation_epoch_end(self):
        self.log('val_epoch_loss', np.mean([e[0].cpu() for e in self.validation_step_outputs]))
        if self.config.classification:
            self.log('val_total_epoch_samples',  np.sum([e[2] for e in self.validation_step_outputs]))
            self.log('val_epoch_accuracy', np.sum([e[1] for e in self.validation_step_outputs]) /  np.sum([e[2] for e in self.validation_step_outputs]))
        self.validation_step_outputs.clear()


    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset, 
                                                   batch_size=self.config.batch_size, 
                                                  shuffle=True)
        return train_loader
    
    def val_dataloader(self):
        if self.val_dataset:
            val_dataloader = torch.utils.data.DataLoader(self.val_dataset, 
                                                    batch_size=self.config.batch_size, 
                                                    shuffle=True)
            return val_dataloader
        
    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.config.lr)
        return optimizer


class LLaMA8bitFineTuner(CommonFinetuner):
    def __init__(self, model_name, fine_tuning_config, train_dataset, val_dataset=None):
        super().__init__(model_name, fine_tuning_config, train_dataset, val_dataset)

    def load(self):
        self.model = prepare_model_for_int8_training(LlamaForCausalLM.from_pretrained(self.model_name, load_in_8bit=True, device_map={'': 0}))
        
        lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_lora_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        self.model = get_peft_model(self.model, lora_config)

        self.model.config.use_cache = False

        old_state_dict = self.model.state_dict
        self.model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(self.model, type(self.model))


class LLaMA4bitFineTuner(CommonFinetuner):
    def __init__(self, model_name, checkpoint_path, fine_tuning_config, train_dataset, val_dataset=None):
        super().__init__(model_name, fine_tuning_config, train_dataset, val_dataset)
        self.checkpoint_path = checkpoint_path

    def load(self):
        self.model = load_llama_model_4bit_low_ram(self.model_name, self.checkpoint_path, half=True)
        lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_lora_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        self.model = get_peft_model(self.model, lora_config)
        self.model.config.use_cache = False
        old_state_dict = self.model.state_dict
        self.model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(self.model, type(self.model))



class GPTJ8bitFineTuner(CommonFinetuner):
        def __init__(self, model_name, fine_tuning_config, train_dataset, val_dataset=None):
            super().__init__(model_name, fine_tuning_config, train_dataset, val_dataset)
            self.squeeze_input = False

        def load(self):    
            self.model =  prepare_model_for_int8_training(GPTJForCausalLM.from_pretrained(self.model_name, load_in_8bit=True, device_map={'': 0}))
            lora_config = LoraConfig(
                    r=self.config.lora_r,
                    lora_alpha=self.config.lora_alpha,
                    target_modules=self.config.target_lora_modules,
                    lora_dropout=self.config.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
            
            self.model = get_peft_model(self.model, lora_config)

            self.model.config.use_cache = False

            old_state_dict = self.model.state_dict
            self.model.state_dict = (
                lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
            ).__get__(self.model, type(self.model))



class GPTJ4bitFineTuner(CommonFinetuner):
    def __init__(self, model_name, checkpoint_path, fine_tuning_config, train_dataset, val_dataset=None):
        super().__init__(model_name, fine_tuning_config, train_dataset, val_dataset)
        self.checkpoint_path = checkpoint_path
        self.squeeze_input = False
        
    def load(self):
        self.model = load_gptj_model_4bit_low_ram(self.model_name, self.checkpoint_path, half=True)
        lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_lora_modules,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
        self.model = get_peft_model(self.model, lora_config)
        self.model.config.use_cache = False
        old_state_dict = self.model.state_dict
        self.model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(self.model, type(self.model))