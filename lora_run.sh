accelerate launch --mixed_precision="fp16"  train_lora.py \
  --pretrained_model_name_or_path='CompVis/stable-diffusion-v1-4' \
  --dataset_name=None \
  --dataloader_num_workers=8 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --max_train_steps=15000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="cosine" --lr_warmup_steps=0 \
  --output_dir=./lora_output \
  --report_to=wandb \
  --checkpointing_steps=3000 \
  --validation_prompts=None