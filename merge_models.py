import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import gc

base_model_name = "Qwen/Qwen3-14B"

adapter_path = "./qwen_finetuned-bf16"

merged_model_path = "./qwen_finetuned_merged"

print(f"Загрузка базовой модели: {base_model_name}")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

print(f"Загрузка токенизатора для базовой модели")
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

print(f"Загрузка LoRA-адаптера из: {adapter_path}")
# Загружаем модель Peft, которая объединяет базовую модель и адаптер
model = PeftModel.from_pretrained(base_model, adapter_path)

print("Произвожу слияние весов...")
# Сливаем веса адаптера с базовой моделью и выгружаем адаптер
model = model.merge_and_unload()
print("Слияние завершено.")

# Создаем директорию для сохранения, если её нет
os.makedirs(merged_model_path, exist_ok=True)

print(f"Сохранение объединенной модели в: {merged_model_path}")
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
print("Модель и токенизатор успешно сохранены.")

# Очистка памяти
del model
del base_model
gc.collect()
torch.cuda.empty_cache()