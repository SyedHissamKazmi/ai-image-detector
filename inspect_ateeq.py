from transformers import AutoConfig

MODEL_NAME = "Ateeqq/ai-vs-human-image-detector"
config = AutoConfig.from_pretrained(MODEL_NAME)
print("id2label:")
for idx, label in config.id2label.items():
    print(f"  {idx} -> {label}")
print("label2id:")
for label, idx in config.label2id.items():
    print(f"  {label} -> {idx}")