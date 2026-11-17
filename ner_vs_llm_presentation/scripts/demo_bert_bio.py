
import torch
from transformers import pipeline

# Input sentence
text = "Apple CEO Tim Cook announced plans to open a $200 million research center in Istanbul."

# Load pretrained BERT NER model
ner = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",
    aggregation_strategy="first",
    device="mps" if torch.backends.mps.is_available() else "cpu",
)

tokenizer = ner.tokenizer
model = ner.model

# Tokenization
encoded = tokenizer(
    text,
    return_tensors="pt",
    return_offsets_mapping=True
)

offsets = encoded.pop("offset_mapping")[0].tolist()

tokens = tokenizer.convert_ids_to_tokens(
    encoded["input_ids"][0].tolist()
)

inputs = {
    k: v.to(next(model.parameters()).device)
    for k, v in encoded.items()
}

# BIO predictions
with torch.inference_mode():
    predictions = model(**inputs).logits.argmax(dim=-1)[0].tolist()

print("\nINPUT SENTENCE:")
print(text)

print("\n1. BERT BIO TAGGING")
print("-" * 35)
print(f"{'TOKEN':<18} {'BIO LABEL'}")
print("-" * 35)

for token, label_id, (start, end) in zip(tokens, predictions, offsets):
    if start == end:
        continue

    label = model.config.id2label[label_id]
    print(f"{text[start:end]:<18} {label}")

# Final entity extraction
print("\n2. EXTRACTED ENTITIES")
print("-" * 35)

for entity in ner(text):
    print(f"{entity['word']:<18} {entity['entity_group']}")
