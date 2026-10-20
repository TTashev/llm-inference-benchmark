
from transformers import pipeline

print("Loading model...")

generator = pipeline(
    "text-generation",
    model="sshleifer/tiny-gpt2",
    device="cpu"
)

print("Model loaded!")

result = generator(
    "Once upon a time",
    max_new_tokens=30,
    do_sample=False,
    pad_token_id=generator.tokenizer.eos_token_id
)

print(result[0]["generated_text"])

