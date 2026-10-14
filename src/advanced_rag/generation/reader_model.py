from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

READER_MODEL_NAME = "mlx-community/Qwen2.5-7B-Instruct-4bit"
model, tokenizer = load(READER_MODEL_NAME)

_sampler = make_sampler(temp=0.2, top_p=0.95)
_logits_processors = make_logits_processors(repetition_penalty=1.1)

def READER_LLM(prompt, max_tokens=500):
    text = generate(
        model, tokenizer, prompt=prompt,
        max_tokens=max_tokens,
        sampler=_sampler,
        logits_processors=_logits_processors,
        verbose=False,
    )
    return [{"generated_text": text}]


# READER_LLM = generate(
#     model, tokenizer, prompt=prompt,
#     max_tokens=500,
#     sampler=_sampler,
#     logits_processors=_logits_processors,
#     verbose=False,
# )