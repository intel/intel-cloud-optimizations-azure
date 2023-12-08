import torch
import argparse
import tiktoken
from model import GPT
from contextlib import nullcontext

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str, default="ckpt.pt")
parser.add_argument("--prompt", type=str, default="It is interesting ")
args = parser.parse_args()
print(args)

# -----------------------------------------------------------------------------
ckpt_path = args.ckpt_path  # ignored if init_from is not 'resume'
prompt = args.prompt

init_from = "resume"  # either 'resume' or a gpt2 variant (e.g. 'gpt2-xl')
num_samples = 3  # number of samples to draw
max_new_tokens = 100  # number of tokens generated in each sample
temperature = (
    0.6  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
)
top_k = (
    200  # retain only the top_k most likely tokens, clamp others to have 0 probability
)
seed = 1337
device = "cpu"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = "float16"
compile = False  # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
ctx = nullcontext()

# model
if init_from == "resume":
    # init from a model saved in a specific directory
    checkpoint = torch.load(ckpt_path, map_location=device)
    override_args = dict(dropout=0.0)
    model = GPT.from_pretrained("gpt2", override_args)
    model.load_state_dict(checkpoint)
elif init_from.startswith("gpt2"):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

enc = tiktoken.get_encoding("gpt2")
encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
start_ids = encode(prompt)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# run generation
print(f"Input Prompt: {prompt}")
with torch.no_grad():
    with ctx:
        print(f"--------------- Generated Text ---------------")
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        print(decode(y[0].tolist()))
        print("-------------------------------------------")