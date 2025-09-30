import torch, numpy as np, pandas as pd
print("Torch:", torch.__version__, "CUDA available:", torch.cuda.is_available())
x = torch.randn(2,3)
print("CPU tensor OK:", x.device)
