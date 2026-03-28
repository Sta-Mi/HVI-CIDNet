import torch
import gc

# 清空缓存
gc.collect()
torch.cuda.empty_cache()

# 检查是否释放了连续空间
print(torch.cuda.memory_summary())