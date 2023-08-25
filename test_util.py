import numpy as np
import torch

images_list = []
temp = np.zeros((3, 112, 112), dtype=torch.float32)

images_list.extend([np.zeros((3, 112, 112), dtype=torch.float32)] * (32 - len(images_list)))

print('success')