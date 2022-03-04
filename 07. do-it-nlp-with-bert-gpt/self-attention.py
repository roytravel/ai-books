import torch
import numpy as np
from torch.nn.functional import softmax

# 1. define input vector sequence: 3 x 4 = (number of words what entered) * (word embedding dimension)
x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 2.0, 0.0, 2.0],
    [1.0, 1.0, 1.0, 1.0],
])

# 2. define weighted query, weighted key, weigted value
w_query = torch.tensor([
    [1.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0]
])

w_key = torch.tensor([
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0]
])

v_value = torch.tensor([
    [0.0, 2.0, 0.0],
    [0.0, 3.0, 0.0],
    [1.0, 0.0, 3.0],
    [1.0, 1.0, 0.0]
])

# 3. create query, key, value
queries = torch.matmul(x, w_query)
keys = torch.matmul(x, w_key)
values = torch.matmul(x, v_value)

# 4. create attension score
attn_scores = torch.matmul(queries, keys.T)

# 5. apply softmax function
key_dim_sqrt = np.sqrt(keys.shape[-1])
attn_probs = softmax(attn_scores / key_dim_sqrt)

# 6. weighted sum with values
weighted_values = torch.matmul(attn_probs, values)
print (weighted_values)