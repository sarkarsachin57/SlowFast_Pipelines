# import time

# while True:
#     time.sleep(1)
#     print("running!")

import torch, time
for i in range(100000):
    x = torch.randn(1000, 1000).cuda()
    x = x @ x
    print(x)
    time.sleep(0.05)
print("completed")