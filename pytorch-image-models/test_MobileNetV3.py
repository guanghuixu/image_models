import timm
from thop import profile
import torch
def MADDs_Params(model):
    # pip install thop
    input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input, ))
    return flops/1e6, params/1e6

model_dict = {}
for ratio in range(100, 151, 10):
    model_name = 'mobilenetv3_large_' + str(ratio)
    model = timm.create_model(model_name)
    flops, params = MADDs_Params(model)
    model_dict[model_name] = [flops, params]

for key, value in model_dict.items():
    print('{}, Madds: {}, Params: {}'.format(key, value[0], value[1]))

