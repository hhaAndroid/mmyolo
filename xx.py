from mmengine.fileio import FileClient, load
import json
from collections import OrderedDict
import torch

x = torch.load('damoyolo_tinynasL25_S.pth')
x['state_dict'] = x['model']
del x['model']

state_dict = OrderedDict()
for key, weight in x['state_dict'].items():
    if 'head' in key:
        key = key.replace('head', 'bbox_head')
    state_dict[key] = weight

torch.save({'state_dict': state_dict}, 'damoyolo_tinynasL25_S_mm.pth')
