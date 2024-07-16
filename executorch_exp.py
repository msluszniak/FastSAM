import torch
from torch.export import export
from executorch.exir import to_edge
import cv2
from ultralytics.nn.tasks import attempt_load_one_weight
import numpy as np

model, _ = attempt_load_one_weight('./weights/FastSAM-s.pt')

IMAGE_PATH = './images/parrot.jpg'
im = cv2.imread(IMAGE_PATH)
img0 = im.copy()
im = cv2.resize(im, (1024, 1024), interpolation = cv2.INTER_AREA)
im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
im = np.ascontiguousarray(im)
# Convert into torch
im = torch.from_numpy(im)
im = im.float()  # uint8 to fp16/32
im /= 255  # 0 - 255 to 0.0 - 1.0
if len(im.shape) == 3:
    im = im[None]  # expand for batch dim

model(im)

#aten_dialect = export(model, (im,))
pre_autograd_aten_dialect = capture_pre_autograd_graph(model, (im,))
aten_dialect = export(pre_autograd_aten_dialect, (im,))
#edge_program = to_edge(aten_dialect)
#executorch_program = edge_program.to_executorch()
#with open(fast_sam.pte, 'wb') as file:
#    file.write(executorch_program.buffer)

