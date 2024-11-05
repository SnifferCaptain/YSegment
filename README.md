# YSegment
A Quick Segment Net Based on Pytorch
not tested yet. lack of dataset.

# ys1

A net based on yolo's design.
yolo yaml may look something like this:

backbone:
  - [-1, 1, Ynet2B1, []] # 0
  - [-1, 1, Ynet2B2, []] # 1
  - [-1, 1, Ynet2B3, []] # 2

head:
  - [[-1,1], 1, Ynet2Hu1, []] # 3
  - [[-1,0], 1, Ynet2Hu2, []] # 4
  - [[-1,3], 1, Ynet2Hd1, []] # 5
  - [[-1,2], 1, Ynet2Hd2, []] # 6

  - [[4, 5, 6], 1, Detect, [nc]] # Detect(P3, P4, P5)



with an modele layer view:            
ynet2 summary: 540 layers, 5667345 parameters, 5667329 gradients, 14.8 GFLOPs
