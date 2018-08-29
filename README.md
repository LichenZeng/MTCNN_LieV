# MTCNN_LieV
Implement MTCNN base on Celeba dataset at LieV

Note:
======

Author: Lycan
Date: 20180829
Subject: The note of MTCNN bug debug

1, LongTensor 使用主要事项
start_index = start_index.data.numpy()  # Important: LongTensor to numpy
LongTensor类型的张量 除以 浮点数，会自动去掉 浮点数小数点后面的数。所以需要小心其出现除零错误，如：

LongTensor(3)/1.5 = 3
LongTensor(3)/0.5 会报错“interrupted by signal 8: SIGFPE”


2, 在使用DataLoader时，如果样本集少于batch_size指定的数量，在后面使用时可能会出现 实际测试数据 不足batch_size指定的数量 的错误。
dataloader = DataLoader(faceDataset, batch_size=10, shuffle=True, num_workers=2)
offset_index = torch.nonzero(offset_mask)[:, 0]  # 选出非负样本的索引


