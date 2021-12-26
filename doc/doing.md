**2021.12.21**

- 尝试对原label提取contour，然后提取内部每个voxel到轮廓的最近距离，除以max（取连通域之内的最大值）距离（归一化），得到距离比例，根据该比例和某一个阈值得到不确定的voxel位置，在这些位置使用soft的label，soft的label由SVLS得到