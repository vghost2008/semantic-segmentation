##预测图像

```
predict_on_imgs.py
```

##输入图像格式
使用RGB顺序
图像需要经过以下处理

- 除以255
- 减均值[0.485, 0.456, 0.406]
- 除以方差[0.229, 0.224, 0.225]
- 将HxWxC顺序转换为CxHxW
- 扩展为4维(增加batch_size维度)

可以参考 predict_on_imgs.py

##输出mask格式

[B,H,W] 其中的值代表当前像素点的类别(B表示batch size)

##类别表

```
0:"construction--flat--road",
1:"construction--flat--sidewalk",
2:"object--street-light",
3:"construction--structure--bridge",
4:"construction--structure--building",
5:"human",
6:"object--support--pole",
7:"marking--continuous--dashed",
8:"marking--continuous--solid",
9:"marking--discrete--crosswalk-zebra",
10:"nature--sand",
11:"nature--sky",
12:"nature--snow",
13:"nature--terrain",
14:"nature--vegetation",
15:"nature--water",
16:"object--vehicle--bicycle",
17:"object--vehicle--boat",
18:"object--vehicle--bus",
19:"object--vehicle--car",
20:"object--vehicle--caravan",
21:"object--vehicle--motorcycle",
22:"object--vehicle--on-rails",
23:"object--vehicle--truck",
24:"construction--flat--pedestrian-area",
25:"construction--structure--tunnel",
26:"nature--wasteland",
```

