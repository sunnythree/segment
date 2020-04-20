import torch
import numpy as np

INDEX_GET_COLOR = \
    ((0, 0, 0),
     (128, 0, 0),
     (0, 128, 0),
     (128, 128, 0),
     (0, 0, 128),
     (128, 0, 128),
     (0, 128, 128),
     (128, 128, 128),
     (64, 0, 0),
     (192, 0, 0),
     (64, 128, 0),
     (192, 128, 0),
     (64, 0, 128),
     (192, 0, 128),
     (64, 128, 128),
     (192, 128, 128),
     (0, 64, 0),
     (128, 64, 0),
     (0, 192, 0),
     (128, 192, 0),
     (0, 64, 128),
     (224, 224, 192),)

COLOR_GET_INDEX = {
     0: 0,
     128*256*256: 1,
     128*256: 2,
     128*256*256 + 128*256: 3,
     128: 4,
     128*256*256 + 128: 5,
     128*256 + 128: 6,
     128*256*256 + 128*256 + 128: 7,
     64*256*256: 8,
     192*256*256: 9,
     64*256*256 + 128*256: 10,
     192*256*256 + 128*256: 11,
     64*256*256 + 128: 12,
     192*256*256 + 128: 13,
     64*256*256 + 128*256 + 128: 14,
     192*256*256 + 128*256 + 128: 15,
     64*256: 16,
     128*256*256 + 64*256: 17,
     192*256: 18,
     128*256*256 + 192*256: 19,
     64*256 + 128: 20,
     224*256*256 + 224*256 + 192: 0}

def color2class(color_tensor):
    assert color_tensor.dim() == 4 and color_tensor.shape[1] == 3
    color_tensor_shape = color_tensor.shape
    class_tensor_shape = (color_tensor_shape[0], 21, color_tensor_shape[2], color_tensor_shape[3])
    class_tensor = torch.zeros(class_tensor_shape)
    for i in range(color_tensor_shape[0]):
        for j in range(color_tensor_shape[2]):
            for k in range(color_tensor_shape[3]):
                tmp = color_tensor[i, :, j, k]
                tmp_key = (int(tmp[0]) & ~0xf)*256*256 + (int(tmp[1]) & ~0xf)*256+int(tmp[2]) & ~0xf
                #print(tmp_key,tmp)
                index = COLOR_GET_INDEX[tmp_key]
                print(index, end='')
                tmp1 = class_tensor[i, :, j, k]
                tmp1[index] = 1
    return class_tensor

def color2class2(color_tensor):
    assert color_tensor.dim() == 4 and color_tensor.shape[1] == 3
    color_tensor_shape = color_tensor.shape
    class_tensor_shape = (color_tensor_shape[0], color_tensor_shape[2], color_tensor_shape[3])
    class_tensor = torch.zeros(class_tensor_shape, dtype=torch.long)
    for i in range(color_tensor_shape[0]):
        for j in range(color_tensor_shape[2]):
            for k in range(color_tensor_shape[3]):
                tmp = color_tensor[i, :, j, k]
                tmp_key = (int(tmp[0]) & ~0xf)*256*256 + (int(tmp[1]) & ~0xf)*256+int(tmp[2]) & ~0xf
                #print(tmp_key, tmp)
                index = COLOR_GET_INDEX[tmp_key]
                class_tensor[i, j, k] = int(index)
    return class_tensor


def class2color(class_tensor):
    assert class_tensor.dim() == 4 and class_tensor.shape[1] == 21
    class_tensor_shape = class_tensor.shape
    color_tensor_shape = (class_tensor_shape[0], 3, class_tensor_shape[2], class_tensor_shape[3])
    color_tensor = torch.zeros(color_tensor_shape)
    for i in range(class_tensor_shape[0]):
        for j in range(class_tensor_shape[2]):
            for k in range(class_tensor_shape[3]):
                tmp = class_tensor[i, :, j, k]
                index = tmp.argmax()
                tmp1 = color_tensor[i, :, j, k]
                color = INDEX_GET_COLOR[index]
                tmp1[0] = color[0]
                tmp1[1] = color[1]
                tmp1[2] = color[2]
    return color_tensor
def test_color2class():
    print(color2class(torch.zeros((1, 3, 3, 3), dtype=torch.uint8)))

def test_class2color():
    data = [[[[128, 0, 0],
       [0, 0, 0],
       [0, 128, 0]],

      [[0, 0, 128],
       [64, 64, 0],
       [0, 128, 0]],

      [[0, 0, 0],
       [0, 0, 128],
       [0, 0, 0]]]]
    result = class2color(color2class(torch.from_numpy(np.array(data))))
    result = result.numpy()
    print(result)
    assert (data==result).all()
#test_class2color()