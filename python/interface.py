import sys
import integration
import numpy as np


UINT8 = integration.dtype.UINT8
FLOAT32 = integration.dtype.FLOAT32
HWCN = integration.ndshape.HWCN
CHWN = integration.ndshape.CHWN
NCHW = integration.ndshape.NCHW
NHWC = integration.ndshape.NHWC
Tensor = integration.Tensor


def __get4DHWCN(org_shape: tuple,  ndshape_type: integration.ndshape) -> tuple:
    src_shape_len = len(org_shape)
    src_shape = None

    if src_shape_len == 4:
        v1, v2, v3, v4 = org_shape
        if ndshape_type == NCHW:
            # (1, 3, 720, 1280)
            src_shape = (v3, v4, v2, v1)
        elif ndshape_type == HWCN:
            # (720, 1280, 3, 1)
            src_shape = (v1, v2, v3, v4)
        elif ndshape_type == NHWC:
            # (1, 720, 1280, 3)
            src_shape = (v2, v3, v4, v1)
        elif ndshape_type == CHWN:
            # (3, 720, 1280, 1)
            src_shape = (v2, v3, v1, v4)

    elif src_shape_len == 3:
        # if the size of shape is 3, it will return (height, width, channel, 1).
        # p.s. batch size is only 1.
        src_shape = (org_shape[0], org_shape[1], org_shape[2], 1)
    elif src_shape_len == 2:
        # if the size of shape is 3, it will return (height, width, 1, 1).
        # p.s. batch size and channel size are only 1.
        src_shape = (org_shape[0], org_shape[1], 1, 1)
    return src_shape


def __get4DShape(height: int, width: int, channel: int, batch: int, 
                 shape_type: integration.ndshape) -> tuple:
    dst_shape = None
    if shape_type == HWCN:
        dst_shape = (height, width, channel, batch)
    elif shape_type == CHWN:
        dst_shape = (channel, height, width, batch)
    elif shape_type == NCHW:
        dst_shape = (batch, channel, height, width)
    elif shape_type == NHWC:
        dst_shape = (batch, height, width, channel)
    return dst_shape


def nd2tensor(src: np.ndarray, 
              ndshape_type: integration.ndshape) -> integration.Tensor:
    dst = integration.Tensor()
    h, w, c, b = __get4DHWCN(src.shape, ndshape_type)
    #print(h, w, c, b)

    if src.dtype == np.uint8:
        integration.nd2tensorImple(src, dst, h, w, c, b, ndshape_type, UINT8)
    elif src.dtype == np.float32:
        integration.nd2tensorImple(src, dst, h, w, c, b, ndshape_type, FLOAT32)
    else:
        print("interface.py: the " + str(np.uint8) + " is not supported in the 'nd2tensor'.")
        sys.exit(0)

    return dst

def tensor2nd(src: integration.Tensor,
              ndshape_type: integration.ndshape) -> np.ndarray:
    """
    
    """
    dst = None
    src_dtype = src.getDtype()
    height = src.getHeight()
    width = src.getWidth()
    channel = src.getChannel()
    batch = src.getBatch()
    
    if src_dtype == UINT8:
        dst = np.zeros(__get4DShape(height, width, channel, batch, ndshape_type), np.uint8)
        integration.tensor2ndImple(src, dst,height, width, channel, batch,
                                   ndshape_type, UINT8)
    elif src_dtype == FLOAT32:
        dst = np.zeros(__get4DShape(height, width, channel, batch, ndshape_type), np.float32)
        integration.tensor2ndImple(src, dst,height, width, channel, batch,
                                   ndshape_type, FLOAT32)

    return dst


def regular2fisheye(src: integration.Tensor, angle: float = 0.0, 
                    k1: float = 0.0000007, k2: float = 0.00000000005, k3: float = 0.7) -> integration.Tensor:
    """
    tranform regular 2d image to fisheye image.
    """
    dst = integration.Tensor()
    integration.regular2fisheye(src, dst, angle, k1, k2, k3)
    return dst