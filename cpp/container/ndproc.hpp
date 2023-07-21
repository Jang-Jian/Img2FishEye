#ifndef __ndproc_hpp__
#define __ndproc_hpp__

enum ndshape
{ 
    HWCN = 0, 
    CHWN = 1,
    NCHW = 2,
    NHWC = 3
};


template<typename dtype1, typename dtype2>
void nd2tensor(const dtype1 *nd_ptr, dtype2 *tensor_ptr, 
               const int h, const int w, const int c, const int b,
               const ndshape nd_shape_type);

template<typename dtype1, typename dtype2>
void tensor2nd(dtype1 *nd_ptr, const dtype2 *tensor_ptr, 
               const int h, const int w, const int c, const int b,
               const ndshape nd_shape_type);

#endif