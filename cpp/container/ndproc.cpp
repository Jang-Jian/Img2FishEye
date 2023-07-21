#ifndef __ndproc_cpp__
#define __ndproc_cpp__

#include <omp.h>
#include <cstdint>

#include "ndproc.hpp"
#include "teproc.hpp"


#define NDARRAY_INDEX(index, now_x, now_y, now_c, now_b, \
                      height, width, channels, batch, type) \
    int index = 0; \
    switch(type) \
    { \
        case HWCN: \
            index = ((now_b + now_c * batch) + (now_y * width + now_x) * channels * batch); break; \
        case CHWN: \
            index = now_b + (now_y * width + now_x) * batch + now_c * batch * width * height; break; \
        case NCHW: \
            index = (now_y * width + now_x) + width * height * now_c + now_b * width * height * channels; break; \
        case NHWC: \
            index = now_c + (now_y * width + now_x) * channels + now_b * width * height * channels; break; \
    };

#define GET_NDARRAY_DEFINE(now_x, now_y, now_c, now_b, \
                           height, width, channels, batch, type) \
    if (now_x < 0 || now_x >= width || now_y < 0 || now_y >= height || \
        now_c < 0 || now_c >= channels) return 0; \
    NDARRAY_INDEX(index, now_x, now_y, now_c, now_b, height, width, channels, batch, type); \
    return src[index];

#define SET_NDARRAY_DEFINE(now_x, now_y, now_c, now_b, \
                           height, width, channels, batch, type) \
    if (now_x >= 0 && now_x < width && now_y >= 0 && now_y < height && \
        now_c >= 0 && now_c < channels) \
    { \
        NDARRAY_INDEX(index, now_x, now_y, now_c, now_b, height, width, channels, batch, type); \
        dst[index] = value; \
    }


template<typename dtype>
inline dtype getNDHost(const dtype *src, const int now_x, const int now_y, const int now_c, const int now_b,
                       const int height, const int width, const int channels, const int batch, 
                       const ndshape type)
{
    GET_NDARRAY_DEFINE(now_x, now_y, now_c, now_b,
                       height, width, channels, batch, type);
}

template<typename dtype>
inline void setNDHost(dtype *dst, const dtype value,
                      const int now_x, const int now_y, const int now_c, const int now_b,
                      const int height, const int width, const int channels, const int batch,
                      const ndshape type)
{
    SET_NDARRAY_DEFINE(now_x, now_y, now_c, now_b,
                       height, width, channels, batch, type);
}




template<typename dtype1, typename dtype2>
void nd2tensor(const dtype1 *nd_ptr, dtype2 *tensor_ptr, 
               const int h, const int w, const int c, const int b,
               const ndshape nd_shape_type)
{
    #pragma omp parallel for
    for (int now_y = 0; now_y < h; ++now_y)
    {
        for (int now_x = 0; now_x < w; ++now_x)
        {
            for (int now_c = 0; now_c < c; ++now_c)
            {
                for (int now_b = 0; now_b < b; ++now_b)
                {
                    dtype1 nd_val = getNDHost(nd_ptr, now_x, now_y, now_c, now_b,
                                              h, w, c, b, nd_shape_type);
                    setTsHost(tensor_ptr, static_cast<dtype2>(nd_val),
                              now_x, now_y, now_c, now_b, h, w, c);
                }
            }
        }
    }
}

template<typename dtype1, typename dtype2>
void tensor2nd(dtype1 *nd_ptr, const dtype2 *tensor_ptr, 
               const int h, const int w, const int c, const int b,
               const ndshape nd_shape_type)
{
    #pragma omp parallel for
    for (int now_y = 0; now_y < h; ++now_y)
    {
        for (int now_x = 0; now_x < w; ++now_x)
        {
            for (int now_c = 0; now_c < c; ++now_c)
            {
                for (int now_b = 0; now_b < b; ++now_b)
                {
                    dtype2 te_val = getTsHost(tensor_ptr, now_x, now_y, now_c, now_b, h, w, c);

                    setNDHost(nd_ptr, static_cast<dtype1>(te_val),
                              now_x, now_y, now_c, now_b, h, w, c, b, nd_shape_type);
                }
            }
        }
    }
}
    



template void nd2tensor<uint8_t, uint8_t>(const uint8_t *nd_ptr, uint8_t *tensor_ptr, 
                                          const int h, const int w, const int c, const int b,
                                          const ndshape nd_shape_type);
template void nd2tensor<float, float>(const float *nd_ptr, float *tensor_ptr, 
                                      const int h, const int w, const int c, const int b,
                                      const ndshape nd_shape_type);

template void tensor2nd<uint8_t, uint8_t>(uint8_t *nd_ptr, const uint8_t *tensor_ptr, 
                                          const int h, const int w, const int c, const int b,
                                          const ndshape nd_shape_type);
template void tensor2nd<float, float>(float *nd_ptr, const float *tensor_ptr, 
                                      const int h, const int w, const int c, const int b,
                                      const ndshape nd_shape_type);

#endif