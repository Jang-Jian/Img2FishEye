#ifndef __teproc_hpp__
#define __teproc_hpp__

#define TENSOR_INDEX \
    (now_c * (height * width) + now_y * width + now_x) + now_b * width * height * channels


#define GET_TENSOR_DEFINE \
    if (now_x < 0 || now_x >= width || now_y < 0 || now_y >= height || \
        now_c < 0 || now_c >= channels) return 0; \
    return src[TENSOR_INDEX];

#define SET_TENSOR_DEFINE \
    if (now_x >= 0 && now_x < width && now_y >= 0 && now_y < height && \
        now_c >= 0 && now_c < channels) \
    { \
        dst[TENSOR_INDEX] = value; \
    }


template<typename dtype>
inline dtype getTsHost(const dtype *src, 
                       const int now_x, const int now_y, const int now_c, const int now_b,
                       const int height, const int width, const int channels)
{
    // check_boundary.
    GET_TENSOR_DEFINE;
}

template<typename dtype>
inline void setTsHost(dtype *dst, const dtype value,
                      const int now_x, const int now_y, const int now_c, const int now_b,
                      const int height, const int width, const int channels)
{
    SET_TENSOR_DEFINE;
}

#endif