#ifndef __lenproc_hpp__
#define __lenproc_hpp__

#include "../container/tensor.hpp"

typedef struct
{
    float x;
    float y;
}Point;


Point cvtptrl2fe(const int src_x, const int src_y,
                 const int map_width, const int map_height, const float radian,
                 const float k1, const float k2, const float k3);


void regular2fisheye(const Tensor &src, Tensor &dst, 
                     const float angle = 0.0f, const float k1 = 0.0000007,
                     const float k2 = 0.00000000005, const float k3 = 0.7);

#endif