#ifndef __lenproc_hpp__
#define __lenproc_hpp__

#include "../container/tensor.hpp"


void regular2fisheye(const Tensor &src, Tensor &dst, 
                     const float angle = 0.0f, const float k1 = 0.0000007,
                     const float k2 = 0.00000000005, const float k3 = 0.7);

#endif