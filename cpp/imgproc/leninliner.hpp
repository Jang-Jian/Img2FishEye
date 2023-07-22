#ifndef __leninliner_hpp__
#define __leninliner_hpp__

#include <cmath>

using namespace std;


inline void pt4rl2fe(const int src_x, const int src_y, float &dst_x, float &dst_y,
                     const int center_x, const int center_y, const float radian,
                     const float k1, const float k2, const float k3)
{
    float theta = atan2f(src_y - center_y, src_x - center_x);
    float radius = sqrtf(powf(src_x - center_x, 2) + powf(src_y - center_y, 2));
    float radius_d = (k1 * powf(radius, 3) + k2 * powf(radius, 2) + k3 * radius);

    dst_x = center_x + radius_d * cosf(theta - radian);
    dst_y = center_y + radius_d * sinf(theta - radian);
}

#endif