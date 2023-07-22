#ifndef __lenproc_cpp__
#define __lenproc_cpp__

#include <omp.h>
#include <cstdint>
#include <iostream>

#include "lenproc.hpp"
#include "leninliner.hpp"
#include "../container/teproc.hpp"

using namespace std;


inline uint8_t bilinearInterHost(const uint8_t *src, const float x, const float y, const int c, const int b,
                                 const int height, const int width, const int channels)
{
    int ix = static_cast<int>(floorf(x));
    int iy = static_cast<int>(floorf(y));
    float dx = x - ix;
    float dy = y - iy;
    float one_sub_dy = 1.0f - dy;
    float one_sub_dx = 1.0f - dx;

    float val = one_sub_dy * one_sub_dx * getTsHost(src, ix, iy, c, b, height, width, channels) + \
                dy * one_sub_dx * getTsHost(src, ix, iy+1, c, b, height, width, channels) + \
                one_sub_dy * dx * getTsHost(src, ix+1, iy, c, b, height, width, channels) + \
                dy * dx * getTsHost(src, ix+1, iy+1, c, b, height, width, channels);

    //cout << val << endl;
    return static_cast<uint8_t>(val);
}


void remap(const Tensor &src, const Tensor &map_x, const Tensor &map_y, Tensor &dst)
{
    if (src.getDtype() != UINT8) return;

    const int width = src.getWidth();
    const int height = src.getHeight();
    const int channel = src.getChannel();
    const int batch = src.getBatch();

    dst.create(height, width, channel, batch, UINT8);
    const uint8_t *src_ptr = (uint8_t*)src.getPtr();
    uint8_t *dst_ptr = (uint8_t*)dst.getPtr();
    const float *map_x_ptr = (float*)map_x.getPtr();
    const float *map_y_ptr = (float*)map_y.getPtr();

    #pragma omp parallel for
    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            for (int c = 0; c < channel; ++c)
            {
                int map_index = col + row * width;
                const float x = map_x_ptr[map_index];
                const float y = map_y_ptr[map_index];

                uint8_t val = bilinearInterHost(src_ptr, x, y, c, 0,
                                                height, width, channel);

                setTsHost(dst_ptr, val, col, row, c, 0, height, width, channel);
            }
        }
    }
}


Point cvtptrl2fe(const int src_x, const int src_y,
                 const int map_width, const int map_height, const float radian,
                 const float k1, const float k2, const float k3)
{
    Point dst = {};

    const int center_x = static_cast<int>(map_width / 2);
    const int center_y = static_cast<int>(map_height / 2);

    Tensor map_x, map_y;
    map_x.create(map_height, map_width, 1, 1, FLOAT32);
    map_y.create(map_height, map_width, 1, 1, FLOAT32);
    float *map_x_ptr = (float*)map_x.getPtr();
    float *map_y_ptr = (float*)map_y.getPtr();

    #pragma omp parallel for
    for (int y = 0; y < map_height; ++y)
    {
        for (int x = 0; x < map_width; ++x)
        {
            int map_index = x + y * map_width;

            pt4rl2fe(x, y, map_x_ptr[map_index], map_y_ptr[map_index],
                     center_x, center_y, radian, k1, k2, k3);
        }
    }


    float smallest_dist = 9999999999.0f;
    //float x_tmp = 0.0, y_tmp = 0.0;

    for (int y = 0; y < map_height; ++y)
    {
        for (int x = 0; x < map_width; ++x)
        {
            int map_index = x + y * map_width;
            float map_x_val = map_x_ptr[map_index];
            float map_y_val = map_y_ptr[map_index];

            if (map_x_val >= 0 && map_x_val < map_width && 
                map_y_val >= 0 && map_y_val < map_height)
            {
                float dist = sqrtf(powf(map_x_val - src_x, 2) + powf(map_y_val - src_y, 2));
                if (dist < smallest_dist)
                {
                    smallest_dist = dist;
                    dst.x = x;
                    dst.y = y;
                    //x_tmp = map_x_val;
                    //y_tmp = map_y_val;
                }
            }
        }
    }

    //cout << smallest_dist << ", " << x_tmp << ", " << y_tmp << endl;

    return dst;
}


void regular2fisheye(const Tensor &src, Tensor &dst, 
                     const float angle, const float k1, const float k2, const float k3)
{
    const int width = src.getWidth();
    const int height = src.getHeight();

    const int center_x = static_cast<int>(width / 2);
    const int center_y = static_cast<int>(height / 2);

    Tensor map_x, map_y;
    map_x.create(height, width, 1, 1, FLOAT32);
    map_y.create(height, width, 1, 1, FLOAT32);
    float *map_x_ptr = (float*)map_x.getPtr();
    float *map_y_ptr = (float*)map_y.getPtr();

    const float radian = angle * M_PI / 180.0f;

    // get x & y mapping matrix.
    #pragma omp parallel for
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int map_index = x + y * width;

            pt4rl2fe(x, y, map_x_ptr[map_index], map_y_ptr[map_index],
                     center_x, center_y, radian, k1, k2, k3);
        }
    }

    remap(src, map_x, map_y, dst);
}

#endif