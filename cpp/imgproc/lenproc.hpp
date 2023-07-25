#ifndef __lenproc_hpp__
#define __lenproc_hpp__

#include <vector>

#include "../container/tensor.hpp"

using namespace std;


typedef struct
{
    float x;
    float y;
}Point;



class Cvtptrl2fe
{
    public:
        Cvtptrl2fe(const int map_width, const int map_height, const float angle,
                   const float k1, const float k2, const float k3);
        virtual ~Cvtptrl2fe();

        Point cvtCoord(const int src_x, const int src_y);
        void cvtImage(const Tensor &src, Tensor &dst);

    private:
        int map_width, map_height;
        int center_x, center_y;
        vector<float> normal_fe_x;
        vector<float> normal_fe_y;
        vector<int> normal_indexes;
        Tensor map_x, map_y;
};

void remap(const Tensor &src, const Tensor &map_x, const Tensor &map_y, Tensor &dst);

/*Point cvtptrl2fe(const int src_x, const int src_y,
                 const int map_width, const int map_height, const float radian,
                 const float k1, const float k2, const float k3);*/


/*void regular2fisheye(const Tensor &src, Tensor &dst, 
                     const float angle = 0.0f, const float k1 = 0.0000007,
                     const float k2 = 0.00000000005, const float k3 = 0.7);*/

#endif