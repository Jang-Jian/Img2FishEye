#ifndef __tensor_hpp__
#define __tensor_hpp__

#include "dtypecfg.hpp"


struct Tensor
{
    Tensor();
    ~Tensor();

    // p.s. it will free original space when called.
    void create(const int height, const int width, 
                const int channel, const int batch,
                const dtype ts_dtype);

    dtype getDtype() const;
    void* getPtr() const;
    int getHeight() const;
    int getWidth() const;
    int getChannel() const;
    int getBatch() const;
        
private:
    bool allocated;
    void *host;
    int height, width, channel, batch;
    dtype ts_dtype;
};

#endif