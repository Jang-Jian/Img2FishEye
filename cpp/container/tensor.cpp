#ifndef __tensor_cpp__
#define __tensor_cpp__

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#include "tensor.hpp"


Tensor::Tensor() : allocated(false) {}

Tensor::~Tensor()
{
    if (allocated)
    {
        //printf("this\n");
        free(this->host);
    }
}

void Tensor::create(const int height, const int width, 
                    const int channel, const int batch,
                    const dtype ts_dtype)
{
    this->height = height;
    this->width = width;
    this->channel = channel;
    this->batch = batch;
    this->ts_dtype = ts_dtype;

    if (allocated)
    {
        free(this->host);
    }

    switch (ts_dtype)
    {
        case UINT8:
            this->host = (void*)calloc(this->height * this->width * this->channel * this->batch, 
                                       sizeof(uint8_t));
            break;
        case FLOAT32:
            this->host = (void*)calloc(this->height * this->width * this->channel * this->batch, 
                                       sizeof(float));
            break;
    } 

    allocated = true;
}

dtype Tensor::getDtype() const
{
    return this->ts_dtype;
}

void* Tensor::getPtr() const
{
    return this->host;
}

int Tensor::getHeight() const
{
    return this->height;
}
    
int Tensor::getWidth() const
{
    return this->width;
}

int Tensor::getChannel() const
{
    return this->channel;
}

int Tensor::getBatch() const
{
    return this->batch;
}

#endif