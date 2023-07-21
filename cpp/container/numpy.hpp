#ifndef __numpy_hpp__
#define __numpy_hpp__

#include <boost/python.hpp>

#include "ndproc.hpp"
#include "tensor.hpp"

using namespace boost::python;



inline Py_buffer numpyBuffer(const boost::python::object &obj)
{
    PyObject* pobj = obj.ptr();
    Py_buffer pybuf;
    int flag = PyObject_GetBuffer(pobj, &pybuf, PyBUF_SIMPLE);
    if (flag == -1)
    {
        fprintf(stderr, "Error(numpyProc.hpp): numpy.ndarray is unable to convert to Py_buffer. Maybe format is wrong or empty.\n");
        exit(EXIT_FAILURE);
    }
    return pybuf;
}

inline void numpyBufferRelease(Py_buffer &pybuf)
{
    PyBuffer_Release(&pybuf);
}


inline void nd2tensorImple(const boost::python::object &src, Tensor &dst,
                           const int h, const int w, const int c, const int b,
                           const ndshape nd_shape_type, const dtype data_type)
{
    Py_buffer pybuf = numpyBuffer(src);
    dst.create(h, w, c, b, data_type);

    switch (data_type)
    {
        case UINT8:
            nd2tensor((uint8_t*)pybuf.buf, (uint8_t*)dst.getPtr(), 
                       h, w, c, b, nd_shape_type);
            break;
        case FLOAT32:
            nd2tensor((float*)pybuf.buf, (float*)dst.getPtr(), 
                       h, w, c, b, nd_shape_type);
            break;
    }

    numpyBufferRelease(pybuf);
}

inline void tensor2ndImple(const Tensor &src, boost::python::object &dst,
                           const int h, const int w, const int c, const int b,
                           const ndshape nd_shape_type, const dtype data_type)
{
    Py_buffer pybuf = numpyBuffer(dst);
    
    switch (data_type)
    {
        case UINT8:
            tensor2nd((uint8_t*)pybuf.buf, (uint8_t*)src.getPtr(), 
                      h, w, c, b, nd_shape_type);
            break;
        case FLOAT32:
            tensor2nd((float*)pybuf.buf, (float*)src.getPtr(), 
                      h, w, c, b, nd_shape_type);
            break;
    }

    numpyBufferRelease(pybuf);
}

#endif