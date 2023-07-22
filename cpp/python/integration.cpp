#ifndef __cudaPipe4py_cpp__
#define __cudaPipe4py_cpp__

#include <boost/python.hpp>

#include "../container/numpy.hpp"
#include "../container/tensor.hpp"
#include "../imgproc/lenproc.hpp"


BOOST_PYTHON_MODULE(integration)
{
    using namespace boost::python;


    def("nd2tensorImple", nd2tensorImple);
    def("tensor2ndImple", tensor2ndImple);
    def("regular2fisheye", regular2fisheye);
    def("cvtptrl2fe", cvtptrl2fe);

    

    enum_<ndshape>("ndshape")
        .value("HWCN", HWCN)
        .value("CHWN", CHWN)
        .value("NCHW", NCHW)
        .value("NHWC", NHWC);

    enum_<dtype>("dtype")
        .value("UINT8", UINT8)
        .value("FLOAT32", FLOAT32);


    class_<Point>("Point", init<>())
        .def_readwrite("x", &Point::x)
        .def_readwrite("y", &Point::y);


    class_<Tensor>("Tensor", init<>())
        .def("create", &Tensor::create)
        .def("getDtype", &Tensor::getDtype)
        .def("getHeight", &Tensor::getHeight)
        .def("getWidth", &Tensor::getWidth)
        .def("getChannel", &Tensor::getChannel)
        .def("getBatch", &Tensor::getBatch);
}

#endif