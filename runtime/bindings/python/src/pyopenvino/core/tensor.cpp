// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/tensor.hpp"
#include "pyopenvino/core/common.hpp"
#include "openvino/runtime/tensor.hpp"
#include <pybind11/numpy.h>


namespace py = pybind11;
using namespace ov::runtime;


std::map<ov::element::Type, pybind11::dtype> dtype_map = {{ov::element::f16,     py::dtype("float16")},
                                                          {ov::element::bf16,    py::dtype("float16")},
                                                          {ov::element::f32,     py::dtype("float32")},
                                                          {ov::element::f64,     py::dtype("float64")},
                                                          {ov::element::i8,      py::dtype("int8")},
                                                          {ov::element::i16,     py::dtype("int16")},
                                                          {ov::element::i32,     py::dtype("int32")},
                                                          {ov::element::i64,     py::dtype("int64")},
                                                          {ov::element::u8,      py::dtype("uint8")},
                                                          {ov::element::u16,     py::dtype("uint16")},
                                                          {ov::element::u32,     py::dtype("uint32")},
                                                          {ov::element::u64,     py::dtype("uint64")},
                                                          {ov::element::boolean, py::dtype("bool")},
                                                          {ov::element::u1,      py::dtype("uint8")},
                                                          };

template <typename T>
ov::runtime::Tensor create_Tensor(const ov::element::Type type, py::array_t<T>& array){
    std::vector<size_t> shape(array.shape(), array.shape() + array.ndim());
    return Tensor(type, ov::Shape(shape), (void *)array.data());
}

void regclass_Tensor(py::module m){
    py::class_<Tensor, std::shared_ptr<Tensor>> cls(m, "Tensor");

    cls.def(py::init());
    cls.def(py::init<const ov::element::Type, const ov::Shape>());
    cls.def(py::init([](py::array_t<float>& array){
        return create_Tensor(ov::element::f32, array);
    }));
    cls.def(py::init([](py::array_t<double>& array){
        return create_Tensor(ov::element::f64, array);
    }));
    cls.def(py::init([](py::array_t<int64_t>& array){
        return create_Tensor(ov::element::i64, array);
    }));
    cls.def(py::init([](py::array_t<uint64_t>& array){
        return create_Tensor(ov::element::u64, array);
    }));
    cls.def(py::init([](py::array_t<int32_t>& array){
        return create_Tensor(ov::element::i32, array);
    }));
    cls.def(py::init([](py::array_t<uint32_t>& array){
        return create_Tensor(ov::element::u32, array);
    }));
    cls.def(py::init([](py::array_t<int16_t>& array){
        return create_Tensor(ov::element::i16, array);
    }));
    cls.def(py::init([](py::array_t<uint16_t>& array){
        return create_Tensor(ov::element::u16, array);
    }));
    cls.def(py::init([](py::array_t<int8_t>& array){
        return create_Tensor(ov::element::i8, array);
    }));
    cls.def(py::init([](py::array_t<uint8_t>& array){
        return create_Tensor(ov::element::u8, array);
    }));
    cls.def(py::init([](py::array_t<bool>& array){
        return create_Tensor(ov::element::boolean, array);
    }));
    cls.def(py::init([](py::array& f16_array){
        std::vector<size_t> shape(f16_array.shape(), f16_array.shape() + f16_array.ndim());
        return Tensor(ov::element::f16, ov::Shape(shape), (void *)f16_array.data());
    }));

    cls.def_property_readonly("element_type", &Tensor::get_element_type);
    cls.def_property("shape", &Tensor::get_shape, &Tensor::set_shape);

    cls.def_property_readonly("data", [](Tensor& self) {
        ov::element::Type dtype = self.get_element_type();
        return py::array(dtype_map[dtype], self.get_shape(), self.data(), py::cast(self));
    });
}
