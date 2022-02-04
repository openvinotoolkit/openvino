// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/tensor.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "openvino/runtime/tensor.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_Tensor(py::module m) {
    py::class_<ov::Tensor, std::shared_ptr<ov::Tensor>> cls(m, "Tensor");
    cls.doc() = "openvino.runtime.Tensor holding either copy of memory or shared host memory.";

    cls.def(py::init([](py::array& array, bool shared_memory) {
                return Common::tensor_from_numpy(array, shared_memory);
            }),
            py::arg("array"),
            py::arg("shared_memory") = false,
            R"(
                Tensor's special constructor.

                Parameters
                ----------
                array : numpy.array

                shared_memory : bool
                    If `True` this Tensor memory is being shared with a host,
                    that means the responsibility of keeping host memory is
                    on the side of a user. Any action performed on the host
                    memory will be reflected on this Tensor's memory!
                    If `False`, data is being copied to this Tensor.

                    Requires data to be C_CONTIGUOUS if `True`.

                    Default: `False`

                Returns
                ----------
                __init__ : openvino.runtime.Tensor
            )");

    cls.def(py::init([](py::array& array, const ov::Shape& shape) {
                return Common::tensor_from_pointer(array, shape);
            }),
            py::arg("array"),
            py::arg("shape"),
            R"(
                Another Tensor's special constructor.

                It take an array or slice of it and shape that will be
                selected starting from the first element of given array/slice. 
                Please use it only in advanced cases if necessary!

                Parameters
                ----------
                array : numpy.array
                    Underlaying methods will retrieve pointer on first element
                    from it, which is simulating `host_ptr` from C++ API.
                    Tensor memory is being shared with a host,
                    that means the responsibility of keeping host memory is
                    on the side of a user. Any action performed on the host
                    memory will be reflected on this Tensor's memory!

                    Data is required to be C_CONTIGUOUS.

                shape : openvino.runtime.Shape

                Returns
                ----------
                __init__ : openvino.runtime.Tensor

                Examples
                ----------
                import openvino.runtime as ov
                import numpy as np

                arr = np.array([[1, 2, 3], [4, 5, 6]])

                t = ov.Tensor(arr[1][0:1], ov.Shape([3]))

                t.data[0] = 9

                print(arr)
                >>> [[1 2 3]
                >>>  [9 5 6]]
            )");

    cls.def(py::init<const ov::element::Type, const ov::Shape>(), py::arg("type"), py::arg("shape"));

    cls.def(py::init<const ov::element::Type, const std::vector<size_t>>(), py::arg("type"), py::arg("shape"));

    cls.def(py::init([](py::dtype& np_dtype, std::vector<size_t>& shape) {
                return ov::Tensor(Common::dtype_to_ov_type().at(py::str(np_dtype)), shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init([](py::object& np_literal, std::vector<size_t>& shape) {
                return ov::Tensor(Common::dtype_to_ov_type().at(py::str(py::dtype::from_args(np_literal))), shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init([](py::dtype& np_dtype, const ov::Shape& shape) {
                return ov::Tensor(Common::dtype_to_ov_type().at(py::str(np_dtype)), shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init([](py::object& np_literal, const ov::Shape& shape) {
                return ov::Tensor(Common::dtype_to_ov_type().at(py::str(py::dtype::from_args(np_literal))), shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init<ov::Tensor, ov::Coordinate, ov::Coordinate>(), py::arg("other"), py::arg("begin"), py::arg("end"));

    cls.def(py::init<ov::Tensor, std::vector<size_t>, std::vector<size_t>>(),
            py::arg("other"),
            py::arg("begin"),
            py::arg("end"));

    cls.def("get_element_type",
            &ov::Tensor::get_element_type,
            R"(
            Gets Tensor's element type.

            Returns
            ----------
            get_element_type : openvino.runtime.Type
            )");

    cls.def_property_readonly("element_type",
                              &ov::Tensor::get_element_type,
                              R"(
                                Tensor's element type.

                                Returns
                                ----------
                                element_type : openvino.runtime.Type
                              )");

    cls.def("get_size",
            &ov::Tensor::get_size,
            R"(
            Gets Tensor's size as total number of elements.

            Returns
            ----------
            get_size : int
                Total number of elements in this Tensor.
            )");

    cls.def_property_readonly("size",
                              &ov::Tensor::get_size,
                              R"(
                                Tensor's size as total number of elements.

                                Returns
                                ----------
                                size : int
                                    Total number of elements in this Tensor.
                              )");

    cls.def("get_byte_size",
            &ov::Tensor::get_byte_size,
            R"(
            Gets Tensor's size in bytes.

            Returns
            ----------
            get_byte_size : int
                Size in bytes for this Tensor.
            )");

    cls.def_property_readonly("byte_size",
                              &ov::Tensor::get_byte_size,
                              R"(
                                Tensor's size in bytes.

                                Returns
                                ----------
                                byte_size : int
                                    Size in bytes for this Tensor.
                              )");

    cls.def("get_strides",
            &ov::Tensor::get_strides,
            R"(
            Gets Tensor's strides in bytes.

            Returns
            ----------
            get_strides : openvino.runtime.Strides
                Sizes in bytes for this Tensor's strides.
            )");

    cls.def_property_readonly("strides",
                              &ov::Tensor::get_strides,
                              R"(
                                Tensor's strides in bytes.

                                Returns
                                ----------
                                strides : openvino.runtime.Strides
                                    Sizes in bytes for this Tensor's strides.
                              )");

    cls.def_property_readonly(
        "data",
        [](ov::Tensor& self) {
            return py::array(Common::ov_type_to_dtype().at(self.get_element_type()),
                             self.get_shape(),
                             self.get_strides(),
                             self.data(),
                             py::cast(self));
        },
        R"(
            Access to Tensor's data.

            Returns
            ----------
            data : numpy.array
        )");

    cls.def("get_shape",
            &ov::Tensor::get_shape,
            R"(
            Gets Tensor's shape.

            Returns
            ----------
            get_shape : openvino.runtime.Shape
            )");

    cls.def("set_shape",
            &ov::Tensor::set_shape,
            R"(
            Sets Tensor's shape.

            Parameters
            ----------
            shape : list[int]

            Returns
            ----------
            set_shape : None
            )");

    cls.def(
        "set_shape",
        [](ov::Tensor& self, std::vector<size_t>& shape) {
            self.set_shape(shape);
        },
        R"(
            Sets Tensor's shape.

            Parameters
            ----------
            shape : openvino.runtime.Shape

            Returns
            ----------
            set_shape : None
            )");

    cls.def_property("shape",
                     &ov::Tensor::get_shape,
                     &ov::Tensor::set_shape,
                     R"(
                        Tensor's shape.

                        Parameters
                        ----------
                        shape : openvino.runtime.Shape

                        Returns
                        ----------
                        shape : openvino.runtime.Shape
                     )");

    cls.def_property(
        "shape",
        &ov::Tensor::get_shape,
        [](ov::Tensor& self, std::vector<size_t>& shape) {
            self.set_shape(shape);
        },
        R"(
            Tensor's shape.

            Parameters
            ----------
            shape : list[int]

            Returns
            ----------
            shape : openvino.runtime.Shape
        )");
}
