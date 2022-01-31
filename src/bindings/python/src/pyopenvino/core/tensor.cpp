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
                    If true this Tensor memory is being shared with a host,
                    that means the responsibility of keeping host memory is
                    on the side of a user. Any action performed on the host
                    memory will be reflected on this Tensor's memory!
                    If false, data is being copied to this Tensor.
                    Default: false

                Returns
                ----------
                __init__ : openvino.runtime.Tensor
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

    cls.def_property_readonly("element_type",
                              &ov::Tensor::get_element_type,
                              R"(
                                Tensor's element type.

                                Returns
                                ----------
                                element_type : openvino.runtime.Type
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

    cls.def_property_readonly("byte_size",
                              &ov::Tensor::get_byte_size,
                              R"(
                                Tensor's size in bytes.

                                Returns
                                ----------
                                byte_size : int
                                    Size in bytes for this Tensor.
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
