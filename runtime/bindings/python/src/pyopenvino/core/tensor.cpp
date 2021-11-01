// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/tensor.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "openvino/runtime/tensor.hpp"
#include "pyopenvino/core/common.hpp"

#define C_CONTIGUOUS py::detail::npy_api::constants::NPY_ARRAY_C_CONTIGUOUS_

namespace py = pybind11;

void regclass_Tensor(py::module m) {
    py::class_<ov::runtime::Tensor, std::shared_ptr<ov::runtime::Tensor>> cls(m, "Tensor");

    cls.def(py::init([](py::array& array, bool shared_memory) {
                auto type = Common::dtype_to_ov_type().at(py::str(array.dtype()));
                std::vector<size_t> shape(array.shape(), array.shape() + array.ndim());
                if (shared_memory) {
                    if (C_CONTIGUOUS == (array.flags() & C_CONTIGUOUS)) {
                        std::vector<size_t> strides(array.strides(), array.strides() + array.ndim());
                        return ov::runtime::Tensor(type, shape, const_cast<void*>(array.data(0)), strides);
                    } else {
                        IE_THROW() << "Tensor with shared memory must be C contiguous!";
                    }
                }
                array = py::module::import("numpy").attr("ascontiguousarray")(array).cast<py::array>();
                auto tensor = ov::runtime::Tensor(type, shape);
                std::memcpy(tensor.data(), array.data(0), array.nbytes());
                return tensor;
            }),
            py::arg("array"),
            py::arg("shared_memory") = false);

    cls.def(py::init<const ov::element::Type, const ov::Shape>(), py::arg("type"), py::arg("shape"));

    cls.def(py::init<const ov::element::Type, const std::vector<size_t>>(), py::arg("type"), py::arg("shape"));

    cls.def(py::init([](py::dtype& np_dtype, std::vector<size_t>& shape) {
                return ov::runtime::Tensor(Common::dtype_to_ov_type().at(py::str(np_dtype)), shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init([](py::object& np_literal, std::vector<size_t>& shape) {
                return ov::runtime::Tensor(Common::dtype_to_ov_type().at(py::str(py::dtype::from_args(np_literal))),
                                           shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init([](py::dtype& np_dtype, const ov::Shape& shape) {
                return ov::runtime::Tensor(Common::dtype_to_ov_type().at(py::str(np_dtype)), shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init([](py::object& np_literal, const ov::Shape& shape) {
                return ov::runtime::Tensor(Common::dtype_to_ov_type().at(py::str(py::dtype::from_args(np_literal))),
                                           shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init<ov::runtime::Tensor, ov::Coordinate, ov::Coordinate>(),
            py::arg("other"),
            py::arg("begin"),
            py::arg("end"));

    cls.def(py::init<ov::runtime::Tensor, std::vector<size_t>, std::vector<size_t>>(),
            py::arg("other"),
            py::arg("begin"),
            py::arg("end"));

    cls.def_property_readonly("element_type", &ov::runtime::Tensor::get_element_type);

    cls.def_property_readonly("size", &ov::runtime::Tensor::get_size);

    cls.def_property_readonly("byte_size", &ov::runtime::Tensor::get_byte_size);

    cls.def_property_readonly("data", [](ov::runtime::Tensor& self) {
        return py::array(Common::ov_type_to_dtype().at(self.get_element_type()),
                         self.get_shape(),
                         self.get_strides(),
                         self.data(),
                         py::cast(self));
    });

    cls.def_property("shape", &ov::runtime::Tensor::get_shape, &ov::runtime::Tensor::set_shape);

    cls.def_property("shape",
                     &ov::runtime::Tensor::get_shape,
                     [](ov::runtime::Tensor& self, std::vector<size_t>& shape) {
                         self.set_shape(shape);
                     });
}
