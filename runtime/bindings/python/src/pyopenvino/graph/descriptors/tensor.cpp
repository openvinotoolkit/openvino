// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/descriptors/tensor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "openvino/core/descriptor/tensor.hpp"

namespace py = pybind11;

using PyRTMap = std::map<std::string, std::shared_ptr<ov::Variant>>;

PYBIND11_MAKE_OPAQUE(PyRTMap);

void regclass_graph_descriptor_Tensor(py::module m) {
    py::class_<ov::descriptor::Tensor, std::shared_ptr<ov::descriptor::Tensor>> tensor(m, "TensorDescriptor");

    tensor.def(py::init<const ov::element::Type, const ov::PartialShape, const std::string>(),
               py::arg("element_type"),
               py::arg("shape"),
               py::arg("name"));

    tensor.def("get_shape", &ov::descriptor::Tensor::get_shape);

    tensor.def("get_rt_info",
               (PyRTMap & (ov::descriptor::Tensor::*)()) & ov::descriptor::Tensor::get_rt_info,
               py::return_value_policy::reference_internal);

    tensor.def("size", &ov::descriptor::Tensor::size);

    tensor.def("get_partial_shape", &ov::descriptor::Tensor::get_partial_shape);

    tensor.def("get_element_type", &ov::descriptor::Tensor::get_element_type);

    tensor.def("get_names", &ov::descriptor::Tensor::get_names);

    tensor.def("set_names", &ov::descriptor::Tensor::set_names);

    tensor.def_property_readonly("shape", &ov::descriptor::Tensor::get_shape);

    tensor.def_property_readonly("rt_info",
                                 (PyRTMap & (ov::descriptor::Tensor::*)()) & ov::descriptor::Tensor::get_rt_info,
                                 py::return_value_policy::reference_internal);

    tensor.def_property_readonly("size", &ov::descriptor::Tensor::size);

    tensor.def_property_readonly("partial_shape", &ov::descriptor::Tensor::get_partial_shape);

    tensor.def_property_readonly("element_type", &ov::descriptor::Tensor::get_element_type);

    tensor.def_property("names", &ov::descriptor::Tensor::get_names, &ov::descriptor::Tensor::set_names);
}
