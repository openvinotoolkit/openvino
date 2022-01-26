// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/descriptors/tensor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "openvino/core/descriptor/tensor.hpp"

namespace py = pybind11;

using PyRTMap = ov::RTMap;

PYBIND11_MAKE_OPAQUE(PyRTMap);

void regclass_graph_descriptor_Tensor(py::module m) {
    py::class_<ov::descriptor::Tensor, std::shared_ptr<ov::descriptor::Tensor>> tensor(m, "DescriptorTensor");

    tensor.doc() = "openvino.descriptor.Tensor wraps ov::descriptor::Tensor";

    tensor.def("get_shape",
               &ov::descriptor::Tensor::get_shape,
               R"(
                Returns the shape description.

                Returns
                ----------
                get_shape : Shape
                   The shape description.
             )");

    tensor.def("get_rt_info",
               (PyRTMap & (ov::descriptor::Tensor::*)()) & ov::descriptor::Tensor::get_rt_info,
               py::return_value_policy::reference_internal,
               R"(
                Returns PyRTMap which is a dictionary of user defined runtime info.

                Returns
                ----------
                get_rt_info : PyRTMap
                    A dictionary of user defined data.
             )");

    tensor.def("size",
               &ov::descriptor::Tensor::size,
               R"(
                Returns the size description

                Returns
                ----------
                size : size_t
                    The size description.
             )");

    tensor.def("get_partial_shape",
               &ov::descriptor::Tensor::get_partial_shape,
               R"(
                Returns the partial shape description

                Returns
                ----------
                get_partial_shape : PartialShape
                    PartialShape description.
             )");

    tensor.def("get_element_type",
               &ov::descriptor::Tensor::get_element_type,
               R"(
                Returns the element type description

                Returns
                ----------
                get_element_type : Type
                    Type description
             )");

    tensor.def("get_names",
               &ov::descriptor::Tensor::get_names,
               R"(
                Returns names

                Returns
                ----------
                get_names : set
                    Set of names
             )");

    tensor.def("set_names",
               &ov::descriptor::Tensor::set_names,
               py::arg("names"),
               R"(
                Set names for tensor

                Parameters
                ----------
                names : set
                    Set of names
             )");

    tensor.def("get_any_name",
               &ov::descriptor::Tensor::get_any_name,
               R"(
                Returns any of set name

                Returns
                ----------
                get_any_name : string
                    Any name
             )");

    tensor.def_property_readonly("shape", &ov::descriptor::Tensor::get_shape);

    tensor.def_property_readonly("rt_info",
                                 (PyRTMap & (ov::descriptor::Tensor::*)()) & ov::descriptor::Tensor::get_rt_info,
                                 py::return_value_policy::reference_internal);

    tensor.def_property_readonly("size", &ov::descriptor::Tensor::size);

    tensor.def_property_readonly("partial_shape", &ov::descriptor::Tensor::get_partial_shape);

    tensor.def_property_readonly("element_type", &ov::descriptor::Tensor::get_element_type);

    tensor.def_property_readonly("any_name", &ov::descriptor::Tensor::get_any_name);

    tensor.def_property("names", &ov::descriptor::Tensor::get_names, &ov::descriptor::Tensor::set_names);
}
