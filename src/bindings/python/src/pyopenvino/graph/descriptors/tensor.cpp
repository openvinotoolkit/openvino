// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/descriptors/tensor.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "openvino/core/descriptor/tensor.hpp"
#include "pyopenvino/core/common.hpp"

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

                :return: The shape description.
                :rtype:  openvino.runtime.Shape
             )");

    tensor.def("get_rt_info",
               (PyRTMap & (ov::descriptor::Tensor::*)()) & ov::descriptor::Tensor::get_rt_info,
               py::return_value_policy::reference_internal,
               R"(
                Returns PyRTMap which is a dictionary of user defined runtime info.

                :return: A dictionary of user defined data.
                :rtype: openvino.runtime.RTMap
             )");

    tensor.def("size",
               &ov::descriptor::Tensor::size,
               R"(
                Returns the size description.

                :return: The size description.
                :rtype: size_t
             )");

    tensor.def("get_partial_shape",
               &ov::descriptor::Tensor::get_partial_shape,
               R"(
                Returns the partial shape description.

                :return: PartialShape description.
                :rtype: openvino.runtime.PartialShape
             )");

    tensor.def("get_element_type",
               &ov::descriptor::Tensor::get_element_type,
               R"(
                Returns the element type description.

                :return: Type description.
                :rtype: openvino.runtime.Type
             )");

    tensor.def("get_names",
               &ov::descriptor::Tensor::get_names,
               R"(
                Returns names.

                :return: Get names.
                :rtype: set
             )");

    tensor.def("set_names",
               &ov::descriptor::Tensor::set_names,
               py::arg("names"),
               R"(
                Set names for tensor.

                :param names: Set of names.
                :type names: set
             )");

    tensor.def("add_names",
               &ov::descriptor::Tensor::add_names,
               py::arg("names"),
               R"(
                Adds names for tensor.

                :param names: Add names.
                :type names: set
             )");

    tensor.def("get_any_name",
               &ov::descriptor::Tensor::get_any_name,
               R"(
                Returns any of set name.

                :return: Any name.
                :rtype: string
             )");

    tensor.def("get_lower_value",
               &ov::descriptor::Tensor::get_lower_value,
               R"(
                Returns the lower bound of the tensor.

                :return: Lower bound.
                :rtype: openvino.runtime.Tensor
             )");

    tensor.def("set_lower_value",
               &ov::descriptor::Tensor::set_lower_value,
               py::arg("lower_bound"),
               R"(
                Sets the lower bound of the tensor.

                :param lower_bound: The lower bound value.
                :type lower_bound: openvino.runtime.Tensor
             )");

    tensor.def("get_upper_value",
               &ov::descriptor::Tensor::get_upper_value,
               R"(
                Returns the upper bound of the tensor.

                :return: Upper bound.
                :rtype: openvino.runtime.Tensor
             )");

    tensor.def("set_upper_value",
               &ov::descriptor::Tensor::set_upper_value,
               py::arg("upper_bound"),
               R"(
                Sets the upper bound of the tensor.

                :param upper_bound: Sets the upper bound value.
                :type upper_bound: openvino.runtime.Tensor
             )");

    tensor.def("get_value_symbol",
               &ov::descriptor::Tensor::get_value_symbol,
               R"(
                Returns the List of symbols.

                :return: List of Symbols.
                :rtype: List[openvino.runtime.Symbol]
             )");

    tensor.def("set_value_symbol",
               &ov::descriptor::Tensor::set_value_symbol,
               py::arg("value_symbol"),
               R"(
                Sets the value symbol of the tensor.

                :param value_symbol: List of Symbols
                :type value_symbol: List[openvino.runtime.Symbol]
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

    tensor.def("__repr__", [](const ov::descriptor::Tensor& self) {
        std::stringstream shape_type_ss;
        shape_type_ss << " shape" << self.get_partial_shape() << " type: " << self.get_element_type();
        auto names_str = Common::docs::container_to_string(self.get_names(), ", ");
        return "<" + Common::get_class_name(self) + ": names[" + names_str + "]" + shape_type_ss.str() + ">";
    });
}
