// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/parameter.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"  // ov::PartialShape
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/ops/parameter.hpp"

namespace py = pybind11;

void regclass_graph_op_Parameter(py::module m) {
    py::class_<ov::op::v0::Parameter, std::shared_ptr<ov::op::v0::Parameter>, ov::Node> parameter(m, "Parameter");
    parameter.doc() = "openvino.op.Parameter wraps ov::op::v0::Parameter";
    parameter.def("__repr__", [](const ov::Node& self) {
        std::string class_name = py::cast(self).get_type().attr("__name__").cast<std::string>();
        std::string shape = py::cast(self.get_output_partial_shape(0)).attr("__str__")().cast<std::string>();
        std::string type = self.get_element_type().c_type_string();
        return "<" + class_name + ": '" + self.get_friendly_name() + "' (" + shape + ", " + type + ")>";
    });

    parameter.def(py::init<const ov::element::Type&, const ov::Shape&>());
    parameter.def(py::init<const ov::element::Type&, const ov::PartialShape&>());
    //    parameter.def_property_readonly("description", &ov::op::v0::Parameter::description);

    parameter.def(
        "get_partial_shape",
        (const ov::PartialShape& (ov::op::v0::Parameter::*)() const) & ov::op::v0::Parameter::get_partial_shape);
    parameter.def("get_partial_shape",
                  (ov::PartialShape & (ov::op::v0::Parameter::*)()) & ov::op::v0::Parameter::get_partial_shape);
    parameter.def("set_partial_shape", &ov::op::v0::Parameter::set_partial_shape, py::arg("partial_shape"));

    parameter.def("get_element_type", &ov::op::v0::Parameter::get_element_type);

    parameter.def("set_element_type", &ov::op::v0::Parameter::set_element_type, py::arg("element_type"));

    parameter.def("get_layout", &ov::op::v0::Parameter::get_layout);

    parameter.def("set_layout", &ov::op::v0::Parameter::set_layout, py::arg("layout"));

    parameter.def_property("partial_shape",
                           (ov::PartialShape & (ov::op::v0::Parameter::*)()) & ov::op::v0::Parameter::get_partial_shape,
                           &ov::op::v0::Parameter::set_partial_shape);

    parameter.def_property("element_type",
                           &ov::op::v0::Parameter::get_element_type,
                           &ov::op::v0::Parameter::set_element_type);

    parameter.def_property("layout", &ov::op::v0::Parameter::get_layout, &ov::op::v0::Parameter::set_layout);

    parameter.def("__repr__", [](const ov::op::v0::Parameter& self) {
        std::stringstream shapes_ss;
        for (size_t i = 0; i < self.get_output_size(); ++i) {
            if (i > 0) {
                shapes_ss << ", ";
            }
            shapes_ss << self.get_output_partial_shape(i);
        }
        return "<" + Common::get_class_name(self) + ": '" + self.get_friendly_name() + "' (" + shapes_ss.str() + ")>";
    });
}
