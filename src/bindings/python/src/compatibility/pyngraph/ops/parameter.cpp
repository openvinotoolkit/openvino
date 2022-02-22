// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/parameter.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "ngraph/node.hpp"
#include "ngraph/partial_shape.hpp"  // ngraph::PartialShape
#include "pyngraph/ops/parameter.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_Parameter(py::module m) {
    py::class_<ngraph::op::Parameter, std::shared_ptr<ngraph::op::Parameter>, ngraph::Node> parameter(
        m,
        "Parameter",
        py::module_local());
    parameter.doc() = "ngraph.impl.op.Parameter wraps ngraph::op::Parameter";
    parameter.def("__repr__", [](const ngraph::Node& self) {
        std::string class_name = py::cast(self).get_type().attr("__name__").cast<std::string>();
        std::string shape = py::cast(self.get_output_partial_shape(0)).attr("__str__")().cast<std::string>();
        std::string type = self.get_element_type().c_type_string();
        return "<" + class_name + ": '" + self.get_friendly_name() + "' (" + shape + ", " + type + ")>";
    });

    parameter.def(py::init<const ngraph::element::Type&, const ngraph::Shape&>());
    parameter.def(py::init<const ngraph::element::Type&, const ngraph::PartialShape&>());
    //    parameter.def_property_readonly("description", &ngraph::op::Parameter::description);

    parameter.def(
        "get_partial_shape",
        (const ngraph::PartialShape& (ngraph::op::Parameter::*)() const) & ngraph::op::Parameter::get_partial_shape);
    parameter.def("get_partial_shape",
                  (ngraph::PartialShape & (ngraph::op::Parameter::*)()) & ngraph::op::Parameter::get_partial_shape);
    parameter.def("set_partial_shape", &ngraph::op::Parameter::set_partial_shape);
}
