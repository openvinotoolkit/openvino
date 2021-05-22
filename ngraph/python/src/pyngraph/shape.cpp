// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "ngraph/shape.hpp" // ngraph::Shape
#include "pyngraph/shape.hpp"

namespace py = pybind11;

void regclass_pyngraph_Shape(py::module m)
{
    py::class_<ngraph::Shape, std::shared_ptr<ngraph::Shape>> shape(m, "Shape");
    shape.doc() = "ngraph.impl.Shape wraps ngraph::Shape";
    shape.def(py::init<const std::initializer_list<size_t>&>(), py::arg("axis_lengths"));
    shape.def(py::init<const std::vector<size_t>&>(), py::arg("axis_lengths"));
    shape.def(py::init<const ngraph::Shape&>(), py::arg("axis_lengths"));
    shape.def("__len__", [](const ngraph::Shape& v) { return v.size(); });
    shape.def("__getitem__", [](const ngraph::Shape& v, int key) { return v[key]; });

    shape.def(
        "__iter__",
        [](ngraph::Shape& v) { return py::make_iterator(v.begin(), v.end()); },
        py::keep_alive<0, 1>()); /* Keep vector alive while iterator is used */

    shape.def("__str__", [](const ngraph::Shape& self) -> std::string {
        std::stringstream ss;
        ss << self;
        return ss.str();
    });

    shape.def("__repr__", [](const ngraph::Shape& self) -> std::string {
        return "<" + py::cast(self).attr("__str__")().cast<std::string>() + ">";
    });
}
