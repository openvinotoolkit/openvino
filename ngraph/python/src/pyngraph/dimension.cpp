//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "ngraph/dimension.hpp" // ngraph::Dimension
#include "pyngraph/dimension.hpp"

namespace py = pybind11;

void regclass_pyngraph_Dimension(py::module m)
{
    using value_type = ngraph::Dimension::value_type;

    py::class_<ngraph::Dimension, std::shared_ptr<ngraph::Dimension>> dim(m, "Dimension");
    dim.doc() = "ngraph.impl.Dimension wraps ngraph::Dimension";
    dim.def(py::init<>());
    dim.def(py::init<value_type&>());
    dim.def(py::init<value_type&, value_type&>());

    dim.def_static("dynamic", &ngraph::Dimension::dynamic);

    dim.def_property_readonly("is_dynamic", &ngraph::Dimension::is_dynamic);
    dim.def_property_readonly("is_static", &ngraph::Dimension::is_static);

    dim.def(
        "__eq__",
        [](const ngraph::Dimension& a, const ngraph::Dimension& b) { return a == b; },
        py::is_operator());
    dim.def(
        "__eq__",
        [](const ngraph::Dimension& a, const int64_t& b) { return a == b; },
        py::is_operator());

    dim.def("__len__", &ngraph::Dimension::get_length);
    dim.def("get_length", &ngraph::Dimension::get_length);
    dim.def("get_min_length", &ngraph::Dimension::get_min_length);
    dim.def("get_max_length", &ngraph::Dimension::get_max_length);

    dim.def("same_scheme", &ngraph::Dimension::same_scheme);
    dim.def("compatible", &ngraph::Dimension::compatible);
    dim.def("relaxes", &ngraph::Dimension::relaxes);
    dim.def("refines", &ngraph::Dimension::refines);

    dim.def("__str__", [](const ngraph::Dimension& self) -> std::string {
        std::stringstream ss;
        ss << self;
        return ss.str();
    });

    dim.def("__repr__", [](const ngraph::Dimension& self) -> std::string {
        return "<Dimension: " + py::cast(self).attr("__str__")().cast<std::string>() + ">";
    });
}
