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

#include "ngraph/dimension.hpp"     // ngraph::Dimension
#include "ngraph/partial_shape.hpp" // ngraph::PartialShape
#include "ngraph/shape.hpp"         // ngraph::Shape
#include "pyngraph/partial_shape.hpp"

namespace py = pybind11;

void regclass_pyngraph_PartialShape(py::module m)
{
    py::class_<ngraph::PartialShape, std::shared_ptr<ngraph::PartialShape>> shape(m,
                                                                                  "PartialShape");
    shape.doc() = "ngraph.impl.PartialShape wraps ngraph::PartialShape";

    shape.def(py::init([](const std::vector<int64_t>& dimensions) {
        return ngraph::PartialShape(
            std::vector<ngraph::Dimension>(dimensions.begin(), dimensions.end()));
    }));
    shape.def(py::init<const std::initializer_list<size_t>&>());
    shape.def(py::init<const std::vector<size_t>&>());
    shape.def(py::init<const std::initializer_list<ngraph::Dimension>&>());
    shape.def(py::init<const std::vector<ngraph::Dimension>&>());
    shape.def(py::init<const ngraph::Shape&>());
    shape.def(py::init<const ngraph::PartialShape&>());

    shape.def_static("dynamic", &ngraph::PartialShape::dynamic, py::arg("r") = ngraph::Dimension());

    shape.def_property_readonly("is_dynamic", &ngraph::PartialShape::is_dynamic);
    shape.def_property_readonly("is_static", &ngraph::PartialShape::is_static);
    shape.def_property_readonly("rank", &ngraph::PartialShape::rank);
    shape.def_property_readonly("all_non_negative", &ngraph::PartialShape::all_non_negative);

    shape.def("compatible", &ngraph::PartialShape::compatible);
    shape.def("refines", &ngraph::PartialShape::refines);
    shape.def("relaxes", &ngraph::PartialShape::relaxes);
    shape.def("same_scheme", &ngraph::PartialShape::same_scheme);
    shape.def("get_max_shape", &ngraph::PartialShape::get_max_shape);
    shape.def("get_min_shape", &ngraph::PartialShape::get_min_shape);
    shape.def("get_shape", &ngraph::PartialShape::get_shape);
    shape.def("to_shape", &ngraph::PartialShape::to_shape);

    shape.def(
        "__eq__",
        [](const ngraph::PartialShape& a, const ngraph::PartialShape& b) { return a == b; },
        py::is_operator());
    shape.def(
        "__eq__",
        [](const ngraph::PartialShape& a, const ngraph::Shape& b) { return a == b; },
        py::is_operator());

    shape.def("__str__", [](const ngraph::PartialShape& self) -> std::string {
        std::stringstream ss;
        ss << self;
        return ss.str();
    });

    shape.def("__repr__", [](const ngraph::PartialShape& self) -> std::string {
        return "<PartialShape: " + py::cast(self).attr("__str__")().cast<std::string>() + ">";
    });
}
