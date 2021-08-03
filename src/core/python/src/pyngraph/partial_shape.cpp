// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

    shape.def_property_readonly("is_dynamic",
                                &ngraph::PartialShape::is_dynamic,
                                R"(
                                    False if this shape is static, else True.
                                    A shape is considered static if it has static rank,
                                    and all dimensions of the shape are static.
                                )");
    shape.def_property_readonly("is_static",
                                &ngraph::PartialShape::is_static,
                                R"(
                                    True if this shape is static, else False.
                                    A shape is considered static if it has static rank, 
                                    and all dimensions of the shape are static.
                                )");
    shape.def_property_readonly("rank",
                                &ngraph::PartialShape::rank,
                                R"(
                                    The rank of the shape.
                                )");
    shape.def_property_readonly("all_non_negative",
                                &ngraph::PartialShape::all_non_negative,
                                R"(
                                    True if all static dimensions of the tensor are 
                                    non-negative, else False.
                                )");

    shape.def("compatible",
              &ngraph::PartialShape::compatible,
              py::arg("s"),
              R"(
                Check whether this shape is compatible with the argument, i.e.,
                whether it is possible to merge them.
                
                Parameters
                ----------
                s : PartialShape
                    The shape to be checked for compatibility with this shape.


                Returns
                ----------
                compatible : bool
                    True if this shape is compatible with s, else False.
              )");
    shape.def("refines",
              &ngraph::PartialShape::refines,
              py::arg("s"),
              R"(
                Check whether this shape is a refinement of the argument.

                Parameters
                ----------
                s : PartialShape
                    The shape which is being compared against this shape.        
        
                Returns
                ----------
                refines : bool
                    True if this shape refines s, else False.
              )");
    shape.def("relaxes",
              &ngraph::PartialShape::relaxes,
              py::arg("s"),
              R"(
                Check whether this shape is a relaxation of the argument.

                Parameters
                ----------
                s : PartialShape
                    The shape which is being compared against this shape.        
        
                Returns
                ----------
                relaxes : bool
                    True if this shape relaxes s, else False.
              )");
    shape.def("same_scheme",
              &ngraph::PartialShape::same_scheme,
              py::arg("s"),
              R"(
                Check whether this shape represents the same scheme as the argument.

                Parameters
                ----------
                s : PartialShape
                    The shape which is being compared against this shape.        
        
                Returns
                ----------
                same_scheme : bool
                    True if shape represents the same scheme as s, else False.
              )");
    shape.def("get_max_shape",
              &ngraph::PartialShape::get_max_shape,
              R"(
                Returns
                ----------
                get_max_shape : Shape
                    Get the max bounding shape.
              )");
    shape.def("get_min_shape",
              &ngraph::PartialShape::get_min_shape,
              R"(
                Returns
                ----------
                get_min_shape : Shape
                    Get the min bounding shape.
              )");
    shape.def("get_shape",
              &ngraph::PartialShape::get_shape,
              R"(
                Returns
                ----------
                get_shape : Shape
                    Get the unique shape.
              )");
    shape.def("to_shape",
              &ngraph::PartialShape::to_shape,
              R"(
                Returns
                ----------
                to_shapess : Shape
                    Get the unique shape.
              )");
    shape.def(
        "get_dimension",
        [](const ngraph::PartialShape& self, size_t index) -> ngraph::Dimension {
            return self[index];
        },
        py::arg("index"),
        R"(
                Get the dimension at specified index of a partial shape.

                Parameters
                ----------
                index : int
                    The index of dimension

                Returns
                ----------
                get_dimension : Dimension
                    Get the particular dimension of a partial shape.
              )");

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
