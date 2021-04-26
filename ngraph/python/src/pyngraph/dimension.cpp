// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    dim.def(py::init<value_type&>(),
            py::arg("dimension"),
            R"mydelimiter(
                Construct a static dimension.

                Parameters
                ----------
                 dimension : int
                    Value of the dimension.
            )mydelimiter");
    dim.def(py::init<value_type&, value_type&>(),
            py::arg("min_dimension"),
            py::arg("max_dimension"),
            R"mydelimiter(
                Construct a dynamic dimension with bounded range.

                Parameters
                ----------
                min_dimension : int
                    The lower inclusive limit for the dimension.

                max_dimension : int
                    The upper inclusive limit for the dimension.
            )mydelimiter");

    dim.def_static("dynamic", &ngraph::Dimension::dynamic);

    dim.def_property_readonly("is_dynamic",
                              &ngraph::Dimension::is_dynamic,
                              R"mydelimiter(
                                Check if Dimension is dynamic.

                                Returns
                                ----------
                                is_dynamic : bool
                                    False if dynamic, else True.
                              )mydelimiter");
    dim.def_property_readonly("is_static",
                              &ngraph::Dimension::is_static,
                              R"mydelimiter(
                                Check if Dimension is static.

                                Returns
                                ----------
                                is_static : bool
                                    False if static, else True.
                              )mydelimiter");

    dim.def(
        "__eq__",
        [](const ngraph::Dimension& a, const ngraph::Dimension& b) { return a == b; },
        py::is_operator());
    dim.def(
        "__eq__",
        [](const ngraph::Dimension& a, const int64_t& b) { return a == b; },
        py::is_operator());

    dim.def("__len__", &ngraph::Dimension::get_length);
    dim.def("get_length",
            &ngraph::Dimension::get_length,
            R"mydelimiter(
                Convert this dimension to integer.
                This dimension must be static and non-negative.

                Returns
                ----------
                get_length : int
                    Value of the dimension.
            )mydelimiter");
    dim.def("get_min_length",
            &ngraph::Dimension::get_min_length,
            R"mydelimiter(
                Convert this dimension's min_dimension to integer.
                This dimension must be dynamic and non-negative.

                Returns
                ----------
                get_min_length : int
                    Value of the dimension.
            )mydelimiter");
    dim.def("get_max_length",
            &ngraph::Dimension::get_max_length,
            R"mydelimiter(
                Convert this dimension's max_dimension to integer.
                This dimension must be dynamic and non-negative.

                Returns
                ----------
                get_max_length : int
                    Value of the dimension.
            )mydelimiter");

    dim.def("same_scheme",
            &ngraph::Dimension::same_scheme,
            py::arg("dim"),
            R"mydelimiter(
                Convert this dimension's max_dimension to integer.
                This dimension must be dynamic and non-negative.

                Parameters
                ----------
                dim : Dimension
                    The other dimension to compare this dimension to.

                Returns
                ----------
                same_scheme : bool
                    True if this dimension and dim are both dynamic,
                    or if they are both static and equal, otherwise False.
            )mydelimiter");
    dim.def("compatible",
            &ngraph::Dimension::compatible,
            py::arg("d"),
            R"mydelimiter(
                Check whether this dimension is capable of being merged 
                with the argument dimension.

                Parameters
                ----------
                d : Dimension
                    The dimension to compare this dimension with.

                Returns
                ----------
                compatible : bool
                    True if this dimension is compatible with d, else False.
            )mydelimiter");
    dim.def("relaxes",
            &ngraph::Dimension::relaxes,
            py::arg("d"),
            R"mydelimiter(
                Check whether this dimension is a relaxation of the argument.
                
                Parameters
                ----------
                d : Dimension
                    The dimension to compare this dimension with.

                Returns
                ----------
                relaxes : bool
                    True if this dimension relaxes d, else False.
            )mydelimiter");
    dim.def("refines",
            &ngraph::Dimension::refines,
            py::arg("d"),
            R"mydelimiter(
                Check whether this dimension is a refinement of the argument.

                Parameters
                ----------
                d : Dimension
                    The dimension to compare this dimension with.

                Returns
                ----------
                relaxes : bool
                    True if this dimension refines d, else False.
            )mydelimiter");

    dim.def("__str__", [](const ngraph::Dimension& self) -> std::string {
        std::stringstream ss;
        ss << self;
        return ss.str();
    });

    dim.def("__repr__", [](const ngraph::Dimension& self) -> std::string {
        return "<Dimension: " + py::cast(self).attr("__str__")().cast<std::string>() + ">";
    });
}
