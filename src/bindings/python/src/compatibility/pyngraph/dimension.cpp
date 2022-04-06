// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/dimension.hpp"  // ngraph::Dimension

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "pyngraph/dimension.hpp"

namespace py = pybind11;

void regclass_pyngraph_Dimension(py::module m) {
    using value_type = ngraph::Dimension::value_type;

    py::class_<ngraph::Dimension, std::shared_ptr<ngraph::Dimension>> dim(m, "Dimension", py::module_local());
    dim.doc() = "ngraph.impl.Dimension wraps ngraph::Dimension";
    dim.def(py::init<>());
    dim.def(py::init<value_type&>(),
            py::arg("dimension"),
            R"(
                Construct a static dimension.

                :param dimension: Value of the dimension.
                :type dimension: int
            )");
    dim.def(py::init<value_type&, value_type&>(),
            py::arg("min_dimension"),
            py::arg("max_dimension"),
            R"(
                Construct a dynamic dimension with bounded range.

                :param min_dimension: The lower inclusive limit for the dimension.
                :type min_dimension: int
                :param max_dimension: inclusive limit for the dimension.
                :type max_dimension: The upper inclusive limit for the dimension.
            )");

    dim.def_static("dynamic", &ngraph::Dimension::dynamic);

    dim.def_property_readonly("is_dynamic",
                              &ngraph::Dimension::is_dynamic,
                              R"(
                                Check if Dimension is dynamic.

                                :return: True if dynamic, else False.
                                :rtype: bool
                              )");
    dim.def_property_readonly("is_static",
                              &ngraph::Dimension::is_static,
                              R"(
                                Check if Dimension is static.

                                :return: True if static, else False.
                                :rtype: bool
                              )");

    dim.def(
        "__eq__",
        [](const ngraph::Dimension& a, const ngraph::Dimension& b) {
            return a == b;
        },
        py::is_operator());
    dim.def(
        "__eq__",
        [](const ngraph::Dimension& a, const int64_t& b) {
            return a == b;
        },
        py::is_operator());

    dim.def("__len__", &ngraph::Dimension::get_length);
    dim.def("get_length",
            &ngraph::Dimension::get_length,
            R"(
                Return this dimension as integer.
                This dimension must be static and non-negative.

                :return Value of the dimension.
                :rtype: int
            )");
    dim.def("get_min_length",
            &ngraph::Dimension::get_min_length,
            R"(
                Return this dimension's min_dimension as integer.
                This dimension must be dynamic and non-negative.

                :return: Value of the dimension.
                :rtype: int
            )");
    dim.def("get_max_length",
            &ngraph::Dimension::get_max_length,
            R"(
                Return this dimension's max_dimension as integer.
                This dimension must be dynamic and non-negative.

                :return: Value of the dimension.
                :rtype: int
            )");

    dim.def("same_scheme",
            &ngraph::Dimension::same_scheme,
            py::arg("dim"),
            R"(
                Return this dimension's max_dimension as integer.
                This dimension must be dynamic and non-negative.

                :param dim: The other dimension to compare this dimension to.
                :type dim: Dimension
                :return: True if this dimension and dim are both dynamic,
                or if they are both static and equal, otherwise False.
                :rtype: bool
            )");
    dim.def("compatible",
            &ngraph::Dimension::compatible,
            py::arg("d"),
            R"(
                Check whether this dimension is capable of being merged 
                with the argument dimension.

                :param d: The dimension to compare this dimension with.
                :type d: Dimension
                :return: True if this dimension is compatible with d, else False.
                :rtype: bool
            )");
    dim.def("relaxes",
            &ngraph::Dimension::relaxes,
            py::arg("d"),
            R"(
                Check whether this dimension is a relaxation of the argument.
                This dimension relaxes (or is a relaxation of) d if:

                (1) this and d are static and equal
                (2) this dimension contains d dimension

                this.relaxes(d) is equivalent to d.refines(this).

                :param d: The dimension to compare this dimension with.
                :type d: Dimension
                :return: True if this dimension relaxes d, else False.
                :rtype: bool
            )");
    dim.def("refines",
            &ngraph::Dimension::refines,
            py::arg("d"),
            R"(
                Check whether this dimension is a refinement of the argument.
                This dimension refines (or is a refinement of) d if:

                (1) this and d are static and equal
                (2) d dimension contains this dimension

                this.refines(d) is equivalent to d.relaxes(this).

                :param d: The dimension to compare this dimension with.
                :type d: Dimension
                :return: True if this dimension refines d, else False.
                :rtype: bool
            )");

    dim.def("__str__", [](const ngraph::Dimension& self) -> std::string {
        std::stringstream ss;
        ss << self;
        return ss.str();
    });

    dim.def("__repr__", [](const ngraph::Dimension& self) -> std::string {
        return "<Dimension: " + py::cast(self).attr("__str__")().cast<std::string>() + ">";
    });
}
