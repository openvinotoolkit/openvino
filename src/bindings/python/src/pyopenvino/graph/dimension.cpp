// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/dimension.hpp"  // ov::Dimension

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "pyopenvino/graph/dimension.hpp"

namespace py = pybind11;

void regclass_graph_Dimension(py::module m) {
    using value_type = ov::Dimension::value_type;

    py::class_<ov::Dimension, std::shared_ptr<ov::Dimension>> dim(m, "Dimension");
    dim.doc() = "openvino.runtime.Dimension wraps ov::Dimension";
    dim.def(py::init<>());
    dim.def(py::init<value_type&>(),
            py::arg("dimension"),
            R"(
                Construct a static dimension.

                Parameters
                ----------
                 dimension : int
                    Value of the dimension.
            )");
    dim.def(py::init<value_type&, value_type&>(),
            py::arg("min_dimension"),
            py::arg("max_dimension"),
            R"(
                Construct a dynamic dimension with bounded range.

                Parameters
                ----------
                min_dimension : int
                    The lower inclusive limit for the dimension.

                max_dimension : int
                    The upper inclusive limit for the dimension.
            )");

    dim.def_static("dynamic", &ov::Dimension::dynamic);

    dim.def_property_readonly("is_dynamic",
                              &ov::Dimension::is_dynamic,
                              R"(
                                Check if Dimension is dynamic.

                                Returns
                                ----------
                                is_dynamic : bool
                                    True if dynamic, else False.
                              )");
    dim.def_property_readonly("is_static",
                              &ov::Dimension::is_static,
                              R"(
                                Check if Dimension is static.

                                Returns
                                ----------
                                is_static : bool
                                    True if static, else False.
                              )");

    dim.def(
        "__eq__",
        [](const ov::Dimension& a, const ov::Dimension& b) {
            return a == b;
        },
        py::is_operator());
    dim.def(
        "__eq__",
        [](const ov::Dimension& a, const int64_t& b) {
            return a == b;
        },
        py::is_operator());

    dim.def("__len__", &ov::Dimension::get_length);
    dim.def("get_length",
            &ov::Dimension::get_length,
            R"(
                Return this dimension as integer.
                This dimension must be static and non-negative.

                Returns
                ----------
                get_length : int
                    Value of the dimension.
            )");
    dim.def("get_min_length",
            &ov::Dimension::get_min_length,
            R"(
                Return this dimension's min_dimension as integer.
                This dimension must be dynamic and non-negative.

                Returns
                ----------
                get_min_length : int
                    Value of the dimension.
            )");
    dim.def("get_max_length",
            &ov::Dimension::get_max_length,
            R"(
                Return this dimension's max_dimension as integer.
                This dimension must be dynamic and non-negative.

                Returns
                ----------
                get_max_length : int
                    Value of the dimension.
            )");

    dim.def("same_scheme",
            &ov::Dimension::same_scheme,
            py::arg("dim"),
            R"(
                Return this dimension's max_dimension as integer.
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
            )");
    dim.def("compatible",
            &ov::Dimension::compatible,
            py::arg("dim"),
            R"(
                Check whether this dimension is capable of being merged
                with the argument dimension.

                Parameters
                ----------
                dim : Dimension
                    The dimension to compare this dimension with.

                Returns
                ----------
                compatible : bool
                    True if this dimension is compatible with d, else False.
            )");
    dim.def("relaxes",
            &ov::Dimension::relaxes,
            py::arg("dim"),
            R"(
                Check whether this dimension is a relaxation of the argument.
                This dimension relaxes (or is a relaxation of) d if:

                (1) this and d are static and equal
                (2) this dimension contains d dimension

                this.relaxes(d) is equivalent to d.refines(this).

                Parameters
                ----------
                dim : Dimension
                    The dimension to compare this dimension with.

                Returns
                ----------
                relaxes : bool
                    True if this dimension relaxes d, else False.
            )");
    dim.def("refines",
            &ov::Dimension::refines,
            py::arg("dim"),
            R"(
                Check whether this dimension is a refinement of the argument.
                This dimension refines (or is a refinement of) d if:

                (1) this and d are static and equal
                (2) d dimension contains this dimension

                this.refines(d) is equivalent to d.relaxes(this).

                Parameters
                ----------
                dim : Dimension
                    The dimension to compare this dimension with.

                Returns
                ----------
                relaxes : bool
                    True if this dimension refines d, else False.
            )");

    dim.def("__str__", [](const ov::Dimension& self) -> std::string {
        std::stringstream ss;
        ss << self;
        return ss.str();
    });

    dim.def("__repr__", [](const ov::Dimension& self) -> std::string {
        return "<Dimension: " + py::cast(self).attr("__str__")().cast<std::string>() + ">";
    });
}
