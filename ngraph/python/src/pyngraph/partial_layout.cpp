// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/partial_layout.hpp"  // ngraph::PartialShape

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "pyngraph/partial_layout.hpp"

namespace py = pybind11;

void regclass_pyngraph_PartialLayout(py::module m) {
    py::class_<ov::PartialLayout, std::shared_ptr<ov::PartialLayout>> layout(m, "PartialLayout");
    layout.doc() = "ngraph.impl.PartialLayout wraps ov::PartialLayout";

    layout.def(py::init<>());
    layout.def(py::init<const std::string&>());

    layout.def_property_readonly("is_empty",
                                 &ov::PartialLayout::is_empty,
                                 R"(
                                    True if this layout is empty, else True.
                                )");
    layout.def_property_readonly("size",
                                 &ov::PartialLayout::size,
                                 R"(
                                    The size/rank of the layout.
                                )");
    layout.def_property(
        "channels",
        [](const ov::PartialLayout& l) -> py::object {
            py::object res = py::none();
            if (ov::layouts::has_channels(l)) {
                res = py::int_(ov::layouts::channels(l));
            }
            return res;
        },
        [](ov::PartialLayout& l, int c) {
            ov::layouts::set_channels(l, c);
        },
        R"(
                Property representing channels index of layout. E.g. 'NCHW' will have `layout.channels = 1`
                Can be `None` if layout doesn't have channels attribute
              )");
    //    shape.def(
    //        "__eq__",
    //        [](const ngraph::PartialShape& a, const ngraph::Shape& b) {
    //            return a == b;
    //        },
    //        py::is_operator());
    //
    //    shape.def("__str__", [](const ngraph::PartialShape& self) -> std::string {
    //        std::stringstream ss;
    //        ss << self;
    //        return ss.str();
    //    });
    //
    //    shape.def("__repr__", [](const ngraph::PartialShape& self) -> std::string {
    //        return "<PartialShape: " + py::cast(self).attr("__str__")().cast<std::string>() + ">";
    //    });
}
