// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "ngraph/axis_set.hpp" // ngraph::AxisSet
#include "pyngraph/axis_set.hpp"

namespace py = pybind11;

void regclass_pyngraph_AxisSet(py::module m)
{
    py::class_<ngraph::AxisSet, std::shared_ptr<ngraph::AxisSet>> axis_set(m, "AxisSet");
    axis_set.doc() = "ngraph.impl.AxisSet wraps ngraph::AxisSet";
    axis_set.def(py::init<const std::initializer_list<size_t>&>(), py::arg("axes"));
    axis_set.def(py::init<const std::set<size_t>&>(), py::arg("axes"));
    axis_set.def(py::init<const std::vector<size_t>&>(), py::arg("axes"));
    axis_set.def(py::init<const ngraph::AxisSet&>(), py::arg("axes"));

    axis_set.def("__len__", [](const ngraph::AxisSet& v) { return v.size(); });

    axis_set.def(
        "__iter__",
        [](ngraph::AxisSet& v) { return py::make_iterator(v.begin(), v.end()); },
        py::keep_alive<0, 1>()); /* Keep set alive while iterator is used */

    axis_set.def("__repr__", [](const ngraph::AxisSet& self) -> std::string {
        std::stringstream data_ss;
        std::copy(self.begin(), self.end(), std::ostream_iterator<int>(data_ss, ", "));
        std::string data_str = data_ss.str();
        return "<AxisSet {" + data_str.substr(0, data_str.size() - 2) + "}>";
    });
}
