// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/util.hpp"

#include <pybind11/numpy.h>

#include <openvino/core/node_output.hpp>

#include "openvino/core/graph_util.hpp"
#include "openvino/pass/manager.hpp"
#include "pyopenvino/graph/node.hpp"
#include "pyopenvino/graph/node_output.hpp"
#include "pyopenvino/graph/ops/constant.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

inline void* numpy_to_c(py::array a) {
    py::buffer_info info = a.request();
    return info.ptr;
}

void regmodule_graph_util(py::module m) {
    py::module mod = m.def_submodule("util", "openvino.utils");
    mod.def("numpy_to_c", &numpy_to_c);

    mod.def("replace_output_update_name", &ov::replace_output_update_name, py::arg("output"), py::arg("target_output"));

    mod.def("replace_node",
            overload_cast_<const std::shared_ptr<ov::Node>&, const std::shared_ptr<ov::Node>&>()(&ov::replace_node),
            py::arg("target"),
            py::arg("replacement"));

    mod.def("replace_node",
            overload_cast_<const std::shared_ptr<ov::Node>&, const ov::OutputVector&>()(&ov::replace_node),
            py::arg("target"),
            py::arg("replacement"));

    mod.def("replace_node",
            overload_cast_<const std::shared_ptr<ov::Node>&,
                           const std::shared_ptr<ov::Node>&,
                           const std::vector<int64_t>&>()(&ov::replace_node),
            py::arg("target"),
            py::arg("replacement"),
            py::arg("outputs_order"));

    mod.def(
        "deprecation_warning",
        [](const std::string& function_name, const std::string& version, const std::string& message, int stacklevel) {
            Common::utils::deprecation_warning(function_name, version, message);
        },
        py::arg("function_name"),
        py::arg("version") = "",
        py::arg("message") = "",
        py::arg("stacklevel") = 2,
        R"(
            Prints deprecation warning "{function_name} is deprecated and will be removed in version {version}. {message}".

            :param function_name: The name of the deprecated function.
            :param version: The version in which the code will be removed.
            :param message: A message explaining why the function is deprecated.
            :param stacklevel: How many layers should be propagated.
        )");
}
