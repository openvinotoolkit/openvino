// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/util.hpp"

#include <pybind11/numpy.h>

#include <openvino/core/node_output.hpp>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/pass/manager.hpp"
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
    py::module mod = m.def_submodule("util", "openvino.runtime.utils");
    mod.def("numpy_to_c", &numpy_to_c);
    OPENVINO_SUPPRESS_DEPRECATED_START
    mod.def("get_constant_from_source",
            &ov::get_constant_from_source,
            py::arg("output"),
            R"(
                Runs an estimation of source tensor.

                :param index: Output node.
                :type index: openvino.runtime.Output
                :return: If it succeeded to calculate both bounds and
                         they are the same, returns Constant operation
                         from the resulting bound, otherwise Null.
                :rtype: openvino.runtime.op.Constant or openvino.runtime.Node
            )");
    OPENVINO_SUPPRESS_DEPRECATED_END
    mod.def(
        "clone_model",
        [](ov::Model& model) {
            return model.clone();
        },
        py::arg("model"),
        R"(
                Creates a copy of a model object.

                :param model: Model to copy.
                :type model: openvino.runtime.Model
                :return: A copy of Model.
                :rtype: openvino.runtime.Model
            )");

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
        [](const std::string& function_name, const std::string& version, const std::string& message) {
            Common::utils::deprecation_warning(function_name, version, message);
        },
        py::arg("function_name"),
        py::arg("version") = "",
        py::arg("message") = "",
        R"(
            Prints deprecation warning "{function_name} is deprecated and will be removed in version {version}. {message}".

            :param function_name: The name of the deprecated function.
            :param version: The version in which the code will be removed.
            :param message: A message explaining why the function is deprecated.
        )");
}
