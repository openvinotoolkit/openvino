// Copyright (C) 2018-2026 Intel Corporation
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
#include "transformations/rt_info/disable_precision_conversion.hpp"

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

    using NodePtr = std::shared_ptr<ov::Node>;
    using ov::element::Type;
    using ov::element::TypeVector;

    mod.def("disable_conversion",
            overload_cast_<const NodePtr&, const Type&>()(&ov::disable_conversion),
            py::arg("node"),
            py::arg("to_type"),
            R"(
                Keeps a node in its original precision: marks it so that it is not converted
                to `to_type` (from any type), e.g. by the f16 compression a GPU applies under
                the f16 inference precision hint.

                Python equivalent of the C++ `ov::disable_conversion(node, to)`.

                :param node: The node to mark.
                :type node: openvino.Node
                :param to_type: The element type the node must not be converted to, e.g. `Type.f16`.
                :type to_type: openvino.Type
            )");

    mod.def("disable_conversion",
            overload_cast_<const NodePtr&, const Type&, const Type&>()(&ov::disable_conversion),
            py::arg("node"),
            py::arg("from_type"),
            py::arg("to_type"),
            R"(
                Marks a node so that it is not converted from `from_type` to `to_type`.

                :param node: The node to mark.
                :type node: openvino.Node
                :param from_type: The source element type.
                :type from_type: openvino.Type
                :param to_type: The target element type.
                :type to_type: openvino.Type
            )");

    mod.def("disable_conversion",
            overload_cast_<const NodePtr&, const TypeVector&, const TypeVector&>()(&ov::disable_conversion),
            py::arg("node"),
            py::arg("from_types"),
            py::arg("to_types"),
            R"(
                Marks a node so that it is not converted from any of `from_types` to any of `to_types`.

                :param node: The node to mark.
                :type node: openvino.Node
                :param from_types: The source element types.
                :type from_types: list[openvino.Type]
                :param to_types: The target element types.
                :type to_types: list[openvino.Type]
            )");

    mod.def("enable_conversion",
            overload_cast_<const NodePtr&, const Type&>()(&ov::enable_conversion),
            py::arg("node"),
            py::arg("to_type"),
            R"(
                Removes a mark set by `disable_conversion(node, to_type)`.

                :param node: The node to unmark.
                :type node: openvino.Node
                :param to_type: The target element type.
                :type to_type: openvino.Type
            )");

    mod.def("enable_conversion",
            overload_cast_<const NodePtr&, const Type&, const Type&>()(&ov::enable_conversion),
            py::arg("node"),
            py::arg("from_type"),
            py::arg("to_type"),
            R"(
                Removes a mark set by `disable_conversion(node, from_type, to_type)`.

                :param node: The node to unmark.
                :type node: openvino.Node
                :param from_type: The source element type.
                :type from_type: openvino.Type
                :param to_type: The target element type.
                :type to_type: openvino.Type
            )");

    mod.def("enable_conversion",
            overload_cast_<const NodePtr&, const TypeVector&, const TypeVector&>()(&ov::enable_conversion),
            py::arg("node"),
            py::arg("from_types"),
            py::arg("to_types"),
            R"(
                Removes the marks set by `disable_conversion(node, from_types, to_types)`.

                :param node: The node to unmark.
                :type node: openvino.Node
                :param from_types: The source element types.
                :type from_types: list[openvino.Type]
                :param to_types: The target element types.
                :type to_types: list[openvino.Type]
            )");

    mod.def(
        "is_conversion_disabled",
        [](const NodePtr& node, const Type& to_type) {
            return ov::is_conversion_disabled(node, to_type);
        },
        py::arg("node"),
        py::arg("to_type"),
        R"(
                Checks whether converting the node to `to_type` is disabled.

                :param node: The node to check.
                :type node: openvino.Node
                :param to_type: The target element type.
                :type to_type: openvino.Type
                :return: True if the conversion is disabled.
                :rtype: bool
            )");

    mod.def(
        "is_conversion_disabled",
        [](const NodePtr& node, const Type& from_type, const Type& to_type) {
            return ov::is_conversion_disabled(node, from_type, to_type);
        },
        py::arg("node"),
        py::arg("from_type"),
        py::arg("to_type"),
        R"(
                Checks whether converting the node from `from_type` to `to_type` is disabled.

                :param node: The node to check.
                :type node: openvino.Node
                :param from_type: The source element type.
                :type from_type: openvino.Type
                :param to_type: The target element type.
                :type to_type: openvino.Type
                :return: True if the conversion is disabled.
                :rtype: bool
            )");

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
