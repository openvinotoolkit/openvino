// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/util.hpp"

#include <pybind11/numpy.h>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/pass/manager.hpp"

namespace py = pybind11;

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

void* numpy_to_c(py::array a) {
    py::buffer_info info = a.request();
    return info.ptr;
}

void regmodule_graph_util(py::module m) {
    py::module mod = m.def_submodule("util", "openvino.runtime.utils");
    mod.def("numpy_to_c", &numpy_to_c);
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
}
