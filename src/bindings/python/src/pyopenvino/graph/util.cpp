// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/util.hpp"
#include "openvino/core/graph_util.hpp"

#include <pybind11/numpy.h>

#include "openvino/core/validation_util.hpp"

namespace py = pybind11;

void* numpy_to_c(py::array a) {
    py::buffer_info info = a.request();
    return info.ptr;
}

void regmodule_graph_util(py::module m) {
    py::module mod = m.def_submodule("util", "openvino.runtime.util");
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
    mod.def("clone_function",
            (std::shared_ptr<ov::Model>(ov::*)(const ov::Model&)) & ov::clone_model,
            py::arg("model"),
            R"(
                Create a copy of Model.

                :param model: Model to copy.
                :type model: openvino.runtime.Model
                :return: A copy of Model.
                :rtype: openvino.runtime.Model
            )");
