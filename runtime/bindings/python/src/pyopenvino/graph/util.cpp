// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/util.hpp"

#include <pybind11/numpy.h>

#include "openvino/core/validation_util.hpp"

namespace py = pybind11;

void* numpy_to_c(py::array a) {
    py::buffer_info info = a.request();
    return info.ptr;
}

void regmodule_graph_util(py::module m) {
    py::module mod = m.def_submodule("util", "ngraph.impl.util");
    mod.def("numpy_to_c", &numpy_to_c);
    mod.def("get_constant_from_source",
            &ov::get_constant_from_source,
            py::arg("output"),
            R"(
                    Runs an estimation of source tensor.

                    Parameters
                    ----------
                    output : Output
                        output node

                    Returns
                    ----------
                    get_constant_from_source : Constant or None
                        If it succeeded to calculate both bounds and
                        they are the same returns Constant operation
                        from the resulting bound, otherwise Null.
                )");
    mod.def(
        "get_tensor_name",
        [](const std::shared_ptr<ov::op::v0::Result> result) {
            const auto prev_node_output = result->input_value(0);
            return prev_node_output.get_tensor().get_name();
        },
        py::arg("result"),
        R"(
                Returns the name of a tensor associated with a given Result object (Function output).
                Parameters
                ----------
                result : Result
                    result node
                Returns
                ----------
                get_tensor_name : string
                    Returns the tensor name if it's set.
                    Otherwise returns an empty string.
            )");
}
