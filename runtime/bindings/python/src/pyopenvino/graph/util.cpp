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

std::shared_ptr<ov::op::v0::Constant> _py_get_constant_from_source(const ov::Output<ov::Node>& source) {
    return ov::get_constant_from_source(source);
}


void regmodule_graph_util(py::module m) {
    py::module mod = m.def_submodule("util", "ngraph.impl.util");
    mod.def("numpy_to_c", &numpy_to_c);
    mod.def("get_constant_from_source",
            &_py_get_constant_from_source,
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
}
