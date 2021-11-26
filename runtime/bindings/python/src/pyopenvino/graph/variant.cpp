// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/variant.hpp"

#include <pybind11/pybind11.h>

#include "openvino/core/any.hpp"

namespace py = pybind11;

void regclass_graph_Variant(py::module m) {
    py::class_<ov::Any> variant_base(m, "Variant", py::module_local());
    variant_base.doc() = "openvino.impl.Variant wraps ov::Any";

    variant_base.def(
        "__eq__",
        [](const ov::Any& a, const ov::Any& b) {
            return a == b;
        },
        py::is_operator());
    variant_base.def(
        "__eq__",
        [](const ov::Any& a, const std::string& b) {
            return a.as<std::string>() == b;
        },
        py::is_operator());
    variant_base.def(
        "__eq__",
        [](const ov::Any& a, const int64_t& b) {
            return a.as<int64_t>() == b;
        },
        py::is_operator());

    variant_base.def("__repr__", [](const ov::Any self) {
        std::stringstream ret;
        self.print(ret);
        return ret.str();
    });
}

template void regclass_graph_VariantWrapper<std::string>(py::module m, std::string typestring);
template void regclass_graph_VariantWrapper<int64_t>(py::module m, std::string typestring);
