// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "pyngraph/variant.hpp"

namespace py = pybind11;

void regclass_pyngraph_Variant(py::module m) {
    py::class_<ov::Any> variant_base(m, "Variant", py::module_local());
    variant_base.doc() = "ngraph.impl.Variant wraps ngraph::Variant";

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
