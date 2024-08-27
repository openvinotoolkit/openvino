// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/symbol.hpp"  // ov::Symbol

#include <pybind11/pybind11.h>

#include "pyopenvino/graph/symbol.hpp"

namespace py = pybind11;

void regclass_graph_Symbol(py::module m) {
    py::class_<ov::Symbol, std::shared_ptr<ov::Symbol>> symbol(m, "Symbol");
    symbol.doc() = "openvino.runtime.Symbol wraps ov::Symbol";

    symbol.def(py::init([]() {
        return std::make_shared<ov::Symbol>();
    }));

    symbol.def(
        "__eq__",
        [](const std::shared_ptr<ov::Symbol>& a, const std::shared_ptr<ov::Symbol>& b) {
            return ov::symbol::are_equal(a, b);
        },
        py::is_operator());

    symbol.def(
        "__bool__",
        [](const std::shared_ptr<ov::Symbol>& self) -> bool {
            return self != nullptr;
        },
        "Check whether the symbol is meaningful");

    symbol.def("__hash__", [](const std::shared_ptr<ov::Symbol>& self) {
        auto ancestor = ov::symbol::ancestor_of(self);
        return std::hash<std::shared_ptr<ov::Symbol>>{}(ancestor);
    });
}
