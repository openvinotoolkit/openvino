// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/type/element_type.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngraph/op/parameter.hpp"
#include "pyngraph/types/element_type.hpp"

namespace py = pybind11;

void regclass_pyngraph_Type(py::module m) {
    py::class_<ngraph::element::Type, std::shared_ptr<ngraph::element::Type>> type(m, "Type", py::module_local());
    type.doc() = "ngraph.impl.Type wraps ngraph::element::Type";
    type.attr("boolean") = ngraph::element::boolean;
    type.attr("f16") = ngraph::element::f16;
    type.attr("f32") = ngraph::element::f32;
    type.attr("f64") = ngraph::element::f64;
    type.attr("i8") = ngraph::element::i8;
    type.attr("i16") = ngraph::element::i16;
    type.attr("i32") = ngraph::element::i32;
    type.attr("i64") = ngraph::element::i64;
    type.attr("u1") = ngraph::element::u1;
    type.attr("u8") = ngraph::element::u8;
    type.attr("u16") = ngraph::element::u16;
    type.attr("u32") = ngraph::element::u32;
    type.attr("u64") = ngraph::element::u64;
    type.attr("bf16") = ngraph::element::bf16;

    type.def("__repr__", [](const ngraph::element::Type& self) {
        std::string bitwidth = std::to_string(self.bitwidth());
        if (self.is_signed()) {
            return "<Type: '" + self.c_type_string() + bitwidth + "'>";
        }
        return "<Type: 'u" + self.c_type_string() + bitwidth + "'>";
    });

    type.def(
        "__eq__",
        [](const ngraph::element::Type& a, const ngraph::element::Type& b) {
            return a == b;
        },
        py::is_operator());

    type.def_property_readonly("bitwidth", &ngraph::element::Type::bitwidth);
    type.def_property_readonly("is_real", &ngraph::element::Type::is_real);
    type.def("get_type_name", &ngraph::element::Type::get_type_name);
    type.def("to_string", &ngraph::element::Type::to_string);
}
