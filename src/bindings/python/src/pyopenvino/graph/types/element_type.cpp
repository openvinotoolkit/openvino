// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/op/parameter.hpp"
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/types/element_type.hpp"

namespace py = pybind11;

void regclass_graph_Type(py::module m) {
    py::class_<ov::element::Type, std::shared_ptr<ov::element::Type>> type(m, "Type");
    type.doc() = "openvino.runtime.Type wraps ov::element::Type";

    type.def(py::init([](py::object& np_literal) {
                 return Common::dtype_to_ov_type().at(py::str(py::dtype::from_args(np_literal)));
             }),
             py::arg("dtype"),
             R"(
            Convert numpy dtype into OpenVINO type

            :param dtype: numpy dtype
            :type dtype: numpy.dtype
            :return: OpenVINO type object
            :rtype: ov.Type
        )");

    type.attr("boolean") = ov::element::boolean;
    type.attr("f16") = ov::element::f16;
    type.attr("f32") = ov::element::f32;
    type.attr("f64") = ov::element::f64;
    type.attr("i4") = ov::element::i4;
    type.attr("i8") = ov::element::i8;
    type.attr("i16") = ov::element::i16;
    type.attr("i32") = ov::element::i32;
    type.attr("i64") = ov::element::i64;
    type.attr("u1") = ov::element::u1;
    type.attr("u4") = ov::element::u4;
    type.attr("u8") = ov::element::u8;
    type.attr("u16") = ov::element::u16;
    type.attr("u32") = ov::element::u32;
    type.attr("u64") = ov::element::u64;
    type.attr("bf16") = ov::element::bf16;
    type.attr("undefined") = ov::element::undefined;

    type.def("__repr__", [](const ov::element::Type& self) {
        std::string bitwidth = std::to_string(self.bitwidth());
        if (self == ov::element::undefined) {
            return "<Type: '" + self.c_type_string() + "'>";
        } else if (self.is_signed()) {
            return "<Type: '" + self.c_type_string() + bitwidth + "'>";
        }
        return "<Type: 'u" + self.c_type_string() + bitwidth + "'>";
    });
    type.def("__hash__", &ov::element::Type::hash);
    type.def(
        "__eq__",
        [](const ov::element::Type& a, const ov::element::Type& b) {
            return a == b;
        },
        py::is_operator());
    type.def("get_type_name", &ov::element::Type::get_type_name);
    type.def(
        "to_dtype",
        [](ov::element::Type& self) {
            return Common::ov_type_to_dtype().at(self);
        },
        R"(
            Convert Type to numpy dtype

            :return: dtype object
            :rtype: numpy.dtype
        )");

    type.def_property_readonly("bitwidth", &ov::element::Type::bitwidth);
    type.def_property_readonly("is_real", &ov::element::Type::is_real);
}
