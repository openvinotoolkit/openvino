// Copyright (C) 2018-2024 Intel Corporation
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
                 auto dtype = py::dtype::from_args(np_literal);
                 return Common::type_helpers::get_ov_type(dtype);
             }),
             py::arg("dtype"),
             R"(
            Convert numpy dtype into OpenVINO type

            :param dtype: numpy dtype
            :type dtype: numpy.dtype
            :return: OpenVINO type object
            :rtype: ov.Type
        )");

    type.attr("undefined") = ov::element::undefined;
    type.attr("dynamic") = ov::element::dynamic;
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
    type.attr("nf4") = ov::element::nf4;
    type.attr("f8e4m3") = ov::element::f8e4m3;
    type.attr("f8e5m2") = ov::element::f8e5m2;
    type.attr("string") = ov::element::string;
    type.attr("f4e2m1") = ov::element::f4e2m1;
    type.attr("f8e8m0") = ov::element::f8e8m0;

    type.def("__hash__", &ov::element::Type::hash);
    type.def("__repr__", [](const ov::element::Type& self) {
        std::string class_name = Common::get_class_name(self);
        if (self == ov::element::f32 || self == ov::element::f64) {
            std::string bitwidth = std::to_string(self.bitwidth());
            return "<" + class_name + ": '" + self.c_type_string() + bitwidth + "'>";
        }

        return "<" + class_name + ": '" + self.c_type_string() + "'>";
    });
    type.def(
        "__eq__",
        [](const ov::element::Type& a, const ov::element::Type& b) {
            return a == b;
        },
        py::is_operator());

    type.def("is_static", &ov::element::Type::is_static);
    type.def("is_dynamic", &ov::element::Type::is_dynamic);
    type.def("is_real", &ov::element::Type::is_real);
    type.def_property_readonly("real", &ov::element::Type::is_real);
    type.def("is_integral", &ov::element::Type::is_integral);
    type.def_property_readonly("integral", &ov::element::Type::is_integral);
    type.def("is_integral_number", &ov::element::Type::is_integral_number);
    type.def_property_readonly("integral_number", &ov::element::Type::is_integral_number);
    type.def("is_signed", &ov::element::Type::is_signed);
    type.def_property_readonly("signed", &ov::element::Type::is_signed);
    type.def("is_quantized", &ov::element::Type::is_quantized);
    type.def_property_readonly("quantized", &ov::element::Type::is_quantized);
    type.def("to_string", &ov::element::Type::to_string);
    type.def("get_type_name", &ov::element::Type::get_type_name);
    type.def_property_readonly("type_name", &ov::element::Type::get_type_name);
    type.def("compatible",
             &ov::element::Type::compatible,
             py::arg("other"),
             R"(
                Checks whether this element type is merge-compatible with
                `other`.

                :param other: The element type to compare this element type to.
                :type other: openvino.runtime.Type
                :return: `True` if element types are compatible, otherwise `False`.
                :rtype: bool
             )");
    type.def(
        "merge",
        [](ov::element::Type& self, ov::element::Type& other) {
            ov::element::Type dst;

            if (ov::element::Type::merge(dst, self, other)) {
                return py::cast(dst);
            }

            return py::none().cast<py::object>();
        },
        py::arg("other"),
        R"(
            Merge two element types and return result if successful,
            otherwise return None.

            :param other: The element type to merge with this element type.
            :type other: openvino.runtime.Type
            :return: If element types are compatible return the least
                     restrictive Type, otherwise `None`.
            :rtype: Union[openvino.runtime.Type|None]
        )");

    type.def(
        "to_dtype",
        [](ov::element::Type& self) {
            return Common::type_helpers::get_dtype(self);
        },
        R"(
            Convert Type to numpy dtype.

            :return: dtype object
            :rtype: numpy.dtype
        )");

    type.def_property_readonly("size", &ov::element::Type::size);
    type.def("get_size", &ov::element::Type::size);
    type.def_property_readonly("bitwidth", &ov::element::Type::bitwidth);
    type.def("get_bitwidth", &ov::element::Type::bitwidth);
}
