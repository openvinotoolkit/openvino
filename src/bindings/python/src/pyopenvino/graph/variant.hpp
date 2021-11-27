// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <string>

#include "openvino/core/any.hpp" // ov::Variant

namespace py = pybind11;

void regclass_graph_Variant(py::module m);

template<typename T>
struct AnyT : public ov::Any {
    using ov::Any::Any;
};

template <typename VT>
extern void regclass_graph_VariantWrapper(py::module m, std::string typestring)
{
    auto pyclass_name = py::detail::c_str((std::string("Variant") + typestring));
    py::class_<AnyT<VT>, ov::Any>
        variant_wrapper(m, pyclass_name, py::module_local());
    variant_wrapper.doc() =
        "openvino.impl.Variant[" + typestring + "] wraps ov::Any with " + typestring;

    variant_wrapper.def(py::init<const VT&>());

    variant_wrapper.def(
        "__eq__",
        [](const ov::Any& a, const ov::Any& b) {
            return a.as<VT>() == b.as<VT>();
        },
        py::is_operator());
    variant_wrapper.def(
        "__eq__",
        [](const ov::Any& a, const std::string& b) {
            return a.as<std::string>() == b;
        },
        py::is_operator());
    variant_wrapper.def(
        "__eq__",
        [](const ov::Any& a, const int64_t& b) { return a.as<int64_t>() == b; },
        py::is_operator());

    variant_wrapper.def("__repr__", [](const ov::Any self) {
        std::stringstream ret;
        self.print(ret);
        return ret.str();
    });

    variant_wrapper.def("get",
                        [] (const ov::Any& self) {
                            return self.as<VT>();
                        },
                        R"(
                            Returns
                            ----------
                            get : Variant
                                Value of ov::Any.
                        )");
    variant_wrapper.def("set",
                        [] (ov::Any& self, const VT value) {
                            self = value;
                        },
                        R"(
                            Parameters
                            ----------
                            set : str or int
                                Value to be set in ov::Any.
                        )");

    variant_wrapper.def_property("value",
                                 [] (const ov::Any& self) {
                                    return self.as<VT>();
                                 },
                                 [] (ov::Any& self, const VT value) {
                                    self = value;
                                 });
}
