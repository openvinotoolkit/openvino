// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <string>

#include "openvino/core/any.hpp"

namespace py = pybind11;

void regclass_pyngraph_Variant(py::module m);

template<typename T>
struct AnyT : public ov::Any {
    using ov::Any::Any;
};


template <typename VT>
extern void regclass_pyngraph_VariantWrapper(py::module m, std::string typestring) {
    auto pyclass_name = py::detail::c_str((std::string("Variant") + typestring));
    py::class_<AnyT<VT>, ov::Any>
        variant_wrapper(m, pyclass_name, py::module_local());
    variant_wrapper.doc() =
        "openvino.runtime.Variant[" + typestring + "] wraps ov::Any with " + typestring;

    variant_wrapper.def(py::init<const VT&>());

    variant_wrapper.def(
        "__eq__",
        [](const AnyT<VT>& a, const AnyT<VT>& b) {
            return a.template as<VT>() == b.template as<VT>();
        },
        py::is_operator());
    variant_wrapper.def(
        "__eq__",
        [](const AnyT<VT>& a, const std::string& b) {
            return a.template as<std::string>() == b;
        },
        py::is_operator());
    variant_wrapper.def(
        "__eq__",
        [](const AnyT<VT>& a, const int64_t& b) { return a.template as<int64_t>() == b; },
        py::is_operator());

    variant_wrapper.def("__repr__", [](const AnyT<VT> self) {
        std::stringstream ret;
        self.print(ret);
        return ret.str();
    });

    variant_wrapper.def("get",
                        [] (const AnyT<VT>& self) {
                            return self.template as<VT>();
                        },
                        R"(
                            Returns
                            ----------
                            get : Variant
                                Value of ov::Any.
                        )");
    variant_wrapper.def("set",
                        [] (AnyT<VT>& self, const VT value) {
                            self = value;
                        },
                        R"(
                            Parameters
                            ----------
                            set : str or int
                                Value to be set in ov::Any.
                        )");

    variant_wrapper.def_property("value",
                                 [] (const AnyT<VT>& self) {
                                    return self.template as<VT>();
                                 },
                                 [] (AnyT<VT>& self, const VT value) {
                                    self = value;
                                 });
}
