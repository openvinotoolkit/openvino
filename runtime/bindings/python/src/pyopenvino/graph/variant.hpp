// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <string>

#include "openvino/core/variant.hpp" // ov::Variant

namespace py = pybind11;

void regclass_graph_Variant(py::module m);

template <typename VT>
extern void regclass_graph_VariantWrapper(py::module m, std::string typestring)
{
    auto pyclass_name = py::detail::c_str((std::string("Variant") + typestring));
    py::class_<ov::VariantWrapper<VT>,
               std::shared_ptr<ov::VariantWrapper<VT>>,
               ov::Variant>
        variant_wrapper(m, pyclass_name );
    variant_wrapper.doc() =
        "ngraph.impl.Variant[typestring] wraps ov::VariantWrapper<typestring>";

    variant_wrapper.def(py::init<const VT&>());

    variant_wrapper.def(
        "__eq__",
        [](const ov::VariantWrapper<VT>& a, const ov::VariantWrapper<VT>& b) {
            return a.get() == b.get();
        },
        py::is_operator());
    variant_wrapper.def(
        "__eq__",
        [](const ov::VariantWrapper<std::string>& a, const std::string& b) {
            return a.get() == b;
        },
        py::is_operator());
    variant_wrapper.def(
        "__eq__",
        [](const ov::VariantWrapper<int64_t>& a, const int64_t& b) { return a.get() == b; },
        py::is_operator());

    variant_wrapper.def("__repr__", [](const ov::VariantWrapper<VT> self) {
        std::stringstream ret;
        ret << self.get();
        return ret.str();
    });

    variant_wrapper.def("get",
                        (VT & (ov::VariantWrapper<VT>::*)()) & ov::VariantWrapper<VT>::get,
                        R"(
                            Returns
                            ----------
                            get : Variant
                                Value of Variant.
                        )");
    variant_wrapper.def("set",
                        &ov::VariantWrapper<VT>::set,
                        R"(
                            Parameters
                            ----------
                            set : str or int
                                Value to be set in Variant.
                        )");

    variant_wrapper.def_property("value",
                                 (VT & (ov::VariantWrapper<VT>::*)()) &
                                     ov::VariantWrapper<VT>::get,
                                 &ov::VariantWrapper<VT>::set);
}
