//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "ngraph/ngraph_visibility.hpp"
#include "ngraph/node.hpp"
#include "ngraph/type.hpp"

#include "ngraph/variant.hpp" // ngraph::Variant

namespace py = pybind11;

// template <typename VT>
// extern void regclass_pyngraph_Variant(py::module m, std::string typestring);
template <typename VT>
extern void regclass_pyngraph_Variant(py::module m, std::string typestring)
{
    // a = Variant<string>("aaa");
    // b = Variant<int>(111);

    // VariantString
    // VariantInt

    auto pyclass_name = py::detail::c_str((std::string("Variant") + typestring));
    py::class_<ngraph::VariantWrapper<VT>, std::shared_ptr<ngraph::VariantWrapper<VT>>> variant(
        m, pyclass_name);
    variant.doc() = "ngraph.impl.Variant wraps ngraph::VariantWrapper";

    // variant.def(py::init<>());
    variant.def(py::init<const VT&>());
    variant.def("init", &ngraph::VariantWrapper<VT>::init);
    variant.def("merge", &ngraph::VariantWrapper<VT>::merge);
    variant.def("get_type_info", &ngraph::VariantWrapper<VT>::get_type_info);
    variant.def("get",
                (const VT& (ngraph::VariantWrapper<VT>::*)() const) &
                    ngraph::VariantWrapper<VT>::get);
    variant.def("get", (VT & (ngraph::VariantWrapper<VT>::*)()) & ngraph::VariantWrapper<VT>::get);
    variant.def("set", &ngraph::VariantWrapper<VT>::set);

    variant.def_property("m_value",
                         (const VT& (ngraph::VariantWrapper<VT>::*)() const) &
                             ngraph::VariantWrapper<VT>::get,
                         &ngraph::VariantWrapper<VT>::set);
    variant.def_property_readonly("type_info", &ngraph::VariantWrapper<VT>::get_type_info);
}

template void regclass_pyngraph_Variant<std::string>(py::module m, std::string typestring);
template void regclass_pyngraph_Variant<int64_t>(py::module m, std::string typestring);
