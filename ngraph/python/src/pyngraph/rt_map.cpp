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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "dict_attribute_visitor.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/divide.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/variant.hpp"
#include "pyngraph/node.hpp"
#include "pyngraph/rt_map.hpp"
#include "pyngraph/variant.hpp"

namespace py = pybind11;

using PyRTMap = std::map<std::string, std::shared_ptr<ngraph::Variant>>;

PYBIND11_MAKE_OPAQUE(PyRTMap);

template <typename T>
void _set_with_variant(PyRTMap& m, const std::string& k, const T v)
{
    auto new_v = std::make_shared<ngraph::VariantWrapper<T>>(ngraph::VariantWrapper<T>(v));
    auto it = m.find(k);
    if (it != m.end())
        it->second = new_v;
    else
        m.emplace(k, new_v);
}

void regclass_pyngraph_PyRTMap(py::module m)
{
    auto py_map = py::bind_map<PyRTMap>(m, "PyRTMap");
    py_map.doc() =
        "ngraph.impl.PyRTMap makes bindings for std::map<std::string, "
        "std::shared_ptr<ngraph::Variant>>, which can later be used as ngraph::Node::RTMap";

    py_map.def("__setitem__", [](PyRTMap& m, const std::string& k, const std::string v) {
        _set_with_variant(m, k, v);
    });
    py_map.def("__setitem__", [](PyRTMap& m, const std::string& k, const int64_t v) {
        _set_with_variant(m, k, v);
    });
}
