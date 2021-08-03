// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
