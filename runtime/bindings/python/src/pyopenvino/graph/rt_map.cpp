// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/rt_map.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "dict_attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/variant.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "pyopenvino/graph/node.hpp"
#include "pyopenvino/graph/variant.hpp"

namespace py = pybind11;

using PyRTMap = std::map<std::string, std::shared_ptr<ov::Variant>>;

PYBIND11_MAKE_OPAQUE(PyRTMap);

template <typename T>
void _set_with_variant(PyRTMap& m, const std::string& k, const T v) {
    auto new_v = std::make_shared<ov::VariantWrapper<T>>(ov::VariantWrapper<T>(v));
    auto it = m.find(k);
    if (it != m.end())
        it->second = new_v;
    else
        m.emplace(k, new_v);
}

void regclass_graph_PyRTMap(py::module m) {
    auto py_map = py::bind_map<PyRTMap>(m, "PyRTMap");
    py_map.doc() = "ngraph.impl.PyRTMap makes bindings for std::map<std::string, "
                   "std::shared_ptr<ov::Variant>>, which can later be used as ov::Node::RTMap";

    py_map.def("__setitem__", [](PyRTMap& m, const std::string& k, const std::string v) {
        _set_with_variant(m, k, v);
    });
    py_map.def("__setitem__", [](PyRTMap& m, const std::string& k, const int64_t v) {
        _set_with_variant(m, k, v);
    });
}
