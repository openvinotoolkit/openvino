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

using PyRTMap = ov::RTMap;

PYBIND11_MAKE_OPAQUE(PyRTMap);

void regclass_graph_PyRTMap(py::module m) {
    auto py_map = py::bind_map<PyRTMap>(m, "PyRTMap");
    py_map.doc() = "openvino.impl.PyRTMap makes bindings for std::map<std::string, "
                   "ov::Any, which can later be used as ov::Node::RTMap";

    py_map.def("__setitem__", [](PyRTMap& m, const std::string& k, const std::string v) {
        m[k] = v;
    });
    py_map.def("__setitem__", [](PyRTMap& m, const std::string& k, const int64_t v) {
        m[k] = v;
    });
}
