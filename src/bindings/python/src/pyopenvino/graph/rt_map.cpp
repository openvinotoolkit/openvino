// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/rt_map.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "dict_attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "meta_data.hpp"
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/any.hpp"
#include "pyopenvino/graph/node.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

using PyRTMap = ov::RTMap;

PYBIND11_MAKE_OPAQUE(PyRTMap);

void regclass_graph_PyRTMap(py::module m) {
    auto py_map = py::class_<PyRTMap>(m, "RTMap");
    py_map.doc() = "openvino.runtime.RTMap makes bindings for std::map<std::string, "
                   "ov::Any>, which can later be used as ov::Node::RTMap";

    py_map.def("__setitem__", [](PyRTMap& m, const std::string& k, const std::string v) {
        m[k] = v;
    });
    py_map.def("__setitem__", [](PyRTMap& m, const std::string& k, const int64_t v) {
        m[k] = v;
    });
    py_map.def("__getitem__", [](PyRTMap& m, const std::string& k) -> py::object {
        std::cout << "olvdo";
        if (m[k].is<std::shared_ptr<ov::Meta>>()) {
            const ov::AnyMap& as_map = *m[k].as<std::shared_ptr<ov::Meta>>();
            return Common::utils::from_ov_any_map(as_map);
        }
        return Common::utils::from_ov_any(m[k]);
    });
    py_map.def(
        "__bool__",
        [](const PyRTMap& m) -> bool {
            return !m.empty();
        },
        "Check whether the map is nonempty");

    py_map.def(
        "__iter__",
        [](PyRTMap& m) {
            return py::make_key_iterator(m.begin(), m.end());
        },
        py::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
    );

    py_map.def(
        "items",
        [](PyRTMap& m) {
            // ((x, self[x]) for x in self)
            return py::make_iterator(m.begin(), m.end());
            //return Common::utils::from_ov_any(m.);
        },
       py::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
    );

    py_map.def(
            "values",
            [](const PyRTMap &map) { return py::make_value_iterator(map.begin(), map.end()); },
            py::keep_alive<0, 1>());

    py_map.def("__contains__", [](PyRTMap& m, const std::string& k) -> bool {
        auto it = m.find(k);
        if (it == m.end())
            return false;
        return true;
    });
    py_map.def("__delitem__", [](PyRTMap& m, const std::string& k) {
        auto it = m.find(k);
        if (it == m.end())
            throw py::key_error();
        m.erase(it);
    });

    py_map.def("__len__", &PyRTMap::size);
}
