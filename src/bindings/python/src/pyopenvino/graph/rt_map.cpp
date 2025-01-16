// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/rt_map.hpp"

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "dict_attribute_visitor.hpp"
#include "openvino/core/meta_data.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/runtime_attribute.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/any.hpp"
#include "pyopenvino/graph/node.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

using PyRTMap = ov::RTMap;

PYBIND11_MAKE_OPAQUE(PyRTMap);

// Create our custom iterator to return python object not OVAny itself.
class PyRTMapIterator {
public:
    PyRTMapIterator(const PyRTMap& py_rt_map, py::object ref, bool is_value)
        : py_rt_map(py_rt_map),
          is_value(is_value),
          ref(ref),
          it(py_rt_map.cbegin()) {}

    py::object next() {
        if (it == py_rt_map.end()) {
            throw py::stop_iteration();
        }
        const auto result = *it;
        it++;
        if (is_value) {
            return Common::utils::from_ov_any_no_leaves(result.second);
        } else {
            std::pair<std::string, py::object> res = {result.first,
                                                      Common::utils::from_ov_any_no_leaves(result.second)};
            return py::cast(res);
        }
    }

    const PyRTMap& py_rt_map;
    bool is_value = false;
    py::object ref;  // keep a reference
    std::map<std::string, ov::Any>::const_iterator it;
};

void regclass_graph_PyRTMap(py::module m) {
    auto py_map = py::class_<PyRTMap>(m, "RTMap");
    py_map.doc() = "openvino.RTMap makes bindings for std::map<std::string, "
                   "ov::Any>, which can later be used as ov::Node::RTMap";

    py::class_<PyRTMapIterator>(m, "Iterator")
        .def("__iter__",
             [](PyRTMapIterator& it) -> PyRTMapIterator& {
                 return it;
             })
        .def("__next__", &PyRTMapIterator::next);

    py_map.def("__setitem__", [](PyRTMap& m, const std::string& k, const std::string v) {
        m[k] = v;
    });
    py_map.def("__setitem__", [](PyRTMap& m, const std::string& k, const int64_t v) {
        m[k] = v;
    });
    py_map.def("__getitem__", [](PyRTMap& m, const std::string& k) -> py::object {
        return Common::utils::from_ov_any_no_leaves(m[k]);
    });
    py_map.def(
        "__bool__",
        [](const PyRTMap& m) -> bool {
            return !m.empty();
        },
        "Check whether the map is nonempty");

    py_map.def(
        "__iter__",
        [](PyRTMap& rt_map) {
            return py::make_key_iterator(rt_map.begin(), rt_map.end());
        },
        py::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
    );

    py_map.def(
        "keys",
        [](PyRTMap& rt_map) {
            return py::make_key_iterator(rt_map.begin(), rt_map.end());
        },
        py::keep_alive<0, 1>() /* Essential: keep list alive while iterator exists */
    );

    py_map.def("items", [](py::object rt_map) {
        return PyRTMapIterator(rt_map.cast<const PyRTMap&>(), rt_map, false);
    });

    py_map.def("values", [](py::object rt_map) {
        return PyRTMapIterator(rt_map.cast<const PyRTMap&>(), rt_map, true);
    });

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

    py_map.def("__repr__", [](const PyRTMap& self) {
        return Common::get_simple_repr(self);
    });
}
