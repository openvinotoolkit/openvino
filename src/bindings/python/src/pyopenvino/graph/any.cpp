// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/any.hpp"
#include "meta_data.hpp"
#include <pybind11/pybind11.h>

#include "pyopenvino/graph/any.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

void regclass_graph_Any(py::module m) {
    py::class_<ov::Any, std::shared_ptr<ov::Any>> ov_any(m, "OVAny");

    ov_any.doc() = "openvino.runtime.OVAny provides object wrapper for OpenVINO"
                   "ov::Any class. It allows to pass different types of objects"
                   "into C++ based core of the project.";

    ov_any.def(py::init([](py::object& input_value) {
        return ov::Any(Common::utils::py_object_to_any(input_value));
    }));

    ov_any.def("__repr__", [](const ov::Any& self) {
        std::stringstream ret;
        self.print(ret);
        return ret.str();
    });

    ov_any.def("__hash__", [](ov::Any& self) {
        return Common::utils::from_ov_any(self).attr("__hash__")();
    });

    ov_any.def("__getitem__", [](const ov::Any& self, py::object& k) {
        std::cout << "here " << self.type_info().name() << std::endl;
        if (self.is<std::shared_ptr<ov::Meta>>()) {
            const ov::AnyMap& as_map = *self.as<std::shared_ptr<ov::Meta>>();
            return Common::utils::from_ov_any_map(as_map).attr("__getitem__")(k);
        }
        return Common::utils::from_ov_any(self).attr("__getitem__")(k);
    });

    ov_any.def("__setitem__", [](const ov::Any& self, py::object& k, const std::string& v) {
        Common::utils::from_ov_any(self).attr("__setitem__")(k, v);
    });

    ov_any.def("__setitem__", [](const ov::Any& self, py::object& k, const int64_t& v) {
        Common::utils::from_ov_any(self).attr("__setitem__")(k, v);
    });

    ov_any.def("__get__", [](const ov::Any& self) {
        if (self.is<std::shared_ptr<ov::Meta>>()) {
            const ov::AnyMap& as_map = *self.as<std::shared_ptr<ov::Meta>>();
            return Common::utils::from_ov_any_map(as_map);
        }
        return Common::utils::from_ov_any(self);
    });

    ov_any.def("__set__", [](const ov::Any& self, const ov::Any& val) {
        Common::utils::from_ov_any(self) = Common::utils::from_ov_any(val);
    });

    ov_any.def("__len__", [](const ov::Any& self) {
        return Common::utils::from_ov_any(self).attr("__len__")();
    });

    ov_any.def("__eq__", [](const ov::Any& a, const ov::Any& b) -> bool {
        return a == b;
    });
    ov_any.def("__eq__", [](const ov::Any& a, py::object& b) -> bool {
        return a == ov::Any(Common::utils::py_object_to_any(b));
    });
    ov_any.def(
        "get",
        [](const ov::Any& self) -> py::object {
            return Common::utils::from_ov_any(self);
        },
        R"(
            :return: Value of this OVAny.
            :rtype: Any
        )");
    ov_any.def(
        "set",
        [](ov::Any& self, py::object& value) {
            self = ov::Any(Common::utils::py_object_to_any(value));
        },
        R"(
            :param: Value to be set in OVAny.
            :type: Any
    )");
    ov_any.def_property_readonly(
        "value",
        [](const ov::Any& self) {
            if (self.is<std::shared_ptr<ov::Meta>>()) {
                const ov::AnyMap& as_map = *self.as<std::shared_ptr<ov::Meta>>();
                return Common::utils::from_ov_any_map(as_map);
            }
            return Common::utils::from_ov_any(self);
        },
        R"(
            :return: Value of this OVAny.
            :rtype: Any
    )");
}
