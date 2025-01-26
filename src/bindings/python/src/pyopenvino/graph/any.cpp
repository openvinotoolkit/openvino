// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/any.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/any.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

namespace {
bool check_key(py::object key, py::object obj) {
    return key.is(py::type::of(obj));
}
};  // namespace

void regclass_graph_Any(py::module m) {
    py::class_<ov::Any, std::shared_ptr<ov::Any>> ov_any(m, "OVAny");

    ov_any.doc() = "openvino.OVAny provides object wrapper for OpenVINO"
                   "ov::Any class. It allows to pass different types of objects"
                   "into C++ based core of the project.";

    ov_any.def(py::init([](py::object& input_value) {
        return ov::Any(Common::utils::py_object_to_any(input_value));
    }));

    ov_any.def("__repr__", [](const ov::Any& self) {
        return "<" + Common::get_class_name(self) + " class>";
    });

    ov_any.def("__hash__", [](ov::Any& self) {
        return Common::utils::from_ov_any(self).attr("__hash__")();
    });

    ov_any.def("__getitem__", [](const ov::Any& self, py::object& k) {
        return Common::utils::from_ov_any(self).attr("__getitem__")(k);
    });

    ov_any.def("__setitem__", [](const ov::Any& self, py::object& k, const std::string& v) {
        Common::utils::from_ov_any(self).attr("__setitem__")(k, v);
    });

    ov_any.def("__setitem__", [](const ov::Any& self, py::object& k, const int64_t& v) {
        Common::utils::from_ov_any(self).attr("__setitem__")(k, v);
    });

    ov_any.def("__get__", [](const ov::Any& self) {
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
        "astype",
        [](ov::Any& self, py::object dtype) {
            if (check_key(dtype, py::bool_())) {
                return py::cast(self.as<bool>());
            } else if (check_key(dtype, py::str())) {
                return py::cast(self.as<std::string>());
            } else if (check_key(dtype, py::int_())) {
                return py::cast(self.as<int64_t>());
            } else if (check_key(dtype, py::float_())) {
                return py::cast(self.as<double>());
            } else if (check_key(dtype, py::dict())) {
                return Common::utils::from_ov_any_map_no_leaves(self);
            }
            std::stringstream str;
            str << "Unsupported data type : '" << dtype << "' is passed as an argument.";
            OPENVINO_THROW(str.str());
        },
        R"(
            Returns runtime attribute casted to defined data type.

            :param dtype: Data type in which runtime attribute will be casted.
            :type dtype: Union[bool, int, str, float, dict]

            :return: A runtime attribute.
            :rtype: Any
    )");
    ov_any.def(
        "aslist",
        [](ov::Any& self, py::object dtype) {
            // before serialization
            if (self.is<Common::utils::EmptyList>() || dtype.is_none()) {
                return py::cast<py::object>(py::list());
            } else if (self.is<std::vector<double>>()) {
                return py::cast(self.as<std::vector<double>>());
            } else if (self.is<std::vector<std::string>>()) {
                return py::cast(self.as<std::vector<std::string>>());
            } else if (self.is<std::vector<bool>>()) {
                return py::cast(self.as<std::vector<bool>>());
            } else if (self.is<std::vector<int64_t>>()) {
                return py::cast(self.as<std::vector<int64_t>>());
            }
            // after serialization
            if (check_key(dtype, py::str())) {
                return py::cast(self.as<std::vector<std::string>>());
            } else if (check_key(dtype, py::int_())) {
                return py::cast(self.as<std::vector<int64_t>>());
            } else if (check_key(dtype, py::float_())) {
                return py::cast(self.as<std::vector<double>>());
            } else if (check_key(dtype, py::bool_())) {
                return py::cast(self.as<std::vector<bool>>());
            }
            std::stringstream str;
            str << "Unsupported data type : '" << dtype << "' is passed as an argument.";
            OPENVINO_THROW(str.str());
        },
        py::arg("dtype") = py::none(),
        R"(
            Returns runtime attribute as a list with specified data type.

            :param dtype: Data type of a list in which runtime attribute will be casted.
            :type dtype: Union[bool, int, str, float]

            :return: A runtime attribute as a list.
            :rtype: Union[List[float], List[int], List[str], List[bool]]
    )");
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
            return Common::utils::from_ov_any(self);
        },
        R"(
            :return: Value of this OVAny.
            :rtype: Any
    )");
}
