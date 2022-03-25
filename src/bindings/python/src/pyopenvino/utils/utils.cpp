// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "pyopenvino/utils/utils.hpp"

ov::Any py_object_to_any(const pybind11::object& py_obj) {
    if (pybind11::isinstance<pybind11::str>(py_obj)) {
        return py_obj.cast<std::string>();
    } else if (pybind11::isinstance<pybind11::bool_>(py_obj)) {
        return py_obj.cast<bool>();
    } else if (pybind11::isinstance<pybind11::float_>(py_obj)) {
        return py_obj.cast<double>();
    } else if (pybind11::isinstance<pybind11::int_>(py_obj)) {
        return py_obj.cast<int64_t>();
    } else if (pybind11::isinstance<pybind11::list>(py_obj)) {
        auto _list = py_obj.cast<pybind11::list>();
        enum class PY_TYPE : int { UNKNOWN = 0, STR, INT, FLOAT, BOOL };
        PY_TYPE detected_type = PY_TYPE::UNKNOWN;
        for (const auto& it : _list) {
            auto check_type = [&](PY_TYPE type) {
                if (detected_type == PY_TYPE::UNKNOWN || detected_type == type) {
                    detected_type = type;
                    return;
                }
                OPENVINO_ASSERT("Incorrect attribute. Mixed types in the list are not allowed.");
            };
            if (pybind11::isinstance<pybind11::str>(it)) {
                check_type(PY_TYPE::STR);
            } else if (pybind11::isinstance<pybind11::int_>(it)) {
                check_type(PY_TYPE::INT);
            } else if (pybind11::isinstance<pybind11::float_>(it)) {
                check_type(PY_TYPE::FLOAT);
            } else if (pybind11::isinstance<pybind11::bool_>(it)) {
                check_type(PY_TYPE::BOOL);
            }
        }

        switch (detected_type) {
        case PY_TYPE::STR:
            return _list.cast<std::vector<std::string>>();
        case PY_TYPE::FLOAT:
            return _list.cast<std::vector<double>>();
        case PY_TYPE::INT:
            return _list.cast<std::vector<int64_t>>();
        case PY_TYPE::BOOL:
            return _list.cast<std::vector<bool>>();
        default:
            OPENVINO_ASSERT(false, "Unsupported attribute type.");
        }
    }
    OPENVINO_ASSERT(false, "Unsupported attribute type.");
}
