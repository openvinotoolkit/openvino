// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/frontend/node_context.hpp"

namespace py = pybind11;

using namespace ov::frontend;

template <typename>
struct is_std_vector : std::false_type {};

template <typename T, typename A>
struct is_std_vector<std::vector<T, A>> : std::true_type {};

#define CAST_VEC_TO_PY(any, py_type, c_type)                                       \
    {                                                                              \
        static_assert(is_std_vector<c_type>(), "The type should be std::vector."); \
        if ((any).is<c_type>()) {                                                  \
            auto casted = (any).as<c_type>();                                      \
            if (!(py_type).is_none()) {                                            \
                py::list py_list;                                                  \
                for (auto el : casted) {                                           \
                    py_list.append(py_type(el));                                   \
                }                                                                  \
                return std::move(py_list);                                         \
            }                                                                      \
            return py::cast(casted);                                               \
        }                                                                          \
    }

#define CAST_TO_PY(any, py_type, c_type)      \
    {                                         \
        if ((any).is<c_type>()) {             \
            auto casted = (any).as<c_type>(); \
            if (!(py_type).is_none()) {       \
                return py_type(casted);       \
            }                                 \
            return py::cast(casted);          \
        }                                     \
    }

void regclass_frontend_NodeContext(py::module m) {
    py::class_<ov::frontend::NodeContext, std::shared_ptr<ov::frontend::NodeContext>> ext(m,
                                                                                          "NodeContext",
                                                                                          py::dynamic_attr());

    auto cast_attribute = [](const ov::Any& any, const py::object& dtype) -> py::object {
        CAST_TO_PY(any, dtype, int32_t);
        CAST_TO_PY(any, dtype, int64_t);
        CAST_TO_PY(any, dtype, bool);
        CAST_TO_PY(any, dtype, std::string);
        CAST_TO_PY(any, dtype, float);
        CAST_TO_PY(any, dtype, double);
        CAST_TO_PY(any, dtype, ov::element::Type);
        CAST_TO_PY(any, dtype, ov::PartialShape);

        CAST_VEC_TO_PY(any, dtype, std::vector<int32_t>);
        CAST_VEC_TO_PY(any, dtype, std::vector<int64_t>);
#ifndef __APPLE__
        // TODO: investigate the issue in pybind11 on MacOS
        CAST_VEC_TO_PY(any, dtype, std::vector<bool>);
#endif
        CAST_VEC_TO_PY(any, dtype, std::vector<std::string>);
        CAST_VEC_TO_PY(any, dtype, std::vector<float>);
        CAST_VEC_TO_PY(any, dtype, std::vector<double>);
        CAST_VEC_TO_PY(any, dtype, std::vector<ov::element::Type>);
        CAST_VEC_TO_PY(any, dtype, std::vector<ov::PartialShape>);

        return py::none();
    };

    ext.def(
        "get_attribute",
        [=](NodeContext& self, const std::string& name, const py::object& default_value, const py::object& dtype)
            -> py::object {
            auto any = self.get_attribute_as_any(name);

            auto type = m.attr("Type");
            if (dtype.is(type)) {
                if (any.is<int32_t>() || any.is<int64_t>()) {
                    return py::cast(self.get_attribute<ov::element::Type>(name));
                } else if (any.is<std::vector<int32_t>>() || any.is<std::vector<int64_t>>()) {
                    return py::cast(self.get_attribute<std::vector<ov::element::Type>>(name));
                }
            }

            auto casted = cast_attribute(any, dtype);
            if (!casted.is_none())
                return casted;

            if (default_value.is_none())
                FRONT_END_GENERAL_CHECK(false, "Attribute ", name, " can't be converted to defined types.");
            else
                return default_value;
        },
        py::arg("name"),
        py::arg("default_value") = py::none(),
        py::arg("dtype") = py::none());

    ext.def("get_input", [](NodeContext& self, int idx) {
        return self.get_input(idx);
    });

    ext.def("get_input", [](NodeContext& self, const std::string& name) {
        return self.get_input(name);
    });

    ext.def("get_input", [](NodeContext& self, const std::string& name, int idx) {
        return self.get_input(name, idx);
    });

    ext.def(
        "get_values_from_const_input",
        [=](NodeContext& self, int idx, const py::object& default_value, const py::object& dtype) -> py::object {
            auto any = self.get_values_from_const_input(idx);
            if (any.empty())
                return py::none();

            auto casted = cast_attribute(any, dtype);
            if (!casted.is_none())
                return casted;

            if (default_value.is_none())
                FRONT_END_GENERAL_CHECK(false, "Const input with index ", idx, " can't be converted to defined types.");
            else
                return default_value;
        },
        py::arg("idx"),
        py::arg("default_value") = py::none(),
        py::arg("dtype") = py::none());

    ext.def("get_input_size", [](NodeContext& self) {
        return self.get_input_size();
    });

    ext.def("get_input_size", [](NodeContext& self, std::string& name) {
        return self.get_input_size(name);
    });

    ext.def("get_op_type", [](NodeContext& self, std::string& name) {
        return self.get_op_type();
    });

    ext.def("has_attribute", [](NodeContext& self, std::string& name) {
        return self.has_attribute(name);
    });
}
