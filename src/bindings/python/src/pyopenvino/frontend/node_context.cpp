// Copyright (C) 2018-2021 Intel Corporation
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

#define CHECK_RET(any, T)                   \
    {                                       \
        if ((any).is<T>())                  \
            return py::cast((any).as<T>()); \
    }

void regclass_frontend_NodeContext(py::module m) {
    py::class_<ov::frontend::NodeContext, std::shared_ptr<ov::frontend::NodeContext>> ext(m,
                                                                                          "NodeContext",
                                                                                          py::dynamic_attr());

    ext.def("get_attribute", [](NodeContext& self, const std::string& name) -> py::object {
        auto any = self.get_attribute_as_any(name);

        CHECK_RET(any, int32_t);
        CHECK_RET(any, int64_t);
        CHECK_RET(any, bool);
        CHECK_RET(any, std::string);
        CHECK_RET(any, float);
        CHECK_RET(any, ov::element::Type);
        CHECK_RET(any, ov::PartialShape);

        CHECK_RET(any, std::vector<int32_t>);
        CHECK_RET(any, std::vector<int64_t>);
        CHECK_RET(any, std::vector<bool>);
        CHECK_RET(any, std::vector<std::string>);
        CHECK_RET(any, std::vector<float>);
        CHECK_RET(any, std::vector<ov::element::Type>);
        CHECK_RET(any, std::vector<ov::PartialShape>);

        FRONT_END_GENERAL_CHECK(false, "Attribute type can't be converted.");
    });

    ext.def("get_attribute", [](NodeContext& self, const std::string& name, const py::object& def) -> py::object {
        auto any = self.get_attribute_as_any(name);

        CHECK_RET(any, int32_t);
        CHECK_RET(any, int64_t);
        CHECK_RET(any, bool);
        CHECK_RET(any, std::string);
        CHECK_RET(any, float);
        CHECK_RET(any, ov::element::Type);
        CHECK_RET(any, ov::PartialShape);

        CHECK_RET(any, std::vector<int32_t>);
        CHECK_RET(any, std::vector<int64_t>);
        CHECK_RET(any, std::vector<bool>);
        CHECK_RET(any, std::vector<std::string>);
        CHECK_RET(any, std::vector<float>);
        CHECK_RET(any, std::vector<ov::element::Type>);
        CHECK_RET(any, std::vector<ov::PartialShape>);

        return def;
    });

    ext.def("get_input", [](NodeContext& self, int idx) {
        return self.get_input(idx);
    });

    ext.def("get_input", [](NodeContext& self, const std::string& name) {
        return self.get_input(name);
    });

    ext.def("get_input", [](NodeContext& self, const std::string& name, int idx) {
        return self.get_input(name, idx);
    });

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