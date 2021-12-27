// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "frontend.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/frontend/paddle/extension/conversion.hpp"
#include "openvino/frontend/paddle/frontend.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace py = pybind11;

using namespace ov::frontend::paddle;

#define CHECK_RET(any, T)                   \
    {                                       \
        if ((any).is<T>())                  \
            return py::cast((any).as<T>()); \
    }

void regclass_frontend_paddle_FrontEnd(py::module m) {
    py::class_<FrontEnd, std::shared_ptr<FrontEnd>> fem(m, "FrontEndPaddle", py::dynamic_attr(), py::module_local());
    fem.doc() = "ngraph.impl.FrontEnd wraps ngraph::frontend::paddle::FrontEnd";

    fem.def(py::init([]() {
        return std::make_shared<FrontEnd>();
    }));

    fem.def("convert",
            static_cast<std::shared_ptr<ov::Model> (FrontEnd::*)(const ov::frontend::InputModel::Ptr&) const>(
                &FrontEnd::convert),
            py::arg("model"));

    fem.def("convert",
            static_cast<void (FrontEnd::*)(const std::shared_ptr<ov::Model>&) const>(&FrontEnd::convert),
            py::arg("function"));

    fem.def("decode", &FrontEnd::decode, py::arg("model"));

    fem.def("get_name", &FrontEnd::get_name);

    fem.def("add_extension",
            static_cast<void (FrontEnd::*)(const std::shared_ptr<ov::Extension>& extension)>(&FrontEnd::add_extension));

    fem.def("__repr__", [](const FrontEnd& self) -> std::string {
        return "<FrontEnd '" + self.get_name() + "'>";
    });

    fem.def("convert_partially", &FrontEnd::convert_partially, py::arg("model"));

    fem.def("decode", &FrontEnd::decode, py::arg("model"));
}

void regclass_frontend_paddle_NodeContext(py::module m) {
    py::class_<NodeContext, NodeContext::Ptr, ov::frontend::NodeContext> ext(m,
                                                                             "NodeContextPaddle",
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

    ext.def("get_input", [](NodeContext& self, const std::string& name) {
        return self.get_input(name);
    });

    ext.def("get_input", [](NodeContext& self, const std::string& name, int idx) {
        return self.get_input(name, idx);
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

void regclass_frontend_paddle_ConversionExtension(py::module m) {
    py::class_<ConversionExtension, ConversionExtension::Ptr, ov::frontend::ConversionExtensionBase> ext(
        m,
        "ConversionExtensionPaddle",
        py::dynamic_attr());

    ext.def(py::init([](const std::string& op_type, const CreatorFunction& f) {
        return std::make_shared<ConversionExtension>(op_type, f);
    }));
}