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
    fem.doc() = "openvino.frontend.paddle.FrontEnd wraps ov::frontend::paddle::FrontEnd";

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

void regclass_frontend_paddle_ConversionExtension(py::module m) {
    py::class_<ConversionExtension, ConversionExtension::Ptr, ov::frontend::ConversionExtensionBase> ext(
        m,
        "ConversionExtensionPaddle",
        py::dynamic_attr());

    ext.def(py::init([](const std::string& op_type, const ov::frontend::CreatorFunctionNamed& f) {
        return std::make_shared<ConversionExtension>(op_type, f);
    }));
}