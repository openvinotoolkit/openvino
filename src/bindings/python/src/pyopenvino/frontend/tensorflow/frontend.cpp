// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "frontend.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/tensorflow/extension/conversion.hpp"
#include "openvino/frontend/tensorflow/frontend.hpp"
#include "openvino/frontend/tensorflow/node_context.hpp"

namespace py = pybind11;

using namespace ov::frontend::tensorflow;

void regclass_frontend_tensorflow_FrontEnd(py::module m) {
    py::class_<FrontEnd, std::shared_ptr<FrontEnd>> fem(m, "FrontEnd", py::dynamic_attr(), py::module_local());
    fem.doc() = "ngraph.impl.FrontEnd wraps ngraph::frontend::tensorflow::FrontEnd";

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

    fem.def("convert_partially", &FrontEnd::convert_partially, py::arg("model"));

    fem.def("decode", &FrontEnd::decode, py::arg("model"));

    fem.def("__repr__", [](const FrontEnd& self) -> std::string {
        return "<FrontEnd '" + self.get_name() + "'>";
    });
}

void regclass_frontend_tensorflow_NodeContext(py::module m) {
    py::class_<NodeContext, NodeContext::Ptr, ov::frontend::NodeContext> ext(m, "NodeContext", py::dynamic_attr());
}

void regclass_frontend_tensorflow_ConversionExtension(py::module m) {
    py::class_<ConversionExtension, ConversionExtension::Ptr, ov::frontend::ConversionExtensionBase> ext(
        m,
        "ConversionExtension",
        py::dynamic_attr());

    ext.def(py::init([](const std::string& op_type, const CreatorFunction& f) {
        return std::make_shared<ConversionExtension>(op_type, f);
    }));
}
