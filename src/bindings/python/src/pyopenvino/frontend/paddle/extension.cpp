// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extension.hpp"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "openvino/frontend/paddle/extension/conversion.hpp"

namespace py = pybind11;

using namespace ov::frontend::paddle;

void regclass_frontend_paddle_ConversionExtension(py::module m) {
    py::class_<ConversionExtension, ConversionExtension::Ptr, ov::frontend::ConversionExtensionBase> _ext(
        m,
        "_ConversionExtensionPaddle",
        py::dynamic_attr());
    class PyConversionExtension : public ConversionExtension {
    public:
        using Ptr = std::shared_ptr<PyConversionExtension>;
        using PyCreatorFunctionNamed =
            std::function<std::map<std::string, ov::OutputVector>(const ov::frontend::NodeContext*)>;

        PyConversionExtension(const std::string& op_type, const PyCreatorFunctionNamed& f)
            : ConversionExtension(
                  op_type,
                  [f](const ov::frontend::NodeContext& node) -> std::map<std::string, ov::OutputVector> {
                      return f(static_cast<const ov::frontend::NodeContext*>(&node));
                  }) {}
    };
    py::class_<PyConversionExtension, PyConversionExtension::Ptr, ConversionExtension> ext(m,
                                                                                           "ConversionExtensionPaddle",
                                                                                           py::dynamic_attr());

    ext.def(py::init([](const std::string& op_type, const PyConversionExtension::PyCreatorFunctionNamed& f) {
        return std::make_shared<PyConversionExtension>(op_type, f);
    }));
}
