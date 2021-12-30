// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "extension.hpp"

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

void regclass_frontend_tensorflow_ConversionExtension(py::module m) {
    py::class_<ConversionExtension, ConversionExtension::Ptr, ov::frontend::ConversionExtensionBase> ext(
        m,
        "ConversionExtensionTensorflow",
        py::dynamic_attr());

    ext.def(py::init([](const std::string& op_type, const ov::frontend::ConversionExtension::PyCreatorFunction& f) {
        return std::make_shared<ConversionExtension>(op_type, f);
    }));
}
