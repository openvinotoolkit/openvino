// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include <variant>
#include "openvino/frontend/extension/conversion.hpp"
#include "openvino/frontend/extension/op.hpp"
#include "pyopenvino/graph/op.hpp"

namespace py = pybind11;

class PyFrontendOpExtension : public ov::frontend::ConversionExtensionBase {
public:
    explicit PyFrontendOpExtension(const std::string& op_type) 
    : ov::frontend::ConversionExtensionBase(op_type) {}
    std::variant<std::shared_ptr<ov::frontend::OpExtension<void>>, 
                 std::shared_ptr<ov::frontend::OpExtension<PyOp>>> impl;

};

void regclass_frontend_TelemetryExtension(py::module m);
void regclass_frontend_DecoderTransformationExtension(py::module m);
void regclass_frontend_ConversionExtension(py::module m);
void regclass_frontend_ConversionExtensionBase(py::module m);
void regclass_frontend_ProgressReporterExtension(py::module m);
void regclass_frontend_OpExtension(py::module m);

