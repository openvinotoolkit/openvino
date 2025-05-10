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

    PyFrontendOpExtension(const py::object& dtype, const std::string& op_type) : ConversionExtensionBase(op_type), py_handle_dtype{dtype} {
        py::object py_issubclass = py::module::import("builtins").attr("issubclass");
        if (!py_issubclass(dtype, py::type::of<PyOp>()).cast<bool>()) {
            std::stringstream str;
            str << "Unsupported data type : '" << dtype.attr("__name__").cast<std::string>() << "' is passed as an argument.";
            OPENVINO_THROW(str.str());
        }
    }

    std::variant<std::shared_ptr<ov::frontend::OpExtension<void>>, 
                 std::shared_ptr<ov::frontend::OpExtension<PyOp>>> impl;

private:
    py::object py_handle_dtype;  // Holds the Python object to manage its lifetime
};

void regclass_frontend_TelemetryExtension(py::module m);
void regclass_frontend_DecoderTransformationExtension(py::module m);
void regclass_frontend_ConversionExtension(py::module m);
void regclass_frontend_ConversionExtensionBase(py::module m);
void regclass_frontend_ProgressReporterExtension(py::module m);
void regclass_frontend_OpExtension(py::module m);

