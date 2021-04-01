// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include <string>
#include "Python.h"
#include <ie_parameter.hpp>
#include "ie_common.h"
#include <ie_blob.h>

namespace py = pybind11;

namespace Common {
    InferenceEngine::Layout get_layout_from_string(const std::string &layout);

    const std::string& get_layout_from_enum(const InferenceEngine::Layout &layout);

    PyObject *parse_parameter(const InferenceEngine::Parameter &param);

    const std::shared_ptr<InferenceEngine::Blob> cast_to_blob(const py::handle& blob);
}; // namespace Common
