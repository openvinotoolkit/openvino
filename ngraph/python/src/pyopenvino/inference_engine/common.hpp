// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include "Python.h"
#include <ie_parameter.hpp>
#include "ie_common.h"

namespace Common {
    InferenceEngine::Layout get_layout_from_string(const std::string &layout);

    const std::string& get_layout_from_enum(const InferenceEngine::Layout &layout);

    PyObject *parse_parameter(const InferenceEngine::Parameter &param);
};
