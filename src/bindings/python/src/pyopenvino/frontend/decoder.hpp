// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "openvino/frontend/decoder.hpp"

namespace py = pybind11;

class PyIDecoder : public ov::frontend::IDecoder {
public:
    using IDecoder::IDecoder; // Inherit constructors
};

void regclass_frontend_IDecoder(py::module m);
