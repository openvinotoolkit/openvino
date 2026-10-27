// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "openvino/runtime/tensor.hpp"

// Declared once per translation unit; repeating the specialization breaks unity builds.
PYBIND11_MAKE_OPAQUE(ov::TensorVector);
