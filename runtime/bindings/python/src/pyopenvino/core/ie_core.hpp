// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "common.hpp"

#include <pybind11/stl.h>

#include <ie_core.hpp>

#include <openvino/runtime/core.hpp>


namespace py = pybind11;

void regclass_Core(py::module m);
