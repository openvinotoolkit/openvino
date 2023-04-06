// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>
#include <string>

#include "openvino/core/any.hpp"

namespace py = pybind11;

void regclass_pyngraph_Variant(py::module m);
