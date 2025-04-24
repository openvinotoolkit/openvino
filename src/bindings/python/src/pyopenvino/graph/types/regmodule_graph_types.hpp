// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "pyopenvino/graph/types/element_type.hpp"

namespace py = pybind11;

void regmodule_graph_types(py::module m);
