// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include "pyopenvino/graph/passes/manager.hpp"

namespace py = pybind11;

void regmodule_graph_passes(py::module m);
