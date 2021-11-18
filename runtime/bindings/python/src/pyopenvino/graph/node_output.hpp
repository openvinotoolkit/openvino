// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/core/node_output.hpp"

namespace py = pybind11;

void regclass_graph_Output(py::module m, std::string typestring);
void regclass_graph_ConstOutput(py::module m, std::string typestring);
