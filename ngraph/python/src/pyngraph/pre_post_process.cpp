// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/pre_post_process/pre_post_process.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iterator>
#include <sstream>
#include <string>

#include "pyngraph/pre_post_process.hpp"

namespace py = pybind11;

void regclass_pyngraph_PrePostProcessor(py::module m) {}
