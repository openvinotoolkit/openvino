// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regclass_graph_pattern_Matcher(py::module m);

void regclass_graph_pattern_PassBase(py::module m);

void regclass_graph_pattern_MatcherPass(py::module m);

void regclass_graph_patterns(py::module m);

void regclass_transformations(py::module m);