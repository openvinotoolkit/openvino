// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/types/regmodule_graph_types.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regmodule_graph_types(py::module m) {
    regclass_graph_Type(m);
}
