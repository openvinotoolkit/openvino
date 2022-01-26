// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/passes/regmodule_graph_passes.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regmodule_graph_passes(py::module m) {
    py::module m_passes = m.def_submodule("passes", "Package openvino.runtime.passes wraps ov::passes");
    regclass_graph_passes_Manager(m_passes);
}
