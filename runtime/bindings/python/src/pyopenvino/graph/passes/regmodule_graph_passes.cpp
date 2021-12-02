// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/passes/regmodule_graph_passes.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regmodule_graph_passes(py::module m) {
    py::module m_passes = m.def_submodule("passes", "Package openvino.impl.passes wraps ov::passes");
    regclass_graph_passes_Manager(m_passes);
    regclass_graph_pattern_Matcher(m_passes);
    regclass_graph_pattern_PassBase(m_passes);
    regclass_graph_pattern_MatcherPass(m_passes);
    regclass_graph_patterns(m_passes);
    regclass_transformations(m_passes);
}
