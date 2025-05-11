// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/passes/regmodule_graph_passes.hpp"

#include <pybind11/pybind11.h>

#include "pyopenvino/graph/passes/graph_rewrite.hpp"
#include "pyopenvino/graph/passes/manager.hpp"
#include "pyopenvino/graph/passes/matcher_pass.hpp"
#include "pyopenvino/graph/passes/model_pass.hpp"
#include "pyopenvino/graph/passes/pass_base.hpp"
#include "pyopenvino/graph/passes/pattern_ops.hpp"
#include "pyopenvino/graph/passes/predicate.hpp"
#include "pyopenvino/graph/passes/transformations.hpp"

namespace py = pybind11;

void regmodule_graph_passes(py::module m) {
    py::module m_passes = m.def_submodule("passes", "Package openvino.passes wraps ov::passes");
    reg_passes_predicate(m_passes);
    regclass_passes_PassBase(m_passes);
    regclass_passes_ModelPass(m_passes);
    regclass_passes_Matcher(m_passes);
    regclass_passes_MatcherPass(m_passes);
    regclass_passes_GraphRewrite(m_passes);
    regclass_transformations(m_passes);
    regclass_passes_Manager(m_passes);
    reg_passes_pattern_ops(m_passes);
}
