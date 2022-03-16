// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/graph/passes/regmodule_graph_passes.hpp"
#include "pyopenvino/graph/passes/manager.hpp"
#include "pyopenvino/graph/passes/graph_rewrite.hpp"
#include "pyopenvino/graph/passes/matcher_pass.hpp"
#include "pyopenvino/graph/passes/model_pass.hpp"
#include "pyopenvino/graph/passes/pass_base.hpp"
#include "pyopenvino/graph/passes/pattern_ops.hpp"
#include "pyopenvino/graph/passes/transformations.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regmodule_graph_passes(py::module m) {
    py::module m_passes = m.def_submodule("passes", "Package openvino.runtime.passes wraps ov::passes");
    regclass_PassBase(m_passes);
    regclass_ModelPass(m_passes);
    regclass_GraphRewrite(m_passes);
    regclass_Matcher(m_passes);
    regclass_MatcherPass(m_passes);
    regclass_transformations(m_passes);
    regclass_Manager(m_passes);
    reg_pass_pattern_ops(m_passes);
}
