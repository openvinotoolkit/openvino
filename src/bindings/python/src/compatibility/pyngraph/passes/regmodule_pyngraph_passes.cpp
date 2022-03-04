// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyngraph/passes/regmodule_pyngraph_passes.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regmodule_pyngraph_passes(py::module m) {
    py::module m_passes = m.def_submodule("passes", "Package ngraph.impl.passes wraps ngraph::passes");
    regclass_pyngraph_passes_Manager(m_passes);
}
