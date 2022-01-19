// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/variable_state.hpp"

#include <pybind11/pybind11.h>

#include "openvino/runtime/variable_state.hpp"

namespace py = pybind11;

void regclass_VariableState(py::module m) {
    py::class_<ov::runtime::VariableState, std::shared_ptr<ov::runtime::VariableState>> variable_st(m, "VariableState");

    variable_st.def("reset", &ov::runtime::VariableState::reset);

    variable_st.def_property_readonly("name", &ov::runtime::VariableState::get_name);

    variable_st.def_property("state", &ov::runtime::VariableState::get_state, &ov::runtime::VariableState::set_state);
}
