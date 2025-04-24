// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/variable_state.hpp"

#include <pybind11/pybind11.h>

#include "openvino/runtime/variable_state.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_VariableState(py::module m) {
    py::class_<ov::VariableState, std::shared_ptr<ov::VariableState>> variable_st(m, "VariableState");
    variable_st.doc() = "openvino.VariableState class.";

    variable_st.def("__repr__", [](const ov::VariableState& self) {
        return Common::get_simple_repr(self);
    });

    variable_st.def("reset",
                    &ov::VariableState::reset,
                    R"(
        Reset internal variable state for relevant infer request,
        to a value specified as default for according node.
    )");

    variable_st.def_property_readonly("name",
                                      &ov::VariableState::get_name,
                                      R"(
        Gets name of current variable state.

        :return: A string representing a state name.
        :rtype: str
    )");

    variable_st.def_property("state",
                             &ov::VariableState::get_state,
                             &ov::VariableState::set_state,
                             R"(
        Gets/sets variable state.
    )");
}
