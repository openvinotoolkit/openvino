// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/index_reduction.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openvino/op/op.hpp"
#include "pyopenvino/core/common.hpp"
#include "pyopenvino/graph/ops/util/index_reduction.hpp"

namespace py = pybind11;

void regclass_graph_op_util_IndexReduction(py::module m) {
    py::class_<ov::op::util::IndexReduction, std::shared_ptr<ov::op::util::IndexReduction>> indexReduction(
        m,
        "IndexReduction");

    indexReduction.def("get_reduction_axis", &ov::op::util::IndexReduction::get_reduction_axis);
    indexReduction.def("set_reduction_axis", &ov::op::util::IndexReduction::set_reduction_axis);
    indexReduction.def("get_index_element_type", &ov::op::util::IndexReduction::get_index_element_type);
    indexReduction.def("set_index_element_type", &ov::op::util::IndexReduction::set_index_element_type);

    indexReduction.def_property("reduction_axis",
                                &ov::op::util::IndexReduction::get_reduction_axis,
                                &ov::op::util::IndexReduction::set_reduction_axis);
    indexReduction.def_property("index_element_type",
                                &ov::op::util::IndexReduction::get_index_element_type,
                                &ov::op::util::IndexReduction::set_index_element_type);
    indexReduction.def("__repr__", [](const ov::op::util::IndexReduction& self) {
        return Common::get_simple_repr(self);
    });
}
