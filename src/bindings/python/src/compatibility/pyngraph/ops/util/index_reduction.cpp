// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/index_reduction.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngraph/op/op.hpp"
#include "pyngraph/ops/util/index_reduction.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_util_IndexReduction(py::module m) {
    py::class_<ngraph::op::util::IndexReduction, std::shared_ptr<ngraph::op::util::IndexReduction>> indexReduction(
        m,
        "IndexReduction",
        py::module_local());

    indexReduction.def("get_reduction_axis", &ngraph::op::util::IndexReduction::get_reduction_axis);
    indexReduction.def("set_reduction_axis", &ngraph::op::util::IndexReduction::set_reduction_axis);
    indexReduction.def("get_index_element_type", &ngraph::op::util::IndexReduction::get_index_element_type);
    indexReduction.def("set_index_element_type", &ngraph::op::util::IndexReduction::set_index_element_type);

    indexReduction.def_property("reduction_axis",
                                &ngraph::op::util::IndexReduction::get_reduction_axis,
                                &ngraph::op::util::IndexReduction::set_reduction_axis);
    indexReduction.def_property("index_element_type",
                                &ngraph::op::util::IndexReduction::get_index_element_type,
                                &ngraph::op::util::IndexReduction::set_index_element_type);
}
