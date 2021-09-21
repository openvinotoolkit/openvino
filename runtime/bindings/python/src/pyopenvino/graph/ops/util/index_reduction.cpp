// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/index_reduction.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

<<<<<<< HEAD
#include "ngraph/op/op.hpp"
#include "pyngraph/ops/util/index_reduction.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_util_IndexReduction(py::module m) {
    py::class_<ngraph::op::util::IndexReduction, std::shared_ptr<ngraph::op::util::IndexReduction>> indexReduction(
=======
#include "openvino/op/op.hpp"
#include "pyopenvino/graph/ops/util/index_reduction.hpp"

namespace py = pybind11;

void regclass_graph_op_util_IndexReduction(py::module m) {
    py::class_<ov::op::util::IndexReduction, std::shared_ptr<ov::op::util::IndexReduction>> indexReduction(
>>>>>>> c724fc613... Extend core with new containers and rename namespaces/includes
        m,
        "IndexRedection");

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
}
