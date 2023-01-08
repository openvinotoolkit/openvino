// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/op_annotations.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pyngraph/ops/util/op_annotations.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_util_OpAnnotations(py::module m) {
    py::class_<ngraph::op::util::OpAnnotations, std::shared_ptr<ngraph::op::util::OpAnnotations>> opAnnotations(
        m,
        "OpAnnotations",
        py::module_local());
    opAnnotations.def(py::init<>());
}
