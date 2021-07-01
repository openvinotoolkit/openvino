// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ngraph/op/util/op_annotations.hpp"
#include "pyngraph/ops/util/op_annotations.hpp"

namespace py = pybind11;

void regclass_pyngraph_op_util_OpAnnotations(py::module m)
{
    py::class_<ngraph::op::util::OpAnnotations, std::shared_ptr<ngraph::op::util::OpAnnotations>>
        opAnnotations(m, "OpAnnotations");
    opAnnotations.def(py::init<>());
}
