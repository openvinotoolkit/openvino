// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <vector>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <openvino/runtime/tensor.hpp>
#include <openvino/core/node_output.hpp>

namespace py = pybind11;

namespace Containers {
    using TensorIndexMap = std::map<size_t, ov::runtime::Tensor>;
    using TensorNameMap = std::map<std::string, ov::runtime::Tensor>;

    void regclass_TensorIndexMap(py::module m);
    void regclass_TensorNameMap(py::module m);
}
