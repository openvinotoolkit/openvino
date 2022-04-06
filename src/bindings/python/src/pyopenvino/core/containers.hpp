// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <vector>

#include <pybind11/pybind11.h>

#include <openvino/runtime/tensor.hpp>

namespace py = pybind11;

namespace Containers {
    using TensorIndexMap = std::map<size_t, ov::Tensor>;
    using TensorNameMap = std::map<std::string, ov::Tensor>;

    void regclass_TensorIndexMap(py::module m);
    void regclass_TensorNameMap(py::module m);
}
