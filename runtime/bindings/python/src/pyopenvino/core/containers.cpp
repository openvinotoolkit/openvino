
// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/containers.hpp"

#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(Containers::TensorIndexMap);
PYBIND11_MAKE_OPAQUE(Containers::TensorNameMap);

namespace py = pybind11;

namespace Containers {

void regclass_TensorIndexMap(py::module m) {
    auto tensor_index_map = py::bind_map<TensorIndexMap>(m, "TensorIndexMap");
}

void regclass_TensorNameMap(py::module m) {
    auto tensor_name_map = py::bind_map<TensorNameMap>(m, "TensorNameMap");
}
}  // namespace Containers
