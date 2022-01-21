// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/containers.hpp"

#include <pybind11/stl_bind.h>

PYBIND11_MAKE_OPAQUE(Containers::TensorIndexMap);
PYBIND11_MAKE_OPAQUE(Containers::TensorNameMap);

namespace py = pybind11;

namespace Containers {

void regclass_TensorIndexMap(py::module m) {
    py::bind_map<TensorIndexMap>(m, "TensorIndexMap");
}

void regclass_TensorNameMap(py::module m) {
    py::bind_map<TensorNameMap>(m, "TensorNameMap");
}
}  // namespace Containers
