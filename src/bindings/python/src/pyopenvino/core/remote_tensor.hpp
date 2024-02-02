// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regclass_RemoteTensor(py::module m);
#ifdef PY_ENABLE_OPENCL
void regclass_ClImage2DTensor(py::module m);
#endif  // PY_ENABLE_OPENCL
#ifdef PY_ENABLE_LIBVA
void regclass_VASurfaceTensor(py::module m);
#endif  // PY_ENABLE_LIBVA
