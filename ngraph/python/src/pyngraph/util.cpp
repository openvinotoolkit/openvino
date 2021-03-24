// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/numpy.h>

#include "pyngraph/util.hpp"

namespace py = pybind11;

void* numpy_to_c(py::array a)
{
    py::buffer_info info = a.request();
    return info.ptr;
}

void regmodule_pyngraph_util(py::module m)
{
    py::module mod = m.def_submodule("util", "ngraph.impl.util");
    mod.def("numpy_to_c", &numpy_to_c);
}
