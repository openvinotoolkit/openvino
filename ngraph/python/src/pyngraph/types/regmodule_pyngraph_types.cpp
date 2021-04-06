// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>

#include "pyngraph/types/regmodule_pyngraph_types.hpp"

namespace py = pybind11;

void regmodule_pyngraph_types(py::module m)
{
    regclass_pyngraph_Type(m);
}
