// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyngraph/types/regmodule_pyngraph_types.hpp"

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regmodule_pyngraph_types(py::module m) {
    regclass_pyngraph_Type(m);
}
