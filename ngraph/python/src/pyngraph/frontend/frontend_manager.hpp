// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regclass_pyngraph_FrontEndManager(py::module m);
void regclass_pyngraph_NotImplementedFailureFrontEnd(py::module m);
void regclass_pyngraph_InitializationFailureFrontEnd(py::module m);
void regclass_pyngraph_OpConversionFailureFrontEnd(py::module m);
void regclass_pyngraph_OpValidationFailureFrontEnd(py::module m);
void regclass_pyngraph_GeneralFailureFrontEnd(py::module m);

