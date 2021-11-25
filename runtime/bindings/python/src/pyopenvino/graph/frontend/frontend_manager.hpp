// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

void regclass_graph_FrontEndManager(py::module m);
void regclass_graph_NotImplementedFailureFrontEnd(py::module m);
void regclass_graph_InitializationFailureFrontEnd(py::module m);
void regclass_graph_OpConversionFailureFrontEnd(py::module m);
void regclass_graph_OpValidationFailureFrontEnd(py::module m);
void regclass_graph_GeneralFailureFrontEnd(py::module m);

