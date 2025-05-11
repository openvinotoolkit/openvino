// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>

#include <string>

#include "extension.hpp"
#include "graph_iterator.hpp"
#include "decoder_base.hpp"

namespace py = pybind11;

PYBIND11_MODULE(py_tensorflow_frontend, m) {
    regclass_frontend_tensorflow_ConversionExtension(m);
    regclass_frontend_tensorflow_OpExtension(m);
    regclass_frontend_tensorflow_graph_iterator(m);
    regclass_frontend_tensorflow_decoder_base(m);
}
