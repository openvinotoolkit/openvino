// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "decoder.hpp"

#include "openvino/frontend/decoder.hpp"

namespace py = pybind11;

using namespace ov::frontend;


void regclass_frontend_jax_decoder(py::module m) {
    py::class_<jax::JaxDecoder, IDecoder, PyDecoder, std::shared_ptr<jax::JaxDecoder>>(m, "_FrontEndJaxDecoder")
        .def(py::init<>());
}
