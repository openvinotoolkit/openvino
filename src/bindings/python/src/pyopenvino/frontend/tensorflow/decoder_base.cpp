// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "decoder_base.hpp"


namespace py = pybind11;

using namespace ov::frontend;
using ov::Any;

void regclass_frontend_tensorflow_decoder_base(py::module m) {
    py::class_<ov::frontend::tensorflow::DecoderBase, IDecoder, PyDecoderBase, std::shared_ptr<ov::frontend::tensorflow::DecoderBase>> cls(m, "_FrontEndDecoderBase");
    cls.def(py::init<>());

}