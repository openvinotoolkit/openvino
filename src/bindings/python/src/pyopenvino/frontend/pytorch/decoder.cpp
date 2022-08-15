// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "decoder.hpp"

namespace py = pybind11;

using namespace ov::frontend::pytorch;
using ov::Any;
using ov::PartialShape;
using ov::OutputVector;
using ov::Node;

namespace {

    

}

void regclass_frontend_pytorch_decoder(py::module m) {
    /*py::class_<ConversionExtension, ConversionExtension::Ptr, ov::frontend::ConversionExtensionBase> _ext(
        m,
        "_ConversionExtensionONNX",
        py::dynamic_attr());*/
 
    py::class_<Decoder, PyDecoder, std::shared_ptr<Decoder>>(m, "_FrontEndPytorchDecoder")
        .def(py::init<>());
    // There is no need to register all Decoder methods here. TODO: why? How can they enumerate them without our help? 
    // Looks like they statically register all the methods from PYBIND11_OVERRIDE_PURE when mentioning it in every member of PyDecoder.
        // .def("inputs", &Decoder::inputs);
}
