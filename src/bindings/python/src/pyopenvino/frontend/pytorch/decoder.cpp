// Copyright (C) 2018-2023 Intel Corporation
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
    py::class_<Decoder, PyDecoder, std::shared_ptr<Decoder>>(m, "_FrontEndPytorchDecoder")
        .def(py::init<>());

    auto type_module = m.def_submodule("_Type");

    // Register classes for TS type system

    py::class_<type::Tensor>(type_module, "Tensor").
        def(py::init<Any>());
    py::class_<type::List>(type_module, "List").
        def(py::init<Any>());
    py::class_<type::Str>(type_module, "Str").
        def(py::init<>());
    type_module.def("print", type::print);
}
