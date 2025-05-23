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
using ov::Any;


void regclass_frontend_pytorch_decoder(py::module m) {
    py::class_<pytorch::TorchDecoder, IDecoder, PyDecoder, std::shared_ptr<pytorch::TorchDecoder>>(m, "_FrontEndPytorchDecoder")
        .def(py::init<>());

    auto type_module = m.def_submodule("_Type");

    // Register classes for TorchScript type system
    py::class_<type::Tensor>(type_module, "Tensor").
        def(py::init<Any>());
    py::class_<type::Complex>(type_module, "Complex").
        def(py::init<Any>());
    py::class_<type::List>(type_module, "List").
        def(py::init<Any>());
    py::class_<type::Str>(type_module, "Str").
        def(py::init<>());
    py::class_<type::PyNone>(type_module, "PyNone").
        def(py::init<>());
    py::class_<type::PyScalar>(type_module, "PyScalar").
        def(py::init<Any>());
}
