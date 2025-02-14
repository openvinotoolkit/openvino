// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "graph_iterator.hpp"

#include "openvino/frontend/tensorflow/graph_iterator.hpp"

namespace py = pybind11;

using namespace ov::frontend;
using ov::Any;

void regclass_frontend_tensorflow_graph_iterator(py::module m) {
    py::class_<ov::frontend::tensorflow::GraphIterator, PyGraphIterator, std::shared_ptr<ov::frontend::tensorflow::GraphIterator>>(m, "_FrontEndPyGraphIterator")
            .def(py::init<>());

}