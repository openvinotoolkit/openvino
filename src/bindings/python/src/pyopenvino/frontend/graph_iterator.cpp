// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_iterator.hpp"

#include "openvino/frontend/graph_iterator.hpp"

namespace py = pybind11;

using namespace ov::frontend;

void regclass_frontend_IGraphIterator(py::module m) {
    py::class_<IGraphIterator, PyIGraphIterator, std::shared_ptr<IGraphIterator>>(m, "_IGraphIterator");
}
