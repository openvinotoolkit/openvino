// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <pybind11/pybind11.h>

#include "openvino/frontend/graph_iterator.hpp"

namespace py = pybind11;

class PyIGraphIterator : public ov::frontend::IGraphIterator {
public:
    using IGraphIterator::IGraphIterator; // Inherit constructors
};

void regclass_frontend_IGraphIterator(py::module m);
