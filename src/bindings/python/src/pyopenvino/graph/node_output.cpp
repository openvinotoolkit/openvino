// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/node_output.hpp"

#include <pybind11/stl.h>

#include "dict_attribute_visitor.hpp"
#include "pyopenvino/graph/node_output.hpp"

namespace py = pybind11;

template void regclass_graph_Output<ov::Node>(py::module m, std::string typestring);
template void regclass_graph_Output<const ov::Node>(py::module m, std::string typestring);
