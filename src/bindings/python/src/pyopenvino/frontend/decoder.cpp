// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "decoder.hpp"

#include "openvino/frontend/decoder.hpp"

namespace py = pybind11;

using namespace ov::frontend;

void regclass_frontend_IDecoder(py::module m) {
    py::class_<IDecoder, PyIDecoder, std::shared_ptr<IDecoder>>(m, "_IDecoder");
}
