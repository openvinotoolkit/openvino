// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <pybind11/pybind11.h>

#include <openvino/core/any.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/runtime/intel_gpu/remote_properties.hpp>
#include <openvino/runtime/remote_tensor.hpp>

namespace py = pybind11;

class RemoteTensorWrapper {
public:
    RemoteTensorWrapper() {}

    RemoteTensorWrapper(ov::RemoteTensor& _tensor): tensor{_tensor} {}

    RemoteTensorWrapper(ov::RemoteTensor&& _tensor): tensor{std::move(_tensor)} {}

    ov::RemoteTensor tensor;
};

void regclass_RemoteTensor(py::module m);
