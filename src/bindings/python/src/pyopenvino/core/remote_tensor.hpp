// Copyright (C) 2018-2025 Intel Corporation
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

class VASurfaceTensorWrapper : public RemoteTensorWrapper {
public:
    VASurfaceTensorWrapper(ov::RemoteTensor& _tensor): RemoteTensorWrapper{_tensor} {}

    VASurfaceTensorWrapper(ov::RemoteTensor&& _tensor): RemoteTensorWrapper{std::move(_tensor)} {}

    uint32_t surface_id() {
        return tensor.get_params().at(ov::intel_gpu::dev_object_handle.name()).as<uint32_t>();
    }

    uint32_t plane_id() {
        return tensor.get_params().at(ov::intel_gpu::va_plane.name()).as<uint32_t>();
    }
};

void regclass_VASurfaceTensor(py::module m);
