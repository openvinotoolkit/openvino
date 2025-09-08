// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <pybind11/pybind11.h>

#include <openvino/core/any.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/runtime/intel_gpu/remote_properties.hpp>
#include <openvino/runtime/remote_context.hpp>
#include <openvino/runtime/tensor.hpp>

#include "openvino/core/except.hpp"
#include "pyopenvino/core/remote_tensor.hpp"

namespace py = pybind11;

class RemoteContextWrapper {
public:
    RemoteContextWrapper() {}

    RemoteContextWrapper(ov::RemoteContext& _context): context{_context} {}

    RemoteContextWrapper(ov::RemoteContext&& _context): context{std::move(_context)} {}

    ov::RemoteContext context;
};

void regclass_RemoteContext(py::module m);

class VAContextWrapper : public RemoteContextWrapper {
public:
    VAContextWrapper(ov::RemoteContext& _context): RemoteContextWrapper{_context} {}

    VAContextWrapper(ov::RemoteContext&& _context): RemoteContextWrapper{std::move(_context)} {}
};

void regclass_VAContext(py::module m);
