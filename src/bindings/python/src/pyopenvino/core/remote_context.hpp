// Copyright (C) 2018-2024 Intel Corporation
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

#ifdef PY_ENABLE_GPU
#ifndef _WIN32
#    include <va/va.h>
#endif  // _WIN32
#endif  // PY_ENABLE_GPU

namespace py = pybind11;

class RemoteContextWrapper {
public:
    RemoteContextWrapper() {}

    RemoteContextWrapper(ov::RemoteContext& _context): context{_context} {}

    RemoteContextWrapper(ov::RemoteContext&& _context): context{std::move(_context)} {}

    ov::RemoteContext context;
};

void regclass_RemoteContext(py::module m);

#ifdef PY_ENABLE_GPU
class ClContextWrapper : public RemoteContextWrapper {
public:
    ClContextWrapper(ov::RemoteContext& _context): RemoteContextWrapper{_context} {}

    ClContextWrapper(ov::RemoteContext&& _context): RemoteContextWrapper{std::move(_context)} {}
};

void regclass_ClContext(py::module m);

#ifndef _WIN32
class VADisplayWrapper {
public:
    VADisplayWrapper(VADisplay device) {
        va_display = device;
        fd = -1;
    }

    VADisplay get_display_ptr() {
        return va_display;
    }

    void release() {
        PyErr_WarnEx(PyExc_RuntimeWarning,
                     "Release of VADisplay was not succesful! The display is referencing "
                     "the other pointer. Owner is responsible for memory release.",
                     2);
    }

private:
    int fd;
    VADisplay va_display;
};

void regclass_VADisplayWrapper(py::module m);

class VAContextWrapper : public ClContextWrapper {
public:
    VAContextWrapper(ov::RemoteContext& _context): ClContextWrapper{_context} {}

    VAContextWrapper(ov::RemoteContext&& _context): ClContextWrapper{std::move(_context)} {}
};

void regclass_VAContext(py::module m);
#endif  // _WIN32
#endif  // PY_ENABLE_GPU
