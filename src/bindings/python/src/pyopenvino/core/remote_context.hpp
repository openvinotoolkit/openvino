// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <pybind11/pybind11.h>

#include "openvino/core/except.hpp"

#ifdef PY_ENABLE_LIBVA
#include <va/va.h>
#include <va/va_drm.h>
#include <fcntl.h>
#endif  // PY_ENABLE_LIBVA

namespace py = pybind11;

void regclass_RemoteContext(py::module m);

#ifdef PY_ENABLE_OPENCL
void regclass_ClContext(py::module m);
#endif  // PY_ENABLE_OPENCL

#ifdef PY_ENABLE_LIBVA
class VADisplayWrapper {
public:
    // Get VADisplay based on path passed by user:
    VADisplayWrapper(std::string& device) {
        fd = open(device.c_str(), O_RDWR);
        if (fd < 0) {
            OPENVINO_THROW("Failed to open DRM device!");
        }

        va_display = vaGetDisplayDRM(fd);

        int major_ver, minor_ver;
        VAStatus va_status = vaInitialize(va_display, &major_ver, &minor_ver);
        if (va_status != VA_STATUS_SUCCESS) {
            close(fd);
            OPENVINO_THROW("Failed to initialize libva!");
        }
    }

    // Wrap VADisplay to be recognized by OV:
    VADisplayWrapper(VADisplay device) {
        va_display = device;
        fd = -1;
    }

    VADisplay get_display_ptr() {
        return va_display;
    }

    // Terminate the display and clean up:
    void release() {
        if (fd != -1) {
            vaTerminate(va_display);
            close(fd);
        }
        else {
            PyErr_WarnEx(PyExc_DeprecationWarning,
                         "Release of VADisplay was not succesful! The display is referencing "
                         "the other pointer. Owner is responsible for memory release.",
                         2);
        }
    }

private:
    int fd;
    VADisplay va_display;
};

void regclass_VADisplayWrapper(py::module m);

void regclass_VAContext(py::module m);
#endif  // PY_ENABLE_LIBVA
