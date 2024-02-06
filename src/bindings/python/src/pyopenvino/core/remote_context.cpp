// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/remote_context.hpp"

#include <pybind11/stl.h>

#include <openvino/core/any.hpp>
#include <openvino/runtime/remote_context.hpp>
#include <openvino/runtime/tensor.hpp>
#include <openvino/core/type/element_type.hpp>

#ifdef PY_ENABLE_OPENCL
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#endif  // PY_ENABLE_OPENCL
#ifdef PY_ENABLE_LIBVA
#include <openvino/runtime/intel_gpu/ocl/va.hpp>
#endif  // PY_ENABLE_LIBVA

#include "common.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

void regclass_RemoteContext(py::module m) {
    py::class_<ov::RemoteContext, std::shared_ptr<ov::RemoteContext>> cls(m, "RemoteContext");

    cls.def("get_device_name", &ov::RemoteContext::get_device_name);

    cls.def(
        "get_params",
        [](ov::RemoteContext& self) {
            return self.get_params();
        });

    // This returns ov::RemoteTensor
    // TODO: think about renaming suggestion? create_tensor -> create_remote/device_tensor
    // device seems more natural with OV domain language.
    cls.def(
        "create_device_tensor",
        [](ov::RemoteContext& self,
           const ov::element::Type& type,
           const ov::Shape& shape,
           const std::map<std::string, py::object>& properties) {
            auto _properties = Common::utils::properties_to_any_map(properties);
            return self.create_tensor(type, shape, _properties);
        },
        py::arg("type"),
        py::arg("shape"),
        py::arg("properties"));

    // This returns ov::Tensor
    // TODO: think about renaming suggestion? Or keeping it with rename of above?
    cls.def(
        "create_host_tensor",
        [](ov::RemoteContext& self,
           const ov::element::Type& type,
           const ov::Shape& shape) {
            return self.create_host_tensor(type, shape);
        },
        py::arg("type"),
        py::arg("shape"));
}

// This is namespace: ov::intel_gpu::ocl <-- should following classes be exposed like this?
// IMO these can inherit flatten "openvino.gpu" space instead of ocl one. TBD
#ifdef PY_ENABLE_OPENCL
void regclass_ClContext(py::module m) {
    py::class_<ov::intel_gpu::ocl::ClContext, ov::RemoteContext, std::shared_ptr<ov::intel_gpu::ocl::ClContext>> cls(m, "ClContext");
}
#endif  // PY_ENABLE_OPENCL

#ifdef PY_ENABLE_LIBVA
void regclass_VADisplayWrapper(py::module m) {
    py::class_<VADisplayWrapper, std::shared_ptr<VADisplayWrapper>> cls(m, "VADisplayWrapper");

    // Use of the pointer obtained from external library to wrap around:
    cls.def(py::init([](VADisplay device) {
        return VADisplayWrapper(device);
    }),
    py::arg("device"));

    cls.def(
        "release",
        [](VADisplayWrapper& self) {
            self.release();
        });
}

void regclass_VAContext(py::module m) {
    py::class_<ov::intel_gpu::ocl::VAContext, ov::intel_gpu::ocl::ClContext, std::shared_ptr<ov::intel_gpu::ocl::VAContext>> cls(m, "VAContext");

    cls.def(py::init([](ov::Core& core, /* VADisplay */ VADisplayWrapper& display, int target_tile_id) {
                return ov::intel_gpu::ocl::VAContext(core, display.get_display_ptr(), target_tile_id);
            }),
            py::arg("core"),
            py::arg("display"),
            py::arg("target_tile_id") = -1);

    cls.def(
        "create_tensor_nv12",
        [](ov::intel_gpu::ocl::VAContext& self,
           const size_t height,
           const size_t width,
           const /* VASurfaceID */ uint32_t nv12_surface) {
            auto nv12_pair = self.create_tensor_nv12(height, width, nv12_surface);
            return py::make_tuple(nv12_pair.first, nv12_pair.second);
        },
        py::arg("height"),
        py::arg("width"),
        py::arg("nv12_surface"));

    cls.def(
        "create_tensor",
        [](ov::intel_gpu::ocl::VAContext& self,
           const ov::element::Type& type,
           const ov::Shape shape,
           const /* VASurfaceID */ uint32_t surface,
           const uint32_t plane) {
            return self.create_tensor(type, shape, surface, plane);
        },
        py::arg("type"),
        py::arg("shape"),
        py::arg("surface"),
        py::arg("plane") = 0);
}
#endif  // PY_ENABLE_LIBVA
