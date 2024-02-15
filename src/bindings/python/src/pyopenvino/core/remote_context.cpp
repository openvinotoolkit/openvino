// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/remote_context.hpp"

#include <pybind11/stl.h>

#include <openvino/runtime/core.hpp>

#include "common.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

void regclass_RemoteContext(py::module m) {
    py::class_<RemoteContextWrapper, std::shared_ptr<RemoteContextWrapper>> cls(m, "RemoteContext");

    cls.def("get_device_name", [](RemoteContextWrapper& self) {
        return self.context.get_device_name();
    });

    cls.def("get_params", [](RemoteContextWrapper& self) {
        return self.context.get_params();
    });

    cls.def(
        "create_tensor",
        [](RemoteContextWrapper& self,
           const ov::element::Type& type,
           const ov::Shape& shape,
           const std::map<std::string, py::object>& properties) {
            auto _properties = Common::utils::properties_to_any_map(properties);
            return RemoteTensorWrapper(self.context.create_tensor(type, shape, _properties));
        },
        py::arg("type"),
        py::arg("shape"),
        py::arg("properties"));

    cls.def(
        "create_host_tensor",
        [](RemoteContextWrapper& self, const ov::element::Type& type, const ov::Shape& shape) {
            return self.context.create_host_tensor(type, shape);
        },
        py::arg("type"),
        py::arg("shape"));
}

void regclass_VAContext(py::module m) {
    py::class_<VAContextWrapper, RemoteContextWrapper, std::shared_ptr<VAContextWrapper>> cls(m, "VAContext");

    cls.def(py::init([](ov::Core& core, void* display, int target_tile_id) {
                ov::AnyMap context_params = {
                    {ov::intel_gpu::context_type.name(), ov::intel_gpu::ContextType::VA_SHARED},
                    {ov::intel_gpu::va_device.name(), display},
                    {ov::intel_gpu::tile_id.name(), target_tile_id}};
                auto ctx = core.create_context("GPU", context_params);
                return VAContextWrapper(ctx);
            }),
            py::arg("core"),
            py::arg("display"),
            py::arg("target_tile_id") = -1);

    cls.def(
        "create_tensor_nv12",
        [](VAContextWrapper& self, const size_t height, const size_t width, const uint32_t nv12_surface) {
            ov::AnyMap tensor_params = {
                {ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::VA_SURFACE},
                {ov::intel_gpu::dev_object_handle.name(), nv12_surface},
                {ov::intel_gpu::va_plane.name(), uint32_t(0)}};
            auto y_tensor = self.context.create_tensor(ov::element::u8, {1, height, width, 1}, tensor_params);
            tensor_params[ov::intel_gpu::va_plane.name()] = uint32_t(1);
            auto uv_tensor = self.context.create_tensor(ov::element::u8, {1, height / 2, width / 2, 2}, tensor_params);
            return py::make_tuple(VASurfaceTensorWrapper(y_tensor), VASurfaceTensorWrapper(uv_tensor));
        },
        py::arg("height"),
        py::arg("width"),
        py::arg("nv12_surface"));

    cls.def(
        "create_tensor",
        [](VAContextWrapper& self,
           const ov::element::Type& type,
           const ov::Shape shape,
           const uint32_t surface,
           const uint32_t plane) {
            ov::AnyMap params = {{ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::VA_SURFACE},
                                 {ov::intel_gpu::dev_object_handle.name(), surface},
                                 {ov::intel_gpu::va_plane.name(), plane}};
            return VASurfaceTensorWrapper(self.context.create_tensor(type, shape, params));
        },
        py::arg("type"),
        py::arg("shape"),
        py::arg("surface"),
        py::arg("plane") = 0);
}
