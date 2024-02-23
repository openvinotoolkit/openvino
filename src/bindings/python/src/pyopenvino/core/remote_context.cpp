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

    cls.def(
        "get_device_name",
        [](RemoteContextWrapper& self) {
            return self.context.get_device_name();
        },
        R"(
        Returns name of a device on which the context is allocated.

        :return: A device name string in fully specified format `<device_name>[.<device_id>[.<tile_id>]]`.
        :rtype: str
    )");

    cls.def(
        "get_params",
        [](RemoteContextWrapper& self) {
            return self.context.get_params();
        },
        R"(
        Returns a dict of device-specific parameters required for low-level
        operations with the underlying context.
        Parameters include device/context handles, access flags, etc.
        Content of the returned dict depends on remote execution context that is
        currently set on the device (working scenario).

        :return: A dictionary of device-specific parameters.
        :rtype: dict
    )");

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
        py::arg("properties"),
        R"(
            Allocates memory tensor in device memory or wraps user-supplied memory handle
            using the specified tensor description and low-level device-specific parameters.
            Returns the object that implements the RemoteTensor interface.

            :param type: Defines the element type of the tensor.
            :type type: openvino.Type
            :param shape: Defines the shape of the tensor.
            :type shape: openvino.Shape
            :param properties: dict of the low-level tensor object parameters.
            :type properties: dict
            :return: A remote tensor instance.
            :rtype: openvino.RemoteTensor
        )");

    cls.def(
        "create_host_tensor",
        [](RemoteContextWrapper& self, const ov::element::Type& type, const ov::Shape& shape) {
            return self.context.create_host_tensor(type, shape);
        },
        py::arg("type"),
        py::arg("shape"),
        R"(
            This method is used to create a host tensor object friendly for the device in
            current context. For example, GPU context may allocate USM host memory
            (if corresponding extension is available), which could be more efficient
            than regular host memory.

            :param type: Defines the element type of the tensor.
            :type type: openvino.Type
            :param shape: Defines the shape of the tensor.
            :type shape: openvino.Shape
            :return: A tensor instance with device friendly memory.
            :rtype: openvino.Tensor
        )");
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
            py::arg("target_tile_id") = -1,
            R"(
            Constructs remote context object from valid VA display handle.

            :param core: OpenVINO Runtime Core object.
            :type core: openvino.Core
            :param device: A valid `VADisplay` to create remote context from.
            :type device: Any
            :param target_tile_id: Desired tile id within given context for multi-tile system.
                                   Default value (-1) means that root device should be used.
            :type target_tile_id: int
            :return: A context instance.
            :rtype: openvino.VAContext
        )");

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
        py::arg("nv12_surface"),
        R"(
            This function is used to obtain a NV12 tensor from NV12 VA decoder output.
            The result contains two remote tensors for Y and UV planes of the surface.

            :param height: A height of Y plane.
            :type height: int
            :param width: A width of Y plane
            :type width: int
            :param nv12_surface: NV12 `VASurfaceID` to create NV12 from.
            :type nv12_surface: int
            :return: A pair of remote tensors for each plane.
            :rtype: Tuple[openvino.VASurfaceTensor, openvino.VASurfaceTensor]
        )");

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
        py::arg("plane") = 0,
        R"(
            Create remote tensor from VA surface handle.

            :param type: Defines the element type of the tensor.
            :type type: openvino.Type
            :param shape: Defines the shape of the tensor.
            :type shape: openvino.Shape
            :param surface: `VASurfaceID` to create tensor from.
            :type surface: int
            :param plane: An index of a plane inside `VASurfaceID` to create tensor from. Default: 0
            :type plane: int
            :return: A remote tensor instance wrapping `VASurfaceID`.
            :rtype: openvino.VASurfaceTensor
        )");
}
