// Copyright (C) 2018-2026 Intel Corporation
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
            py::gil_scoped_release release;
            return RemoteTensorWrapper(self.context.create_tensor(type, shape, _properties));
        },
        py::arg("type"),
        py::arg("shape"),
        py::arg("properties"),
        R"(
            Allocates memory tensor in device memory or wraps user-supplied memory handle
            using the specified tensor description and low-level device-specific parameters.
            Returns the object that implements the RemoteTensor interface.

            GIL is released while running this function.

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
        py::call_guard<py::gil_scoped_release>(),
        py::arg("type"),
        py::arg("shape"),
        R"(
            This method is used to create a host tensor object friendly for the device in
            current context. For example, GPU context may allocate USM host memory
            (if corresponding extension is available), which could be more efficient
            than regular host memory.

            GIL is released while running this function.

            :param type: Defines the element type of the tensor.
            :type type: openvino.Type
            :param shape: Defines the shape of the tensor.
            :type shape: openvino.Shape
            :return: A tensor instance with device friendly memory.
            :rtype: openvino.Tensor
        )");
}
