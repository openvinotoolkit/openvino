// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/remote_tensor.hpp"

#include <pybind11/stl.h>

#include "common.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

void regclass_RemoteTensor(py::module m) {
    py::class_<RemoteTensorWrapper, std::shared_ptr<RemoteTensorWrapper>> cls(m,
                                                                              "RemoteTensor",
                                                                              py::base<ov::Tensor>());

    cls.def(
        "get_device_name",
        [](RemoteTensorWrapper& self) {
            return self.tensor.get_device_name();
        },
        R"(
        Returns name of a device on which the tensor is allocated.

        :return: A device name string in fully specified format `<device_name>[.<device_id>[.<tile_id>]]`.
        :rtype: str
    )");

    cls.def(
        "get_params",
        [](RemoteTensorWrapper& self) {
            return self.tensor.get_params();
        },
        R"(
        Returns a dict of device-specific parameters required for low-level
        operations with the underlying tensor.
        Parameters include device/context/surface/buffer handles, access flags, etc.
        Content of the returned dict depends on remote execution context that is
        currently set on the device (working scenario).

        :return: A dictionary of device-specific parameters.
        :rtype: dict
    )");

    cls.def(
        "copy_to",
        [](RemoteTensorWrapper& self, py::object& dst) {
            Common::utils::raise_not_implemented();
        },
        R"(
        This method is not implemented.
    )");

    cls.def_property_readonly(
        "data",
        [](RemoteTensorWrapper& self) {
            Common::utils::raise_not_implemented();
        },
        R"(
        This property is not implemented.
    )");

    cls.def_property(
        "bytes_data",
        [](RemoteTensorWrapper& self) {
            Common::utils::raise_not_implemented();
        },
        [](RemoteTensorWrapper& self, py::object& other) {
            Common::utils::raise_not_implemented();
        },
        R"(
        This property is not implemented.
    )");

    cls.def_property(
        "str_data",
        [](RemoteTensorWrapper& self) {
            Common::utils::raise_not_implemented();
        },
        [](RemoteTensorWrapper& self, py::object& other) {
            Common::utils::raise_not_implemented();
        },
        R"(
        This property is not implemented.
    )");

    cls.def("__repr__", [](const RemoteTensorWrapper& self) {
        std::stringstream ss;

        ss << "shape" << self.tensor.get_shape() << " type: " << self.tensor.get_element_type();

        return "<" + Common::get_class_name(self) + ": " + ss.str() + ">";
    });
}

void regclass_VASurfaceTensor(py::module m) {
    py::class_<VASurfaceTensorWrapper, RemoteTensorWrapper, std::shared_ptr<VASurfaceTensorWrapper>> cls(
        m,
        "VASurfaceTensor");

    cls.def_property_readonly(
        "surface_id",
        [](VASurfaceTensorWrapper& self) {
            return self.surface_id();
        },
        R"(
        Returns ID of underlying video decoder surface.

        :return: VASurfaceID of the tensor.
        :rtype: int
    )");

    cls.def_property_readonly(
        "plane_id",
        [](VASurfaceTensorWrapper& self) {
            return self.plane_id();
        },
        R"(
        Returns plane ID of underlying video decoder surface.

        :return: Plane ID of underlying video decoder surface.
        :rtype: int
    )");

    cls.def_property_readonly(
        "data",
        [](VASurfaceTensorWrapper& self) {
            Common::utils::raise_not_implemented();
        },
        R"(
        This property is not implemented.
    )");

    cls.def("__repr__", [](const VASurfaceTensorWrapper& self) {
        std::stringstream ss;

        ss << "shape" << self.tensor.get_shape() << " type: " << self.tensor.get_element_type();

        return "<" + Common::get_class_name(self) + ": " + ss.str() + ">";
    });
}
