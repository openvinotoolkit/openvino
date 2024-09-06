// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/remote_tensor.hpp"

#include <pybind11/stl.h>

#include "common.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

void regclass_RemoteTensor(py::module m) {
    py::class_<RemoteTensorWrapper, std::shared_ptr<RemoteTensorWrapper>> cls(m, "RemoteTensor");

    cls.def(py::init([](RemoteTensorWrapper& tensor_wrapper, ov::Coordinate& begin, ov::Coordinate& end) {
                return RemoteTensorWrapper(ov::RemoteTensor(tensor_wrapper.tensor, begin, end));
            }),
            py::arg("remote_tensor"),
            py::arg("begin"),
            py::arg("end"),
            R"(
        Constructs a RoiRemoteTensor object using a specified range of coordinates on an existing RemoteTensor.

        :param remote_tensor: The RemoteTensor object on which the RoiRemoteTensor will be based.
        :type remote_tensor: openvino.RemoteTensor
        :param begin: The starting coordinates for the tensor bound.
        :type begin: openvino.runtime.Coordinate
        :param end: The ending coordinates for the tensor bound.
        :type end: openvino.runtime.Coordinate
        )");

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
        [](RemoteTensorWrapper& self, RemoteTensorWrapper& dst) {
            self.tensor.copy_to(dst.tensor);
        },
        py::arg("target_tensor"),
        R"(
        Copy tensor's data to a destination remote tensor. The destination tensor should have the same element type.
        In case of RoiTensor, the destination tensor should also have the same shape.

        :param target_tensor: The destination remote tensor to which the data will be copied.
        :type target_tensor: openvino.RemoteTensor
    )");

    cls.def(
        "copy_to",
        [](RemoteTensorWrapper& self, ov::Tensor& dst) {
            self.tensor.copy_to(dst);
        },
        py::arg("target_tensor"),
        R"(
        Copy tensor's data to a destination tensor. The destination tensor should have the same element type.
        In case of RoiTensor, the destination tensor should also have the same shape.

        :param target_tensor: The destination tensor to which the data will be copied.
        :type target_tensor: openvino.Tensor
    )");

    cls.def(
        "copy_from",
        [](RemoteTensorWrapper& self, RemoteTensorWrapper& src) {
            self.tensor.copy_from(src.tensor);
        },
        py::arg("source_tensor"),
        R"(
        Copy source remote tensor's data to this tensor. Tensors should have the same element type.
        In case of RoiTensor, tensors should also have the same shape.

        :param source_tensor: The source remote tensor from which the data will be copied.
        :type source_tensor: openvino.RemoteTensor
    )");

    cls.def(
        "copy_from",
        [](RemoteTensorWrapper& self, ov::Tensor& src) {
            self.tensor.copy_from(src);
        },
        py::arg("source_tensor"),
        R"(
        Copy source tensor's data to this tensor. Tensors should have the same element type and shape.
        In case of RoiTensor, tensors should also have the same shape.

        :param source_tensor: The source tensor from which the data will be copied.
        :type source_tensor: openvino.Tensor
    )");

    cls.def(
        "get_shape",
        [](RemoteTensorWrapper& self) {
            return self.tensor.get_shape();
        },
        R"(
        Gets Tensor's shape.

        :rtype: openvino.Shape
    )");

    cls.def(
        "get_byte_size",
        [](RemoteTensorWrapper& self) {
            return self.tensor.get_byte_size();
        },
        R"(
        Gets Tensor's size in bytes.

        :rtype: int
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
