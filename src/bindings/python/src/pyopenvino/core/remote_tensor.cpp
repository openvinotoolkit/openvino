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

    cls.def("get_device_name", [](RemoteTensorWrapper& self) {
        return self.tensor.get_device_name();
    });

    cls.def("get_params", [](RemoteTensorWrapper& self) {
        return self.tensor.get_params();
    });

    cls.def("copy_to", [](RemoteTensorWrapper& self, py::object& dst) {
        Common::utils::raise_not_implemented();
    });

    cls.def_property_readonly("data", [](RemoteTensorWrapper& self) {
        Common::utils::raise_not_implemented();
    });

    cls.def_property(
        "bytes_data",
        [](RemoteTensorWrapper& self) {
            Common::utils::raise_not_implemented();
        },
        [](RemoteTensorWrapper& self, py::object& other) {
            Common::utils::raise_not_implemented();
        });

    cls.def_property(
        "str_data",
        [](RemoteTensorWrapper& self) {
            Common::utils::raise_not_implemented();
        },
        [](RemoteTensorWrapper& self, py::object& other) {
            Common::utils::raise_not_implemented();
        });

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

    cls.def_property_readonly("surface_id", [](VASurfaceTensorWrapper& self) {
        return self.surface_id();
    });

    cls.def_property_readonly("plane_id", [](VASurfaceTensorWrapper& self) {
        return self.plane_id();
    });

    cls.def_property_readonly("data", [](VASurfaceTensorWrapper& self) {
        Common::utils::raise_not_implemented();
    });

    cls.def("__repr__", [](const VASurfaceTensorWrapper& self) {
        std::stringstream ss;

        ss << "shape" << self.tensor.get_shape() << " type: " << self.tensor.get_element_type();

        return "<" + Common::get_class_name(self) + ": " + ss.str() + ">";
    });
}
