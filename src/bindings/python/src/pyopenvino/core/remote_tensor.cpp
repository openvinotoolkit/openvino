// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/remote_tensor.hpp"

#include <pybind11/stl.h>

#include <openvino/core/any.hpp>
#include <openvino/runtime/remote_tensor.hpp>
#include <openvino/runtime/tensor.hpp>
#include <pyopenvino/core/tensor.hpp>

#ifdef PY_ENABLE_OPENCL
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#endif  // PY_ENABLE_OPENCL
#ifdef PY_ENABLE_LIBVA
#include <openvino/runtime/intel_gpu/ocl/va.hpp>
#endif  // PY_ENABLE_LIBVA

#include "common.hpp"
#include "pyopenvino/utils/utils.hpp"

namespace py = pybind11;

void regclass_RemoteTensor(py::module m) {
    // TODO: Should it inherit from Tensor? I can be tricky to manage memory.
    // Think about it -- especially for "data" related fields.
    py::class_<ov::RemoteTensor, ov::Tensor, std::shared_ptr<ov::RemoteTensor>> cls(m, "RemoteTensor");

    cls.def("get_device_name", &ov::RemoteTensor::get_device_name);

    cls.def(
        "get_params",
        [](ov::RemoteTensor& self) {
            // TODO: check if conversion is needed - RTMap issues...
            return self.get_params();
        });

    cls.def(
        "copy_to",
        [](ov::RemoteTensor& self, py::object& dst) {
            Common::utils::raise_not_implemented();
        });

    cls.def_property_readonly(
        "data",
        [](ov::RemoteTensor& self) {
            Common::utils::raise_not_implemented();
        });

    cls.def_property(
        "bytes_data",
        [](ov::RemoteTensor& self) {
            Common::utils::raise_not_implemented();
        },
        [](ov::RemoteTensor& self, py::object& other) {
            Common::utils::raise_not_implemented();
        });

    cls.def_property(
        "str_data",
        [](ov::RemoteTensor& self) {
            Common::utils::raise_not_implemented();
        },
        [](ov::RemoteTensor& self, py::object& other) {
            Common::utils::raise_not_implemented();
        });

    cls.def("__repr__", [](const ov::RemoteTensor& self) {
        std::stringstream ss;

        ss << "shape" << self.get_shape() << " type: " << self.get_element_type();

        return "<" + Common::get_class_name(self) + ": " + ss.str() + ">";
    });
}

#ifdef PY_ENABLE_OPENCL
void regclass_ClImage2DTensor(py::module m) {
    py::class_<ov::intel_gpu::ocl::ClImage2DTensor, ov::RemoteTensor, std::shared_ptr<ov::intel_gpu::ocl::ClImage2DTensor>> cls(m, "ClImage2DTensor");

    // TODO: Allow data in some other way...
    // Idea: create custom cl_mem handler and return as host memory!
    cls.def_property_readonly(
        "data",
        [](ov::RemoteTensor& self) {
            Common::utils::raise_not_implemented();
        });

    cls.def("__repr__", [](const ov::intel_gpu::ocl::ClImage2DTensor& self) {
        std::stringstream ss;

        ss << "shape" << self.get_shape() << " type: " << self.get_element_type();

        return "<" + Common::get_class_name(self) + ": " + ss.str() + ">";
    });
}
#endif  // PY_ENABLE_OPENCL

#ifdef PY_ENABLE_LIBVA
void regclass_VASurfaceTensor(py::module m) {
    py::class_<ov::intel_gpu::ocl::VASurfaceTensor, ov::intel_gpu::ocl::ClImage2DTensor, std::shared_ptr<ov::intel_gpu::ocl::VASurfaceTensor>> cls(m, "VASurfaceTensor");

    // Replace operator VASurfaceID() --> VASurfaceID --> VAGenericID --> unsigned int
    cls.def_property_readonly(
        "surface_id",
        [](ov::intel_gpu::ocl::VASurfaceTensor& self) {
            return self.get_params().at(ov::intel_gpu::dev_object_handle.name()).as<uint32_t>();
        });

    cls.def_property_readonly(
        "plane_id",
        [](ov::intel_gpu::ocl::VASurfaceTensor& self) {
            return self.plane();
        });

    // TODO: Allow data in some other way...
    // Idea: create custom cl_mem handler and return as host memory!
    cls.def_property_readonly(
        "data",
        [](ov::RemoteTensor& self) {
            Common::utils::raise_not_implemented();
        });

    cls.def("__repr__", [](const ov::intel_gpu::ocl::VASurfaceTensor& self) {
        std::stringstream ss;

        ss << "shape" << self.get_shape() << " type: " << self.get_element_type();

        return "<" + Common::get_class_name(self) + ": " + ss.str() + ">";
    });
}
#endif  // PY_ENABLE_LIBVA
