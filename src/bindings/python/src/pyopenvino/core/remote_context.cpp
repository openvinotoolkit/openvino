// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/remote_context.hpp"

#include <pybind11/stl_bind.h>
#include <va/va.h>

#include "openvino/runtime/intel_gpu/remote_properties.hpp"
#include "openvino/runtime/remote_context.hpp"

namespace py = pybind11;

void regclass_RemoteContext(py::module m) {
    py::class_<ov::RemoteContext, std::shared_ptr<ov::RemoteContext>> cls(m, "RemoteContext");
    cls.doc() = "openvino.runtime.RemoteContext holding state of remote device.";

    /*
    cls.def(py::init([](py::array& array, bool shared_memory) {
                return Common::object_from_data<ov::Tensor>(array, shared_memory);
            }),
            py::arg("array"),
            py::arg("shared_memory") = false,
            R"(
                Tensor's special constructor.

                :param array: Array to create tensor from.
                :type array: numpy.array
                :param shared_memory: If `True`, this Tensor memory is being shared with a host,
                                      that means the responsibility of keeping host memory is
                                      on the side of a user. Any action performed on the host
                                      memory is reflected on this Tensor's memory!
                                      If `False`, data is being copied to this Tensor.
                                      Requires data to be C_CONTIGUOUS if `True`.
                :type shared_memory: bool
            )");*/

    cls.def(
        "create_tensor_nv12",
        [](ov::RemoteContext& self, const size_t height, const size_t width, const VASurfaceID nv12_surf) {
            ov::AnyMap tensor_params = {
                {ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::VA_SURFACE},
                {ov::intel_gpu::dev_object_handle.name(), nv12_surf},
                {ov::intel_gpu::va_plane.name(), uint32_t(0)}};
            auto y_tensor = self.create_tensor(ov::element::u8, {1, height, width, 1}, tensor_params);
            tensor_params[ov::intel_gpu::va_plane.name()] = uint32_t(1);
            auto uv_tensor = self.create_tensor(ov::element::u8, {1, height / 2, width / 2, 2}, tensor_params);
            return py::make_tuple(y_tensor, uv_tensor);
        },
        R"(
            Create pair of tensors (for Y and UV plane) from NV12 vaapi surface.
        )");
}
