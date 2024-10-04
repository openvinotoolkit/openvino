// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>

int main() {
    ov::Core core;
    auto model = core.read_model("model.xml");
    auto input = model->get_parameters().at(0);

    auto compiled_model = core.compile_model(model, "NPU");
    auto npu_context = compiled_model.get_context().as<ov::intel_npu::level_zero::ZeroContext>();

    auto in_element_type = input->get_element_type();
    auto in_shape = input->get_shape();

    {
        //! [default_context_from_core]
        auto npu_context = core.get_default_context("NPU").as<ov::intel_npu::level_zero::ZeroContext>();
        // Extract raw level zero context handle from RemoteContext
        void* context_handle = npu_context.get();
        //! [default_context_from_core]
    }

    {
        //! [default_context_from_model]
        auto npu_context = compiled_model.get_context().as<ov::intel_npu::level_zero::ZeroContext>();
        // Extract raw level zero context handle from RemoteContext
        void* context_handle = npu_context.get();
        //! [default_context_from_model]
    }

    {
        //! [wrap_nt_handle]
        void* shared_buffer = nullptr;  // create the NT handle
        auto remote_tensor = npu_context.create_tensor(in_element_type, in_shape, shared_buffer);
        //! [wrap_nt_handle]
    }

    {
        //! [wrap_dmabuf_fd]
        int32_t fd_heap = 0;  // create the DMA-BUF System Heap file descriptor
        auto remote_tensor = npu_context.create_tensor(in_element_type, in_shape, fd_heap);
        //! [wrap_dmabuf_fd]
    }

    {
        //! [allocate_remote_level_zero_host]
        auto remote_tensor = npu_context.create_l0_host_tensor(in_element_type, in_shape);
        // Extract raw level zero pointer from remote tensor
        void* level_zero_ptr = remote_tensor.get();
        //! [allocate_remote_level_zero_host]
    }

    {
        //! [allocate_level_zero_host]
        auto tensor = npu_context.create_host_tensor(in_element_type, in_shape);
        // Extract raw level zero pointer from remote tensor
        void* level_zero_ptr = tensor.data();
        //! [allocate_level_zero_host]
    }

    return 0;
}
