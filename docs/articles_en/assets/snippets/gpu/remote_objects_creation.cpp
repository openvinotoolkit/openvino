// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>

#ifndef WIN32
#include <sys/mman.h>
#endif

#ifdef WIN32
#include <openvino/runtime/intel_gpu/ocl/dx.hpp>
#elif defined(ENABLE_LIBVA)
#include <openvino/runtime/intel_gpu/ocl/va.hpp>
#endif

void* allocate_usm_buffer(size_t size);
cl_mem allocate_cl_mem(size_t size);
cl_context get_cl_context();
cl_command_queue get_cl_queue();
cl::Buffer allocate_buffer(size_t size);
cl::Image2D allocate_image(size_t size);
ov::intel_gpu::ocl::os_handle_param get_shared_handle();


#ifdef WIN32
ID3D11Device* get_d3d_device();
#elif defined(ENABLE_LIBVA)
VADisplay get_va_display();
#endif

int main() {
    ov::Core core;
    auto model = core.read_model("model.xml");
    auto input = model->get_parameters().at(0);
    auto input_size = ov::shape_size(input->get_shape());

    auto compiled_model = core.compile_model(model, "GPU");
    auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();

    auto in_element_type = input->get_element_type();
    auto in_shape = input->get_shape();

{
    //! [wrap_usm_pointer]
    void* shared_buffer = allocate_usm_buffer(input_size);
    auto remote_tensor = gpu_context.create_tensor(in_element_type, in_shape, shared_buffer);
    //! [wrap_usm_pointer]
}

#ifndef WIN32
{
    //! [wrap_mmap_pointer]
    size_t shared_buffer_bytes = input_size * in_element_type.size();
    void* shared_buffer = mmap(nullptr,
                               shared_buffer_bytes,
                               PROT_READ | PROT_WRITE,
                               MAP_SHARED | MAP_ANONYMOUS,
                               -1,
                               0);
    if (shared_buffer != MAP_FAILED) {
        auto remote_tensor = gpu_context.create_tensor(in_element_type,
                                                       in_shape,
                                                       shared_buffer,
                                                       ov::intel_gpu::MemType::cpu_pointer);
        munmap(shared_buffer, shared_buffer_bytes);
    }
    //! [wrap_mmap_pointer]
}
#endif

{
    //! [wrap_cl_mem]
    cl_mem shared_buffer = allocate_cl_mem(input_size);
    auto remote_tensor = gpu_context.create_tensor(in_element_type, in_shape, shared_buffer);
    //! [wrap_cl_mem]
}

{
    //! [wrap_cl_buffer]
    cl::Buffer shared_buffer = allocate_buffer(input_size);
    auto remote_tensor = gpu_context.create_tensor(in_element_type, in_shape, shared_buffer);
    //! [wrap_cl_buffer]
}

{
    //! [wrap_cl_image]
    cl::Image2D shared_buffer = allocate_image(input_size);
    auto remote_tensor = gpu_context.create_tensor(in_element_type, in_shape, shared_buffer);
    //! [wrap_cl_image]
}

{
    //! [wrap_shared_handle]
    auto shared_handle = get_shared_handle();
    auto remote_tensor = gpu_context.create_tensor(in_element_type,
                                                   in_shape,
                                                   shared_handle,
                                                   ov::intel_gpu::MemType::SHARED_BUF);
    //! [wrap_shared_handle]
}

{
    //! [allocate_usm_device]
    auto remote_tensor = gpu_context.create_usm_device_tensor(in_element_type, in_shape);
    // Extract raw usm pointer from remote tensor
    void* usm_ptr = remote_tensor.get();
    //! [allocate_usm_device]
}

{
    //! [allocate_usm_host]
    ov::intel_gpu::ocl::USMTensor remote_tensor = gpu_context.create_usm_host_tensor(in_element_type, in_shape);
    // Extract raw usm pointer from remote tensor
    void* usm_ptr = remote_tensor.get();
    //! [allocate_usm_host]
}

{
    //! [allocate_cl_buffer]
    ov::RemoteTensor remote_tensor = gpu_context.create_tensor(in_element_type, in_shape);
    // Cast from base to derived class and extract ocl memory handle
    auto buffer_tensor = remote_tensor.as<ov::intel_gpu::ocl::ClBufferTensor>();
    cl_mem handle = buffer_tensor.get();
    //! [allocate_cl_buffer]
}

{
    size_t width = 1024;
    size_t height = 768;

    size_t y_plane_size = width*height;
    size_t uv_plane_size = width*height / 2;

    //! [wrap_nv12_surface]
    cl::Image2D y_plane_surface = allocate_image(y_plane_size);
    cl::Image2D uv_plane_surface = allocate_image(uv_plane_size);
    auto remote_tensor = gpu_context.create_tensor_nv12(y_plane_surface, uv_plane_surface);
    auto y_tensor = remote_tensor.first;
    auto uv_tensor = remote_tensor.second;
    //! [wrap_nv12_surface]
}

{
    //! [context_from_cl_context]
    cl_context ctx = get_cl_context();
    ov::intel_gpu::ocl::ClContext gpu_context(core, ctx);
    //! [context_from_cl_context]
}


{
    //! [context_from_cl_queue]
    cl_command_queue queue = get_cl_queue();
    ov::intel_gpu::ocl::ClContext gpu_context(core, queue);
    //! [context_from_cl_queue]
}

#ifdef WIN32
{
    //! [context_from_d3d_device]
    ID3D11Device* device = get_d3d_device();
    ov::intel_gpu::ocl::D3DContext gpu_context(core, device);
    //! [context_from_d3d_device]
}
#elif defined(ENABLE_LIBVA)
{
    //! [context_from_va_display]
    VADisplay display = get_va_display();
    ov::intel_gpu::ocl::VAContext gpu_context(core, display);
    //! [context_from_va_display]
}
#endif
{
    //! [default_context_from_core]
    auto gpu_context = core.get_default_context("GPU").as<ov::intel_gpu::ocl::ClContext>();
    // Extract ocl context handle from RemoteContext
    cl_context context_handle = gpu_context.get();
    //! [default_context_from_core]
}

{
    //! [default_context_from_model]
    auto gpu_context = compiled_model.get_context().as<ov::intel_gpu::ocl::ClContext>();
    // Extract ocl context handle from RemoteContext
    cl_context context_handle = gpu_context.get();
    //! [default_context_from_model]
}

    return 0;
}
