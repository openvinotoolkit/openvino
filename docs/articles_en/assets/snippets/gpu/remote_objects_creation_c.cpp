// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/c/openvino.h>
#include <openvino/c/gpu/gpu_plugin_properties.h>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>

#ifdef WIN32
typedef void* ID3D11Device;
#elif defined(ENABLE_LIBVA)
#include <openvino/runtime/intel_gpu/ocl/va.hpp>
#endif

void* allocate_usm_buffer(size_t size);
cl_mem allocate_cl_mem(size_t size);
cl_context get_cl_context();
cl_command_queue get_cl_queue();
cl::Buffer allocate_buffer(size_t size);
cl::Image2D allocate_image(size_t size);

#ifdef WIN32
ID3D11Device* get_d3d_device();
#elif defined(ENABLE_LIBVA)
VADisplay get_va_display();
#endif

int main() {
    ov_core_t* core = NULL;
    ov_model_t* model = NULL;
    ov_compiled_model_t* compiled_model = NULL;
    ov_remote_context_t* gpu_context = NULL;
    ov_tensor_t* remote_tensor = NULL;

    ov_output_const_port* input_port = NULL;
    char* in_tensor_name = NULL;
    char* out_tensor_name = NULL;
    ov_shape_t input_shape;
    ov_element_type_e input_type;

    ov_core_create(&core);
    ov_core_read_model(core, "model.xml", "model.bin", &model);

    ov_model_const_input(model, &input_port);
    ov_port_get_any_name(input_port, &in_tensor_name);
    ov_const_port_get_shape(input_port, &input_shape);
    ov_port_get_element_type(input_port, &input_type);
    size_t input_size = 1;
    for (auto i = 0; i < input_shape.rank; i++)
        input_size *= input_shape.dims[i];

    ov_core_compile_model(core, model, "GPU", 0, &compiled_model);
    ov_compiled_model_get_context(compiled_model, &gpu_context);

{
    //! [wrap_usm_pointer]
    void* shared_buffer = allocate_usm_buffer(input_size);
    ov_remote_context_create_tensor(gpu_context,
                                    input_type,
                                    input_shape,
                                    4,
                                    &remote_tensor,
                                    ov_property_key_intel_gpu_shared_mem_type,
                                    "USM_USER_BUFFER",
                                    ov_property_key_intel_gpu_mem_handle,
                                    shared_buffer);
    //! [wrap_usm_pointer]
}

{
    //! [wrap_cl_mem]
    cl_mem shared_buffer = allocate_cl_mem(input_size);
    ov_remote_context_create_tensor(gpu_context,
                                    input_type,
                                    input_shape,
                                    4,
                                    &remote_tensor,
                                    ov_property_key_intel_gpu_shared_mem_type,
                                    "OCL_BUFFER",
                                    ov_property_key_intel_gpu_mem_handle,
                                    shared_buffer);
    //! [wrap_cl_mem]
}

{
    //! [wrap_cl_buffer]
    cl::Buffer shared_buffer = allocate_buffer(input_size);
    ov_remote_context_create_tensor(gpu_context,
                                    input_type,
                                    input_shape,
                                    4,
                                    &remote_tensor,
                                    ov_property_key_intel_gpu_shared_mem_type,
                                    "OCL_BUFFER",
                                    ov_property_key_intel_gpu_mem_handle,
                                    shared_buffer.get());
    //! [wrap_cl_buffer]
}

{
    //! [wrap_cl_image]
    cl::Image2D shared_buffer = allocate_image(input_size);
    ov_remote_context_create_tensor(gpu_context,
                                    input_type,
                                    input_shape,
                                    4,
                                    &remote_tensor,
                                    ov_property_key_intel_gpu_shared_mem_type,
                                    "OCL_IMAGE2D",
                                    ov_property_key_intel_gpu_mem_handle,
                                    shared_buffer.get());
    //! [wrap_cl_image]
}

{
    //! [allocate_usm_device]
    ov_remote_context_create_tensor(gpu_context,
                                    input_type,
                                    input_shape,
                                    2,
                                    &remote_tensor,
                                    ov_property_key_intel_gpu_shared_mem_type,
                                    "USM_USER_BUFFER");
    // Extract raw usm pointer from remote tensor
    void* usm_ptr = NULL;
    ov_tensor_data(remote_tensor, &usm_ptr);
    //! [allocate_usm_device]
}

{
    //! [allocate_usm_host]
    ov_remote_context_create_tensor(gpu_context,
                                    input_type,
                                    input_shape,
                                    2,
                                    &remote_tensor,
                                    ov_property_key_intel_gpu_shared_mem_type,
                                    "USM_HOST_BUFFER");
    // Extract raw usm pointer from remote tensor
    void* usm_ptr = NULL;
    ov_tensor_data(remote_tensor, &usm_ptr);
    //! [allocate_usm_host]
}

{
    int64_t width = 1024;
    int64_t height = 768;

    int64_t y_plane_size = width * height;
    int64_t uv_plane_size = width * height / 2;

    ov_shape_t shape_y = {0, NULL};
    int64_t dims_y[4] = {1, 1, height, width};
    ov_shape_t shape_uv = {0, NULL};
    int64_t dims_uv[4] = {1, 2, height / 2, width / 2};
    ov_tensor_t* remote_tensor_y = NULL;
    ov_tensor_t* remote_tensor_uv = NULL;

    ov_shape_create(4, dims_y, &shape_y);
    ov_shape_create(4, dims_uv, &shape_uv);

    //! [create_nv12_surface]
    cl::Image2D y_plane_surface = allocate_image(y_plane_size);
    cl::Image2D uv_plane_surface = allocate_image(uv_plane_size);

    ov_remote_context_create_tensor(gpu_context,
                                    input_type,
                                    shape_y,
                                    4,
                                    &remote_tensor_y,
                                    ov_property_key_intel_gpu_shared_mem_type,
                                    "OCL_IMAGE2D",
                                    ov_property_key_intel_gpu_mem_handle,
                                    y_plane_surface.get());

    ov_remote_context_create_tensor(gpu_context,
                                    input_type,
                                    shape_uv,
                                    4,
                                    &remote_tensor_uv,
                                    ov_property_key_intel_gpu_shared_mem_type,
                                    "OCL_IMAGE2D",
                                    ov_property_key_intel_gpu_mem_handle,
                                    uv_plane_surface.get());

    ov_tensor_free(remote_tensor_y);
    ov_tensor_free(remote_tensor_uv);
    ov_shape_free(&shape_y);
    ov_shape_free(&shape_uv);
    //! [create_nv12_surface]
}

{
    //! [context_from_cl_context]
    cl_context cl_context = get_cl_context();
    ov_core_create_context(core,
                           "GPU",
                           4,
                           &gpu_context,
                           ov_property_key_intel_gpu_context_type,
                           "OCL",
                           ov_property_key_intel_gpu_ocl_context,
                           cl_context);
    //! [context_from_cl_context]
}

{
    //! [context_from_cl_queue]
    cl_command_queue cl_queue = get_cl_queue();
    cl_context cl_context = get_cl_context();
    ov_core_create_context(core,
                           "GPU",
                           6,
                           &gpu_context,
                           ov_property_key_intel_gpu_context_type,
                           "OCL",
                           ov_property_key_intel_gpu_ocl_context,
                           cl_context,
                           ov_property_key_intel_gpu_ocl_queue,
                           cl_queue);
    //! [context_from_cl_queue]
}

#ifdef WIN32
{
    //! [context_from_d3d_device]
    ID3D11Device* device = get_d3d_device();
    ov_core_create_context(core,
                           "GPU",
                           4,
                           &gpu_context,
                           ov_property_key_intel_gpu_context_type,
                           "VA_SHARED",
                           ov_property_key_intel_gpu_va_device,
                           device);
    //! [context_from_d3d_device]
}
#elif defined(ENABLE_LIBVA)
{
    //! [context_from_va_display]
    VADisplay display = get_va_display();
    ov_core_create_context(core,
                           "GPU",
                           4,
                           &gpu_context,
                           ov_property_key_intel_gpu_context_type,
                           "VA_SHARED",
                           ov_property_key_intel_gpu_va_device,
                           display);
    //! [context_from_va_display]
}
#endif
{
    //! [default_context_from_core]
    ov_core_get_default_context(core, "GPU", &gpu_context);
    // Extract ocl context handle from RemoteContext
    size_t size = 0;
    char* params = nullptr;
    // params is format like: "CONTEXT_TYPE OCL OCL_CONTEXT 0x5583b2ec7b40 OCL_QUEUE 0x5583b2e98ff0"
    // You need parse it.
    ov_remote_context_get_params(gpu_context, &size, &params);
    //! [default_context_from_core]
}

{
    //! [default_context_from_model]
    ov_compiled_model_get_context(compiled_model, &gpu_context);
    // Extract ocl context handle from RemoteContext
    size_t size = 0;
    char* params = nullptr;
    // params is format like: "CONTEXT_TYPE OCL OCL_CONTEXT 0x5583b2ec7b40 OCL_QUEUE 0x5583b2e98ff0"
    // You need parse it.
    ov_remote_context_get_params(gpu_context, &size, &params);
    //! [default_context_from_model]
}

ov_compiled_model_free(compiled_model);
ov_model_free(model);
ov_remote_context_free(gpu_context);
ov_core_free(core);

return 0;
}
