// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/intel_gpu/ocl/ocl_wrapper.hpp"
#include "ov_test.hpp"

class ov_remote_context_ocl : public ov_capi_test_base {
protected:
    void SetUp() override {
        core = nullptr;
        model = nullptr;
        context = nullptr;
        compiled_model = nullptr;
        infer_request = nullptr;
        remote_tensor = nullptr;
        out_tensor_name = nullptr;
        in_tensor_name = nullptr;
        ov_capi_test_base::SetUp();

        OV_EXPECT_OK(ov_core_create(&core));
        EXPECT_NE(nullptr, core);

        // Check gpu plugin and hardware available.
        bool gpu_device_available = false;
        char* info = nullptr;
        const char* key = ov_property_key_available_devices;
        if (ov_core_get_property(core, "GPU", key, &info) == ov_status_e::OK) {
            if (strlen(info) > 0) {
                gpu_device_available = true;
            }
        }
        ov_free(info);
        if (!gpu_device_available) {
            GTEST_SKIP();
        }

        OV_EXPECT_OK(ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model));
        EXPECT_NE(nullptr, model);

        const unsigned int refVendorID = 0x8086;
        cl_uint n = 0;
        cl_int err = clGetPlatformIDs(0, NULL, &n);
        EXPECT_EQ(err, 0);

        // Get platform list
        std::vector<cl_platform_id> platform_ids(n);
        err = clGetPlatformIDs(n, platform_ids.data(), NULL);

        for (auto const& id : platform_ids) {
            cl::Platform platform = cl::Platform(id);
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
            for (auto const& d : devices) {
                if (refVendorID == d.getInfo<CL_DEVICE_VENDOR_ID>()) {
                    cl_device = d;
                    cl_context = cl::Context(cl_device);
                    cl_platform = id;
                    break;
                }
            }
        }
        cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
        cl_queue = cl::CommandQueue(cl_context, cl_device, props);
        cl::size_type size = 224 * 224 * 3;
        cl_buffer = cl::Buffer(cl_context, CL_MEM_READ_WRITE, size, NULL, &err);
        EXPECT_EQ(err, 0);
    }

    void TearDown() override {
        ov_infer_request_free(infer_request);
        ov_compiled_model_free(compiled_model);
        ov_model_free(model);
        ov_tensor_free(remote_tensor);
        ov_free(out_tensor_name);
        ov_free(in_tensor_name);
        ov_remote_context_free(context);
        ov_core_free(core);
        ov_capi_test_base::TearDown();
    }

public:
    void get_model_tensor_info(ov_model_t* model,
                               bool input,
                               char** name,
                               ov_shape_t* shape,
                               ov_element_type_e* type,
                               size_t* mem_size) {
        ov_output_const_port* port = nullptr;
        if (input) {
            OV_EXPECT_OK(ov_model_const_input(model, &port));
        } else {
            OV_EXPECT_OK(ov_model_const_output(model, &port));
        }
        EXPECT_NE(nullptr, port);

        OV_EXPECT_OK(ov_port_get_any_name(port, name));
        EXPECT_NE(nullptr, *name);

        OV_ASSERT_OK(ov_const_port_get_shape(port, shape));
        OV_EXPECT_OK(ov_port_get_element_type(port, type));

        int64_t size = 1;
        for (int64_t i = 0; i < shape->rank; i++) {
            size *= shape->dims[i];
        }

        switch (*type) {
        case ov_element_type_e::BF16:
        case ov_element_type_e::F16:
        case ov_element_type_e::I16:
        case ov_element_type_e::U16:
            size *= 2;
            break;
        case ov_element_type_e::F32:
        case ov_element_type_e::U32:
        case ov_element_type_e::I32:
            size *= 4;
            break;
        case ov_element_type_e::I8:
        case ov_element_type_e::U8:
            size *= 1;
        case ov_element_type_e::U64:
        case ov_element_type_e::I64:
            size *= 8;
            break;
        default:
            break;
        }
        *mem_size = size;
    }
    ov_core_t* core;
    ov_model_t* model;
    ov_compiled_model_t* compiled_model;
    ov_infer_request_t* infer_request;
    ov_remote_context_t* context;
    ov_tensor_t* remote_tensor;
    char* in_tensor_name;
    char* out_tensor_name;
    cl::Context cl_context;
    cl::Device cl_device;
    cl::CommandQueue cl_queue;
    cl_platform_id cl_platform;
    cl::Buffer cl_buffer;
};

INSTANTIATE_TEST_SUITE_P(intel_gpu, ov_remote_context_ocl, ::testing::Values("GPU"));

TEST_P(ov_remote_context_ocl, get_default_context) {
    OV_EXPECT_OK(ov_core_get_default_context(core, "GPU", &context));
    EXPECT_NE(nullptr, context);
}

TEST_P(ov_remote_context_ocl, create_ocl_context_by_command_queue) {
    const char* context_type = "OCL";
    OV_EXPECT_OK(ov_core_create_context(core,
                                        "GPU",
                                        6,
                                        &context,
                                        ov_property_key_intel_gpu_context_type,
                                        context_type,
                                        ov_property_key_intel_gpu_ocl_context,
                                        cl_context.get(),
                                        ov_property_key_intel_gpu_ocl_queue,
                                        cl_queue.get()));
    EXPECT_NE(nullptr, context);

    size_t size = 0;
    char* params = nullptr;
    OV_EXPECT_OK(ov_remote_context_get_params(context, &size, &params));
    EXPECT_NE(nullptr, params);
    EXPECT_NE(0, size);
    ov_free(params);
}

TEST_P(ov_remote_context_ocl, create_ocl_context_by_device_id) {
    const char* context_type = "OCL";
    const char* device_id_str = "0";
    OV_EXPECT_OK(ov_core_create_context(core,
                                        "GPU",
                                        6,
                                        &context,
                                        ov_property_key_intel_gpu_context_type,
                                        context_type,
                                        ov_property_key_intel_gpu_ocl_context,
                                        cl_context.get(),
                                        ov_property_key_intel_gpu_ocl_context_device_id,
                                        device_id_str));
    EXPECT_NE(nullptr, context);

    size_t size = 0;
    char* params = nullptr;
    OV_EXPECT_OK(ov_remote_context_get_params(context, &size, &params));
    EXPECT_NE(nullptr, params);
    EXPECT_NE(0, size);
    ov_free(params);
}

TEST_P(ov_remote_context_ocl, get_core_default_context) {
    OV_EXPECT_OK(ov_core_get_default_context(core, "GPU", &context));
    EXPECT_NE(nullptr, context);

    size_t size = 0;
    char* params = nullptr;
    OV_EXPECT_OK(ov_remote_context_get_params(context, &size, &params));
    EXPECT_NE(nullptr, params);
    EXPECT_NE(0, size);
    ov_free(params);
}

TEST_P(ov_remote_context_ocl, create_ocl_context_get_device_name) {
    const char* context_type = "OCL";
    OV_EXPECT_OK(ov_core_create_context(core,
                                        "GPU",
                                        6,
                                        &context,
                                        ov_property_key_intel_gpu_context_type,
                                        context_type,
                                        ov_property_key_intel_gpu_ocl_context,
                                        cl_context.get(),
                                        ov_property_key_intel_gpu_ocl_queue,
                                        cl_queue.get()));
    EXPECT_NE(nullptr, context);

    char* device_name = nullptr;
    OV_EXPECT_OK(ov_remote_context_get_device_name(context, &device_name));
    EXPECT_NE(nullptr, device_name);
    ov_free(device_name);
}

TEST_P(ov_remote_context_ocl, compile_mode_with_remote_context) {
    const char* context_type = "OCL";
    OV_EXPECT_OK(ov_core_create_context(core,
                                        "GPU",
                                        6,
                                        &context,
                                        ov_property_key_intel_gpu_context_type,
                                        context_type,
                                        ov_property_key_intel_gpu_ocl_context,
                                        cl_context.get(),
                                        ov_property_key_intel_gpu_ocl_queue,
                                        cl_queue.get()));
    EXPECT_NE(nullptr, context);

    size_t param_size = 0;
    char* params = nullptr;
    OV_EXPECT_OK(ov_remote_context_get_params(context, &param_size, &params));
    EXPECT_NE(nullptr, params);
    EXPECT_NE(0, param_size);
    ov_free(params);

    OV_EXPECT_OK(ov_core_compile_model_with_context(core, model, context, 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);
}

TEST_P(ov_remote_context_ocl, create_remote_tensor_from_ocl_buffer) {
    const char* context_type = "OCL";
    OV_EXPECT_OK(ov_core_create_context(core,
                                        "GPU",
                                        6,
                                        &context,
                                        ov_property_key_intel_gpu_context_type,
                                        context_type,
                                        ov_property_key_intel_gpu_ocl_context,
                                        cl_context.get(),
                                        ov_property_key_intel_gpu_ocl_queue,
                                        cl_queue.get()));
    EXPECT_NE(nullptr, context);

    ov_shape_t shape = {0, nullptr};
    int64_t dims[4] = {1, 224, 224, 3};
    OV_EXPECT_OK(ov_shape_create(4, dims, &shape));
    OV_EXPECT_OK(ov_remote_context_create_tensor(context,
                                                 ov_element_type_e::U8,
                                                 shape,
                                                 4,
                                                 &remote_tensor,
                                                 ov_property_key_intel_gpu_shared_mem_type,
                                                 "OCL_BUFFER",
                                                 ov_property_key_intel_gpu_mem_handle,
                                                 cl_buffer.get()));
    EXPECT_NE(nullptr, remote_tensor);

    size_t param_size = 0;
    char* params = nullptr;
    OV_EXPECT_OK(ov_remote_tensor_get_params(remote_tensor, &param_size, &params));
    EXPECT_NE(nullptr, params);
    EXPECT_NE(0, param_size);
    ov_free(params);
    ov_shape_free(&shape);

    char* device_name = nullptr;
    OV_EXPECT_OK(ov_remote_tensor_get_device_name(remote_tensor, &device_name));
    EXPECT_NE(nullptr, device_name);
    ov_free(device_name);
}

TEST_P(ov_remote_context_ocl, create_remote_tensor_from_ocl_image2D) {
    const char* context_type = "OCL";
    OV_EXPECT_OK(ov_core_create_context(core,
                                        "GPU",
                                        6,
                                        &context,
                                        ov_property_key_intel_gpu_context_type,
                                        context_type,
                                        ov_property_key_intel_gpu_ocl_context,
                                        cl_context.get(),
                                        ov_property_key_intel_gpu_ocl_queue,
                                        cl_queue.get()));
    EXPECT_NE(nullptr, context);

    const int height = 480;
    const int width = 640;
    ov_shape_t shape = {0, nullptr};
    int64_t dims[4] = {1, 1, width, height};

    cl_int err;
    cl_image_format image_format;
    cl_image_desc image_desc = {0};
    image_format.image_channel_order = CL_R;
    image_format.image_channel_data_type = CL_UNORM_INT8;
    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = width;
    image_desc.image_height = height;
    cl_mem mem = clCreateImage(cl_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
    ASSERT_EQ(err, 0);
    cl::Image2D image2D = cl::Image2D(mem);

    OV_EXPECT_OK(ov_shape_create(4, dims, &shape));
    OV_EXPECT_OK(ov_remote_context_create_tensor(context,
                                                 ov_element_type_e::U8,
                                                 shape,
                                                 4,
                                                 &remote_tensor,
                                                 ov_property_key_intel_gpu_shared_mem_type,
                                                 "OCL_IMAGE2D",
                                                 ov_property_key_intel_gpu_mem_handle,
                                                 image2D.get()));
    EXPECT_NE(nullptr, remote_tensor);

    size_t param_size = 0;
    char* params = nullptr;
    OV_EXPECT_OK(ov_remote_tensor_get_params(remote_tensor, &param_size, &params));
    EXPECT_NE(nullptr, params);
    EXPECT_NE(0, param_size);
    ov_free(params);
}

TEST_P(ov_remote_context_ocl, create_remote_tensor_nv12_from_ocl_image2D) {
    const char* context_type = "OCL";
    OV_EXPECT_OK(ov_core_create_context(core,
                                        "GPU",
                                        6,
                                        &context,
                                        ov_property_key_intel_gpu_context_type,
                                        context_type,
                                        ov_property_key_intel_gpu_ocl_context,
                                        cl_context.get(),
                                        ov_property_key_intel_gpu_ocl_queue,
                                        cl_queue.get()));
    EXPECT_NE(nullptr, context);

    const int height = 480;
    const int width = 640;
    ov_shape_t shape_y = {0, nullptr};
    int64_t dims_y[4] = {1, height, width, 1};
    ov_shape_t shape_uv = {0, nullptr};
    int64_t dims_uv[4] = {1, height / 2, width / 2, 2};

    cl_int err;
    cl_image_format image_format;
    cl_image_desc image_desc = {0};
    image_format.image_channel_order = CL_R;
    image_format.image_channel_data_type = CL_UNORM_INT8;
    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = width;
    image_desc.image_height = height;
    cl_mem mem_y = clCreateImage(cl_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
    ASSERT_EQ(err, 0);

    image_format.image_channel_order = CL_RG;
    image_desc.image_width = width / 2;
    image_desc.image_height = height / 2;
    cl_mem mem_uv = clCreateImage(cl_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
    ASSERT_EQ(err, 0);

    cl::Image2D image_y = cl::Image2D(mem_y);
    cl::Image2D image_uv = cl::Image2D(mem_uv);

    ov_tensor_t* remote_tensor_y = nullptr;
    ov_tensor_t* remote_tensor_uv = nullptr;
    OV_EXPECT_OK(ov_shape_create(4, dims_y, &shape_y));
    OV_EXPECT_OK(ov_remote_context_create_tensor(context,
                                                 ov_element_type_e::U8,
                                                 shape_y,
                                                 4,
                                                 &remote_tensor_y,
                                                 ov_property_key_intel_gpu_shared_mem_type,
                                                 "OCL_IMAGE2D",
                                                 ov_property_key_intel_gpu_mem_handle,
                                                 image_y.get()));
    EXPECT_NE(nullptr, remote_tensor_y);

    OV_EXPECT_OK(ov_shape_create(4, dims_uv, &shape_uv));
    OV_EXPECT_OK(ov_remote_context_create_tensor(context,
                                                 ov_element_type_e::U8,
                                                 shape_uv,
                                                 4,
                                                 &remote_tensor_uv,
                                                 ov_property_key_intel_gpu_shared_mem_type,
                                                 "OCL_IMAGE2D",
                                                 ov_property_key_intel_gpu_mem_handle,
                                                 image_uv.get()));
    EXPECT_NE(nullptr, remote_tensor_uv);

    ov_tensor_free(remote_tensor_y);
    ov_tensor_free(remote_tensor_uv);
    ov_shape_free(&shape_y);
    ov_shape_free(&shape_uv);
}

TEST_P(ov_remote_context_ocl, create_remote_tensor_inference) {
    const char* context_type = "OCL";
    OV_EXPECT_OK(ov_core_create_context(core,
                                        "GPU",
                                        6,
                                        &context,
                                        ov_property_key_intel_gpu_context_type,
                                        context_type,
                                        ov_property_key_intel_gpu_ocl_context,
                                        cl_context.get(),
                                        ov_property_key_intel_gpu_ocl_queue,
                                        cl_queue.get()));
    EXPECT_NE(nullptr, context);

    ov_shape_t shape = {0, nullptr};
    ov_element_type_e tensor_type;
    size_t mem_size = 0;
    get_model_tensor_info(model, true, &in_tensor_name, &shape, &tensor_type, &mem_size);

    cl_int err = 0;
    cl::Buffer cl_buffer_tmp =
        cl::Buffer(cl_context, CL_MEM_READ_WRITE, static_cast<cl::size_type>(mem_size), NULL, &err);
    OV_EXPECT_OK(ov_remote_context_create_tensor(context,
                                                 tensor_type,
                                                 shape,
                                                 4,
                                                 &remote_tensor,
                                                 ov_property_key_intel_gpu_shared_mem_type,
                                                 "OCL_BUFFER",
                                                 ov_property_key_intel_gpu_mem_handle,
                                                 cl_buffer_tmp.get()));
    EXPECT_NE(nullptr, remote_tensor);
    ov_shape_free(&shape);

    OV_EXPECT_OK(ov_core_compile_model_with_context(core, model, context, 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    OV_EXPECT_OK(ov_infer_request_set_tensor(infer_request, in_tensor_name, remote_tensor));
    OV_ASSERT_OK(ov_infer_request_infer(infer_request));
}

TEST_P(ov_remote_context_ocl, create_host_tensor) {
    const char* context_type = "OCL";
    OV_EXPECT_OK(ov_core_create_context(core,
                                        "GPU",
                                        6,
                                        &context,
                                        ov_property_key_intel_gpu_context_type,
                                        context_type,
                                        ov_property_key_intel_gpu_ocl_context,
                                        cl_context.get(),
                                        ov_property_key_intel_gpu_ocl_queue,
                                        cl_queue.get()));
    EXPECT_NE(nullptr, context);

    ov_tensor_t* host_tensor = nullptr;
    ov_shape_t shape = {0, nullptr};
    int64_t dims[4] = {1, 224, 224, 3};
    OV_EXPECT_OK(ov_shape_create(4, dims, &shape));
    OV_EXPECT_OK(ov_remote_context_create_host_tensor(context, ov_element_type_e::U8, shape, &host_tensor));
    EXPECT_NE(nullptr, host_tensor);

    size_t tensor_size = 0;
    OV_EXPECT_OK(ov_tensor_get_size(host_tensor, &tensor_size));
    EXPECT_EQ(tensor_size, 224 * 224 * 3);

    ov_shape_free(&shape);
    ov_tensor_free(host_tensor);
}

TEST_P(ov_remote_context_ocl, remote_tensor_nv12_inference) {
    const int height = 480;
    const int width = 640;
    ov_preprocess_prepostprocessor_t* preprocess = nullptr;
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_create(model, &preprocess));
    EXPECT_NE(nullptr, preprocess);

    ov_preprocess_input_info_t* preprocess_input_info = nullptr;
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_get_input_info(preprocess, &preprocess_input_info));
    EXPECT_NE(nullptr, preprocess_input_info);

    ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info = nullptr;
    OV_EXPECT_OK(ov_preprocess_input_info_get_tensor_info(preprocess_input_info, &preprocess_input_tensor_info));
    EXPECT_NE(nullptr, preprocess_input_tensor_info);

    OV_EXPECT_OK(ov_preprocess_input_tensor_info_set_element_type(preprocess_input_tensor_info, ov_element_type_e::U8));
    OV_EXPECT_OK(ov_preprocess_input_tensor_info_set_color_format_with_subname(preprocess_input_tensor_info,
                                                                               ov_color_format_e::NV12_TWO_PLANES,
                                                                               2,
                                                                               "y",
                                                                               "uv"));
    OV_EXPECT_OK(ov_preprocess_input_tensor_info_set_memory_type(preprocess_input_tensor_info, "GPU_SURFACE"));
    OV_EXPECT_OK(ov_preprocess_input_tensor_info_set_spatial_static_shape(preprocess_input_tensor_info, height, width));

    ov_preprocess_preprocess_steps_t* preprocess_input_steps = nullptr;
    OV_EXPECT_OK(ov_preprocess_input_info_get_preprocess_steps(preprocess_input_info, &preprocess_input_steps));
    EXPECT_NE(nullptr, preprocess_input_steps);
    OV_EXPECT_OK(ov_preprocess_preprocess_steps_convert_color(preprocess_input_steps, ov_color_format_e::BGR));
    OV_EXPECT_OK(ov_preprocess_preprocess_steps_resize(preprocess_input_steps, RESIZE_LINEAR));

    ov_preprocess_input_model_info_t* preprocess_input_model_info = nullptr;
    OV_EXPECT_OK(ov_preprocess_input_info_get_model_info(preprocess_input_info, &preprocess_input_model_info));
    EXPECT_NE(nullptr, preprocess_input_model_info);

    ov_layout_t* layout = nullptr;
    OV_EXPECT_OK(ov_layout_create("NCHW", &layout));
    OV_EXPECT_OK(ov_preprocess_input_model_info_set_layout(preprocess_input_model_info, layout));

    ov_model_t* new_model = nullptr;
    OV_EXPECT_OK(ov_preprocess_prepostprocessor_build(preprocess, &new_model));
    EXPECT_NE(nullptr, new_model);

    ov_output_const_port_t* port_0 = nullptr;
    char* input_name_0 = nullptr;
    OV_EXPECT_OK(ov_model_const_input_by_index(new_model, 0, &port_0));
    OV_EXPECT_OK(ov_port_get_any_name(port_0, &input_name_0));

    ov_output_const_port_t* port_1 = nullptr;
    char* input_name_1 = nullptr;
    OV_EXPECT_OK(ov_model_const_input_by_index(new_model, 1, &port_1));
    OV_EXPECT_OK(ov_port_get_any_name(port_1, &input_name_1));

    const char* context_type = "OCL";
    OV_EXPECT_OK(ov_core_create_context(core,
                                        "GPU",
                                        6,
                                        &context,
                                        ov_property_key_intel_gpu_context_type,
                                        context_type,
                                        ov_property_key_intel_gpu_ocl_context,
                                        cl_context.get(),
                                        ov_property_key_intel_gpu_ocl_queue,
                                        cl_queue.get()));
    EXPECT_NE(nullptr, context);

    ov_shape_t shape_y = {0, nullptr};
    int64_t dims_y[4] = {1, height, width, 1};
    ov_shape_t shape_uv = {0, nullptr};
    int64_t dims_uv[4] = {1, height / 2, width / 2, 2};

    cl_int err;
    cl_image_format image_format;
    cl_image_desc image_desc = {0};
    image_format.image_channel_order = CL_R;
    image_format.image_channel_data_type = CL_UNORM_INT8;
    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = width;
    image_desc.image_height = height;
    cl_mem mem_y = clCreateImage(cl_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
    ASSERT_EQ(err, 0);

    image_format.image_channel_order = CL_RG;
    image_desc.image_width = width / 2;
    image_desc.image_height = height / 2;
    cl_mem mem_uv = clCreateImage(cl_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
    ASSERT_EQ(err, 0);

    cl::Image2D image_y = cl::Image2D(mem_y);
    cl::Image2D image_uv = cl::Image2D(mem_uv);

    ov_tensor_t* remote_tensor_y = nullptr;
    ov_tensor_t* remote_tensor_uv = nullptr;
    OV_EXPECT_OK(ov_shape_create(4, dims_y, &shape_y));
    OV_EXPECT_OK(ov_remote_context_create_tensor(context,
                                                 ov_element_type_e::U8,
                                                 shape_y,
                                                 4,
                                                 &remote_tensor_y,
                                                 ov_property_key_intel_gpu_shared_mem_type,
                                                 "OCL_IMAGE2D",
                                                 ov_property_key_intel_gpu_mem_handle,
                                                 image_y.get()));
    EXPECT_NE(nullptr, remote_tensor_y);

    OV_EXPECT_OK(ov_shape_create(4, dims_uv, &shape_uv));
    OV_EXPECT_OK(ov_remote_context_create_tensor(context,
                                                 ov_element_type_e::U8,
                                                 shape_uv,
                                                 4,
                                                 &remote_tensor_uv,
                                                 ov_property_key_intel_gpu_shared_mem_type,
                                                 "OCL_IMAGE2D",
                                                 ov_property_key_intel_gpu_mem_handle,
                                                 image_uv.get()));
    EXPECT_NE(nullptr, remote_tensor_uv);

    OV_EXPECT_OK(ov_core_compile_model_with_context(core, new_model, context, 0, &compiled_model));
    EXPECT_NE(nullptr, compiled_model);

    OV_EXPECT_OK(ov_compiled_model_create_infer_request(compiled_model, &infer_request));
    EXPECT_NE(nullptr, infer_request);

    OV_EXPECT_OK(ov_infer_request_set_tensor(infer_request, input_name_0, remote_tensor_y));
    OV_EXPECT_OK(ov_infer_request_set_tensor(infer_request, input_name_1, remote_tensor_uv));
    OV_ASSERT_OK(ov_infer_request_infer(infer_request));

    ov_free(input_name_0);
    ov_free(input_name_1);
    ov_output_const_port_free(port_0);
    ov_output_const_port_free(port_1);

    ov_layout_free(layout);
    ov_preprocess_input_model_info_free(preprocess_input_model_info);
    ov_preprocess_preprocess_steps_free(preprocess_input_steps);
    ov_preprocess_input_tensor_info_free(preprocess_input_tensor_info);
    ov_preprocess_input_info_free(preprocess_input_info);
    ov_model_free(new_model);
    ov_preprocess_prepostprocessor_free(preprocess);

    ov_tensor_free(remote_tensor_y);
    ov_tensor_free(remote_tensor_uv);
    ov_shape_free(&shape_y);
    ov_shape_free(&shape_uv);
}
