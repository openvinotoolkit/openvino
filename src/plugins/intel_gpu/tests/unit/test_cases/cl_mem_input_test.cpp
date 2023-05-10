// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "opencl_helper_instance.hpp"
#include "ocl/ocl_device.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/runtime/device_query.hpp>

using namespace cldnn;
using namespace ::tests;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
typedef std::chrono::duration<float> fsec;

namespace {
std::vector<unsigned char> createSampleData(int width, int height) {
    int data_size = width * (height + height / 2);
    auto data = std::vector<unsigned char>(data_size);
    srand((unsigned)time(0));
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            data[i * width + j] = (i + j) % 255;
        }
    }
    for (int i = 0; i < height / 2; i++) {
        for (int j = 0; j < width; j += 2) {
            data[width * height + i * width + j] = (i + j) % 255;
            data[width * height + i * width + j + 1] = (i + j) % 255;
        }
    }

    return data;
}

std::vector<float> createReferenceData(std::vector<unsigned char> data, int width, int height, cldnn::format format) {
    auto img = std::vector<float>(width * height * 3);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int y_comp = data[i * width + j];
            int u_comp = data[width * height + i / 2 * width + ((j >> 1) << 1)];
            int v_comp = data[width * height + i / 2 * width + ((j >> 1) << 1) + 1];

            float B = (1.164f * (float)(y_comp - 16) + 1.596f * (float)(v_comp - 128));
            float G = (1.164f * (float)(y_comp - 16) - 0.813f * (float)(v_comp - 128) - 0.391f * (u_comp - 128));
            float R = (1.164f * (float)(y_comp - 16) + 2.018f * (float)(u_comp - 128));

            R = std::min(std::max(R, 0.f), 255.f);
            G = std::min(std::max(G, 0.f), 255.f);
            B = std::min(std::max(B, 0.f), 255.f);

            if (format == cldnn::format::bfyx) {
                img[j + width * i] = R;
                img[j + width * i + width * height] = G;
                img[j + width * i + width * height * 2] = B;
            } else { //byxf
                img[3* width*i + 3 * j] = R;
                img[3 * width * i + 3 * j + 1] = G;
                img[3 * width*i + 3 * j + 2] = B;
            }
        }
    }

    return img;
}
}  // namespace

template <typename T>
void start_cl_mem_check_2_inputs(bool is_caching_test) {
    device_query query(engine_types::ocl, runtime_types::ocl);
    auto devices = query.get_available_devices();
    ASSERT_TRUE(!devices.empty());

    auto iter = devices.find(std::to_string(device_query::device_id));
    auto& device = iter != devices.end() ? iter->second : devices.begin()->second;
    auto engine = engine::create(engine_types::ocl, runtime_types::ocl, device);

    auto ocl_instance = std::make_shared<OpenCL>(std::dynamic_pointer_cast<ocl::ocl_device>(device)->get_device());
    int width = 224;
    int height = 224;
    cl_int err;

    if (!device->get_info().supports_image)
        GTEST_SKIP();

    auto data = createSampleData(width, height);
    cl_image_format image_format;
    image_format.image_channel_order = CL_R;
    image_format.image_channel_data_type = CL_UNORM_INT8;
    cl_image_desc image_desc = { CL_MEM_OBJECT_IMAGE2D, (size_t)width, (size_t)height, 0,
                                 0, 0, 0, 0, 0, { nullptr } };

    cl_mem nv12_image_plane_y = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
    checkStatus(err, "Creating nv12 image plane_y failed");

    image_format.image_channel_order = CL_RG;
    image_desc.image_width = width / 2;
    image_desc.image_height = height / 2;
    image_desc.image_depth = 1;

    cl_mem nv12_image_plane_uv = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
    checkStatus(err, "Creating nv12 image plane_uv failed");

    size_t origin[3] = { 0, 0, 0 };
    size_t y_region[3] = { (size_t)width, (size_t)height, 1 };
    size_t uv_region[3] = { (size_t)width / 2, (size_t)height / 2, 1 };

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_y, true, origin, y_region, 0, 0, &data[0], 0, nullptr, nullptr);
    checkStatus(err, "Writing nv12 image plane_y failed");

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_uv, true, origin, uv_region, 0, 0, &data[width * height], 0, nullptr, nullptr);
    checkStatus(err, "Writing nv12 image plane_uv failed");

    auto input = input_layout("input", { data_types::i8, format::nv12, {1,1,height,width} });
    auto input2 = input_layout("input2", { data_types::i8, format::nv12, {1,1,height / 2,width / 2} });
    auto output_format = cldnn::format::byxf;
    layout output_layout(data_types::f32, output_format, { 1,3,height,width });
    auto input_memory = engine->share_image(input.layout, nv12_image_plane_y);
    auto input_memory2 = engine->share_image(input2.layout, nv12_image_plane_uv);

    topology topology;
    topology.add(input);
    topology.add(input2);
    topology.add(reorder("reorder", input_info("input"), input_info("input2"), output_layout));

    cldnn::network::ptr network = get_network(*engine, topology, get_test_default_config(*engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input_memory);
    network->set_input_data("input2", input_memory2);

    auto outputs = network->execute();

    std::vector<T> reference_results = createReferenceData(data, width, height, output_format);
    auto output_prim = outputs.begin()->second.get_memory();
    cldnn::mem_lock<T> output_ptr(output_prim, get_test_stream());
    int size = width * height * 3;
    for (auto i = 0; i < size; i++) {
        ASSERT_NEAR(reference_results[i], output_ptr[i], 1.001f);
    }
    checkStatus(clReleaseMemObject(nv12_image_plane_uv), "clReleaseMemObject");
    checkStatus(clReleaseMemObject(nv12_image_plane_y), "clReleaseMemObject");
}

TEST(cl_mem_check, check_2_inputs) {
    start_cl_mem_check_2_inputs<float>(false);
}

TEST(export_import_cl_mem_check, check_2_inputs) {
    start_cl_mem_check_2_inputs<float>(true);
}

TEST(cl_mem_check, check_input) {
    device_query query(engine_types::ocl, runtime_types::ocl);
    auto devices = query.get_available_devices();
    ASSERT_TRUE(!devices.empty());
    auto iter = devices.find(std::to_string(device_query::device_id));
    auto& device = iter != devices.end() ? iter->second : devices.begin()->second;
    auto engine = engine::create(engine_types::ocl, runtime_types::ocl, device);
    auto ocl_instance = std::make_shared<OpenCL>(std::dynamic_pointer_cast<ocl::ocl_device>(device)->get_device());

    if (!device->get_info().supports_intel_planar_yuv)
        GTEST_SKIP();

    int width = 224;
    int height = 224;
    cl_int err;

    auto data = createSampleData(width, height);
    cl_image_format image_format;
    image_format.image_channel_order = CL_R;
    image_format.image_channel_data_type = CL_UNORM_INT8;
    cl_image_desc image_desc = { CL_MEM_OBJECT_IMAGE2D, (size_t)width, (size_t)height, 0,
        0, 0, 0, 0, 0, { nullptr } };

    cl_mem nv12_image_plane_y = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
    checkStatus(err, "Creating nv12 image plane_y failed");

    image_format.image_channel_order = CL_RG;
    image_desc.image_width = width / 2;
    image_desc.image_height = height / 2;

    cl_mem nv12_image_plane_uv = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
    checkStatus(err, "Creating nv12 image plane_uv failed");

    size_t origin[3] = { 0, 0, 0 };
    size_t y_region[3] = { (size_t)width, (size_t)height, 1 };
    size_t uv_region[3] = { (size_t)width / 2, (size_t)height / 2, 1 };

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_y, true, origin, y_region, 0, 0, &data[0], 0, nullptr, nullptr);
    checkStatus(err, "Writing nv12 image plane_y failed");

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_uv, true, origin, uv_region, 0, 0, &data[width * height], 0, nullptr, nullptr);
    checkStatus(err, "Writing nv12 image plane_uv failed");

    image_format.image_channel_order = CL_NV12_INTEL;
    image_format.image_channel_data_type = CL_UNORM_INT8;

    image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    image_desc.image_width = width;
    image_desc.image_height = height;
    image_desc.image_array_size = 0;
    image_desc.image_row_pitch = 0;
    image_desc.image_slice_pitch = 0;
    image_desc.num_mip_levels = 0;
    image_desc.num_samples = 0;
    image_desc.buffer = NULL;

    cl_mem img = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_ACCESS_FLAGS_UNRESTRICTED_INTEL,
        &image_format, &image_desc, NULL, &err);
    checkStatus(err, "Creating nv12 image failed");

    image_desc.image_width = 0;
    image_desc.image_height = 0;
    image_desc.buffer = img;
    image_desc.image_depth = 0;
    image_format.image_channel_order = CL_R;

    cl_mem img_y = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
    checkStatus(err, "Creating nv12 image plane_y failed");

    image_desc.image_depth = 1;
    image_format.image_channel_order = CL_RG;
    cl_mem img_uv = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
    checkStatus(err, "Creating nv12 image plane_uv failed");

    size_t regionY[] = { (size_t)width, (size_t)height, 1 };
    size_t regionUV[] = { (size_t)width / 2, (size_t)height / 2, 1 };

    err = clEnqueueCopyImage(ocl_instance->_queue.get(), nv12_image_plane_y, img_y, origin, origin, regionY, 0, 0, 0);
    checkStatus(err, "clEnqueueCopyImage");

    cl_event event_out;
    err = clEnqueueCopyImage(ocl_instance->_queue.get(), nv12_image_plane_uv, img_uv, origin, origin, regionUV, 0, 0, &event_out);
    checkStatus(err, "clEnqueueCopyImage");

    checkStatus(clReleaseMemObject(nv12_image_plane_uv), "clReleaseMemObject");
    checkStatus(clReleaseMemObject(nv12_image_plane_y), "clReleaseMemObject");

    auto input = input_layout("input", { data_types::i8, format::nv12, {1,1,height,width} });
    auto output_format = cldnn::format::byxf;
    layout output_layout(data_types::f32, output_format, { 1,3,height,width });
    auto input_memory = engine->share_image(input.layout, img);

    topology topology;

    topology.add(input);
    topology.add(reorder("reorder", input_info("input"), output_layout));

    network network(*engine, topology, get_test_default_config(*engine));
    network.set_input_data("input", input_memory);

    auto outputs = network.execute();

    std::vector<float> reference_results = createReferenceData(data, width, height, output_format);
    auto output_prim = outputs.begin()->second.get_memory();
    cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());
    int size = width * height * 3;
    for (auto i = 0; i < size; i++) {
        ASSERT_NEAR(reference_results[i], output_ptr[i], 1.001f);
    }
    checkStatus(clReleaseMemObject(img), "clReleaseMemObject");
}

TEST(cl_mem_check, check_write_access_type) {
    device_query query(engine_types::ocl, runtime_types::ocl);
    auto devices = query.get_available_devices();
    ASSERT_TRUE(!devices.empty());
    cldnn::device::ptr device = devices.begin()->second;
    for (auto& dev : devices) {
        if (dev.second->get_info().dev_type == device_type::discrete_gpu)
            device = dev.second;
    }

    auto engine = engine::create(engine_types::ocl, runtime_types::ocl, device);
    auto stream = engine->create_stream({});

    size_t values_count = 100;
    size_t values_bytes_count = values_count * sizeof(float);
    std::vector<float> src_buffer(values_count);
    std::iota(src_buffer.begin(), src_buffer.end(), 0.0f);

    cldnn::layout linear_layout = cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, cldnn::tensor(1, 1, int32_t(values_count), 1));
    auto cldnn_mem_src = engine->allocate_memory(linear_layout, cldnn::allocation_type::cl_mem);
    {
        cldnn::mem_lock<float, cldnn::mem_lock_type::write> lock(cldnn_mem_src, *stream);
        std::copy(src_buffer.begin(), src_buffer.end(), lock.data());
    }

    std::vector<float> dst_buffer(values_count);
    {
        cldnn::mem_lock<float, cldnn::mem_lock_type::read> lock(cldnn_mem_src, *stream);
        std::memcpy(dst_buffer.data(), lock.data(), values_bytes_count);
    }

    bool are_equal = std::equal(src_buffer.begin(), src_buffer.begin() + values_count, dst_buffer.begin());
    ASSERT_TRUE(are_equal);
}

TEST(cl_mem_check, check_read_access_type) {
    device_query query(engine_types::ocl, runtime_types::ocl);
    auto devices = query.get_available_devices();
    ASSERT_TRUE(!devices.empty());
    cldnn::device::ptr device = devices.begin()->second;
    for (auto& dev : devices) {
        if (dev.second->get_info().dev_type == device_type::discrete_gpu)
            device = dev.second;
    }
    if (device->get_info().dev_type == device_type::integrated_gpu) {
        GTEST_SKIP();
    }

    auto engine = engine::create(engine_types::ocl, runtime_types::ocl, device);
    auto stream = engine->create_stream({});

    size_t values_count = 100;
    size_t values_bytes_count = values_count * sizeof(float);
    std::vector<float> src_buffer(values_count);
    std::iota(src_buffer.begin(), src_buffer.end(), 0.0f);

    cldnn::layout linear_layout = cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, cldnn::tensor(1, 1, int32_t(values_count), 1));
    auto cldnn_mem_src = engine->allocate_memory(linear_layout, cldnn::allocation_type::cl_mem);
    {
        cldnn::mem_lock<float, cldnn::mem_lock_type::write> lock(cldnn_mem_src, *stream);
        std::copy(src_buffer.begin(), src_buffer.end(), lock.data());
    }

    {
        cldnn::mem_lock<float, cldnn::mem_lock_type::read> lock(cldnn_mem_src, *stream);
        std::copy(src_buffer.rbegin(), src_buffer.rend(), lock.data());
    }

    std::vector<float> dst_buffer(values_count);
    {
        cldnn::mem_lock<float, cldnn::mem_lock_type::read> lock(cldnn_mem_src, *stream);
        std::memcpy(dst_buffer.data(), lock.data(), values_bytes_count);
    }

    bool are_equal = std::equal(src_buffer.begin(), src_buffer.begin() + values_count, dst_buffer.begin());
    ASSERT_TRUE(are_equal);
}
