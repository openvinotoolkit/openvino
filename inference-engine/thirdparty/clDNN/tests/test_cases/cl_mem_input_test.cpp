/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include "api/activation.hpp"
#include <api/topology.hpp>
#include <api/tensor.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include <api/data.hpp>
#include "test_utils/test_utils.h"

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl2.hpp>
#include "CL/cl_intel_planar_yuv.h"

using namespace cldnn;
using namespace tests;

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::nanoseconds ns;
typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
typedef std::chrono::duration<float> fsec;


void checkStatus(int status, const char *message)
{
    if (status != 0)
    {
        std::string str_message(message + std::string(": "));
        std::string str_number(std::to_string(status));

        throw std::runtime_error(str_message + str_number);
    }
}

std::vector<unsigned char> createSampleData(int width, int height)
{
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

std::vector<float> createReferenceData(std::vector<unsigned char> data, int width, int height, cldnn::format format)
{
    auto img = std::vector<float>(width * height * 3);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int y_comp = data[i * width + j];
            int u_comp = data[width * height + i / 2 * width + ((j >> 1) << 1)];
            int v_comp = data[width * height + i / 2 * width + ((j >> 1) << 1) + 1];

            float R = (1.164f * (float)(y_comp - 16) + 1.596f * (float)(v_comp - 128));
            float G = (1.164f * (float)(y_comp - 16) - 0.813f * (float)(v_comp - 128) - 0.391f * (u_comp - 128));
            float B = (1.164f * (float)(y_comp - 16) + 2.018f * (float)(u_comp - 128));
            
            R = std::min(std::max(R, 0.f), 255.f);
            G = std::min(std::max(G, 0.f), 255.f);
            B = std::min(std::max(B, 0.f), 255.f);

            if (format == cldnn::format::bfyx) {
                img[j + width * i] = R;
                img[j + width * i + width * height] = G;
                img[j + width * i + width * height * 2] = B;
            }
            else { //byxf
                img[3* width*i + 3 * j] = R;
                img[3 * width * i + 3 * j + 1] = G;
                img[3 * width*i + 3 * j + 2] = B;
            }
        }
    }

    return img;
}

struct OpenCL
{
    cl::Context _context;
    cl::Device _device;
    cl::CommandQueue _queue;

    OpenCL()
    {
        // get Intel iGPU OCL device, create context and queue
        {
            static constexpr auto INTEL_PLATFORM_VENDOR = "Intel(R) Corporation";
            const uint32_t device_type = CL_DEVICE_TYPE_GPU;  // only gpu devices
            const uint32_t device_vendor = 0x8086;  // Intel vendor

            cl_uint n = 0;
            cl_int err = clGetPlatformIDs(0, NULL, &n);
            checkStatus(err, "clGetPlatformIDs");

            // Get platform list
            std::vector<cl_platform_id> platform_ids(n);
            err = clGetPlatformIDs(n, platform_ids.data(), NULL);
            checkStatus(err, "clGetPlatformIDs");

            for (auto& id : platform_ids) {
                cl::Platform platform = cl::Platform(id);

                auto vendor_id = platform.getInfo<CL_PLATFORM_VENDOR>();
                if (vendor_id != INTEL_PLATFORM_VENDOR)
                    continue;

                std::vector<cl::Device> devices;
                platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
                for (auto& d : devices) {
                    if (d.getInfo<CL_DEVICE_TYPE>() == device_type ||
                        d.getInfo<CL_DEVICE_VENDOR_ID>() == device_vendor) {
                        _device = d;
                        _context = cl::Context(_device);
                        goto greateQueue;
                    }
                }
            }
            greateQueue:
            cl_command_queue_properties props = CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
            _queue = cl::CommandQueue(_context, _device, props);
        }
    }
    void releaseOclImage(std::shared_ptr<cl_mem> image)
    {
        checkStatus(clReleaseMemObject(*image), "clReleaseMemObject");
    }
};

TEST(cl_mem_check, check_2_inputs) {
    auto ocl_instance = std::make_shared<OpenCL>();
    int width = 224;
    int height = 224;
    cl_int err;

    auto data = createSampleData(width, height);
    cl_image_format image_format;
    image_format.image_channel_order = CL_R;
    image_format.image_channel_data_type = CL_UNORM_INT8;
    cl_image_desc image_desc = { CL_MEM_OBJECT_IMAGE2D, (size_t)width, (size_t)height, 0,
                                 0, 0, 0, 0, 0, NULL };

    cl_mem nv12_image_plane_y = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err);
    checkStatus(err, "Creating nv12 image plane_y failed");

    image_format.image_channel_order = CL_RG;
    image_desc.image_width = width / 2;
    image_desc.image_height = height / 2;
    image_desc.image_depth = 1;

    cl_mem nv12_image_plane_uv = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err);
    checkStatus(err, "Creating nv12 image plane_uv failed");

    size_t origin[3] = { 0, 0, 0 };
    size_t y_region[3] = { (size_t)width, (size_t)height, 1 };
    size_t uv_region[3] = { (size_t)width / 2, (size_t)height / 2, 1 };

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_y, true, origin, y_region, 0, 0, &data[0], 0, NULL, NULL);
    checkStatus(err, "Writing nv12 image plane_y failed");

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_uv, true, origin, uv_region, 0, 0, &data[width * height], 0, NULL, NULL);
    checkStatus(err, "Writing nv12 image plane_uv failed");

    device_query query(static_cast<void*>(ocl_instance->_context.get()));
    auto devices = query.get_available_devices();

    auto engine_config = cldnn::engine_configuration();
    engine engine(devices.begin()->second, engine_config);

    auto input = input_layout("input", { data_types::i8, format::nv12, {1,1,height,width} });
    auto input2 = input_layout("input2", { data_types::i8, format::nv12, {1,1,height / 2,width / 2} });
    auto output_format = cldnn::format::byxf;
    layout output_layout(data_types::f32, output_format, { 1,3,height,width });
    auto input_memory = cldnn::memory::share_image(engine, input.layout, nv12_image_plane_y,  0);
    auto input_memory2 = cldnn::memory::share_image(engine,  input2.layout, nv12_image_plane_uv, 0);

    topology topology;
    topology.add(input);
    topology.add(input2);
    topology.add(reorder("reorder", "input", "input2", output_layout));

    network network(engine, topology);
    network.set_input_data("input", input_memory);
    network.set_input_data("input2", input_memory2);

    auto outputs = network.execute();

    std::vector<float> reference_results = createReferenceData(data, width, height, output_format);
    auto output_prim = outputs.begin()->second.get_memory();
    auto output_ptr = output_prim.pointer<float>();
    int size = width * height * 3;
    for (auto i = 0; i < size; i++) {
        EXPECT_NEAR(reference_results[i], output_ptr[i], 1.001f);
    }
    checkStatus(clReleaseMemObject(nv12_image_plane_uv), "clReleaseMemObject");
    checkStatus(clReleaseMemObject(nv12_image_plane_y), "clReleaseMemObject");
}

TEST(cl_mem_check, check_input) {
    auto ocl_instance = std::make_shared<OpenCL>();
    int width = 224;
    int height = 224;
    cl_int err;

    auto data = createSampleData(width, height);
    cl_image_format image_format;
    image_format.image_channel_order = CL_R;
    image_format.image_channel_data_type = CL_UNORM_INT8;
    cl_image_desc image_desc = { CL_MEM_OBJECT_IMAGE2D, (size_t)width, (size_t)height, 0,
        0, 0, 0, 0, 0, NULL };

    cl_mem nv12_image_plane_y = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err);
    checkStatus(err, "Creating nv12 image plane_y failed");

    image_format.image_channel_order = CL_RG;
    image_desc.image_width = width / 2;
    image_desc.image_height = height / 2;

    cl_mem nv12_image_plane_uv = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err);
    checkStatus(err, "Creating nv12 image plane_uv failed");

    size_t origin[3] = { 0, 0, 0 };
    size_t y_region[3] = { (size_t)width, (size_t)height, 1 };
    size_t uv_region[3] = { (size_t)width / 2, (size_t)height / 2, 1 };

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_y, true, origin, y_region, 0, 0, &data[0], 0, NULL, NULL);
    checkStatus(err, "Writing nv12 image plane_y failed");

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image_plane_uv, true, origin, uv_region, 0, 0, &data[width * height], 0, NULL, NULL);
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
    image_desc.mem_object = NULL;

    cl_mem img = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_ACCESS_FLAGS_UNRESTRICTED_INTEL,
        &image_format, &image_desc, NULL, &err);
    checkStatus(err, "Creating nv12 image failed");

    image_desc.image_width = 0;
    image_desc.image_height = 0;
    image_desc.mem_object = img;
    image_desc.image_depth = 0;
    image_format.image_channel_order = CL_R;

    cl_mem img_y = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err);
    checkStatus(err, "Creating nv12 image plane_y failed");

    image_desc.image_depth = 1;
    image_format.image_channel_order = CL_RG;
    cl_mem img_uv = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, NULL, &err);
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

    device_query query(static_cast<void*>(ocl_instance->_context.get()));
    auto devices = query.get_available_devices();

    auto engine_config = cldnn::engine_configuration();
    engine engine(devices.begin()->second, engine_config);

    auto input = input_layout("input", { data_types::i8, format::nv12, {1,1,height,width} });
    auto output_format = cldnn::format::byxf;
    layout output_layout(data_types::f32, output_format, { 1,3,height,width });
    auto input_memory = cldnn::memory::share_image(engine, input.layout, img,  0);

    topology topology;

    topology.add(input);
    topology.add(reorder("reorder", "input", output_layout));

    network network(engine, topology);
    network.set_input_data("input", input_memory);

    auto outputs = network.execute();

    std::vector<float> reference_results = createReferenceData(data, width, height, output_format);
    auto output_prim = outputs.begin()->second.get_memory();
    auto output_ptr = output_prim.pointer<float>();
    int size = width * height * 3;
    for (auto i = 0; i < size; i++) {
        EXPECT_NEAR(reference_results[i], output_ptr[i], 1.001f);
    }
    checkStatus(clReleaseMemObject(img), "clReleaseMemObject");
}

