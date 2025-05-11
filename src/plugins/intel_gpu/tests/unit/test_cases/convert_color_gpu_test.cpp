// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "opencl_helper_instance.hpp"
#include "ocl/ocl_device.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/convert_color.hpp>
#include <intel_gpu/runtime/device_query.hpp>

#include <ocl/ocl_wrapper.hpp>

using namespace cldnn;
using namespace ::tests;

template <typename T, typename U>
void createReferenceDataNV12(const T* arg_y, const T* arg_uv, U* out_ptr,
                             size_t batch_size, size_t image_h, size_t image_w,
                             size_t stride_y, size_t stride_uv, bool to_rgb) {
    for (size_t batch = 0; batch < batch_size; ++batch) {
        U* out = out_ptr + batch * image_w * image_h * 3;
        auto y_ptr = arg_y + batch * stride_y;
        auto uv_ptr = arg_uv + batch * stride_uv;
        for (size_t h = 0; h < image_h; ++h) {
            for (size_t w = 0; w < image_w; ++w) {
                auto y_index = h * image_w + w;
                auto y_val = static_cast<float>(y_ptr[y_index]);
                auto uv_index = (h / 2) * image_w + (w / 2) * 2;
                auto u_val = static_cast<float>(uv_ptr[uv_index]);
                auto v_val = static_cast<float>(uv_ptr[uv_index + 1]);
                auto c = y_val - 16.f;
                auto d = u_val - 128.f;
                auto e = v_val - 128.f;
                auto clip = [](float a) -> U {
                    if (std::is_integral<U>()) {
                        return static_cast<U>(std::min(std::max(std::round(a), 0.f), 255.f));
                    } else {
                        return static_cast<U>(std::min(std::max(a, 0.f), 255.f));
                    }
                };
                auto b = clip(1.164f * c + 2.018f * d);
                auto g = clip(1.164f * c - 0.391f * d - 0.813f * e);
                auto r = clip(1.164f * c + 1.596f * e);

                if (to_rgb) {
                    out[y_index * 3] = r;
                    out[y_index * 3 + 1] = g;
                    out[y_index * 3 + 2] = b;
                } else {
                    out[y_index * 3] = b;
                    out[y_index * 3 + 1] = g;
                    out[y_index * 3 + 2] = r;
                }
            }
        }
    }
}

TEST(convert_color, nv12_to_rgb_two_planes_buffer_fp32) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    int width = 224;
    int height = 448;

    auto input_y = engine.allocate_memory({ { 1, height, width, 1 }, data_types::f32, format::bfyx });
    auto input_uv = engine.allocate_memory({ { 1, height / 2 , width / 2, 2 }, data_types::f32, format::bfyx});

    std::vector<float> input_y_data = rg.generate_random_1d<float>(width * height, 0, 255);
    std::vector<float> input_uv_data = rg.generate_random_1d<float>(width * height / 2, 0, 255);

    set_values(input_y, input_y_data);
    set_values(input_uv, input_uv_data);

    topology topology;
    topology.add(input_layout("input_y", input_y->get_layout()));
    topology.add(input_layout("input_uv", input_uv->get_layout()));
    topology.add(convert_color("convert_color",
                               { input_info("input_y"), input_info("input_uv") },
                               cldnn::convert_color::color_format::NV12,
                               cldnn::convert_color::color_format::RGB,
                               cldnn::convert_color::memory_type::buffer));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input_y", input_y);
    network.set_input_data("input_uv", input_uv);

    auto outputs = network.execute();

    std::vector<float> ref_res(width * height * 3);
    createReferenceDataNV12<float, float>(input_y_data.data(), input_uv_data.data(), ref_res.data(),
                                          1, height, width, height * width, height * width / 2, true);
    auto output = outputs.at("convert_color").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_res.size(); ++i) {
        ASSERT_NEAR(ref_res[i], output_ptr[i], 1.001f);
    }
}

TEST(convert_color, nv12_to_bgr_two_planes_buffer_fp32) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    int width = 224;
    int height = 224;

    auto input_y = engine.allocate_memory({ { 1, height, width, 1 }, data_types::f32, format::bfyx });
    auto input_uv = engine.allocate_memory({ { 1, height / 2 , width / 2, 2 }, data_types::f32, format::bfyx});

    std::vector<float> input_y_data = rg.generate_random_1d<float>(width * height, 0, 255);
    std::vector<float> input_uv_data = rg.generate_random_1d<float>(width * height / 2, 0, 255);

    set_values(input_y, input_y_data);
    set_values(input_uv, input_uv_data);

    topology topology;
    topology.add(input_layout("input_y", input_y->get_layout()));
    topology.add(input_layout("input_uv", input_uv->get_layout()));
    topology.add(convert_color("convert_color",
                               { input_info("input_y"), input_info("input_uv") },
                               cldnn::convert_color::color_format::NV12,
                               cldnn::convert_color::color_format::BGR,
                               cldnn::convert_color::memory_type::buffer));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input_y", input_y);
    network.set_input_data("input_uv", input_uv);

    auto outputs = network.execute();

    std::vector<float> ref_res(width * height * 3);
    createReferenceDataNV12<float>(input_y_data.data(), input_uv_data.data(), ref_res.data(),
                                   1, height, width, height * width, height * width / 2, false);

    auto output = outputs.at("convert_color").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_res.size(); ++i) {
        ASSERT_NEAR(ref_res[i], output_ptr[i], 1.001f);
    }
}

TEST(convert_color, nv12_to_rgb_two_planes_buffer_u8) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    int width = 224;
    int height = 224;

    auto input_y = engine.allocate_memory({ { 1, height, width, 1 }, data_types::u8, format::bfyx });
    auto input_uv = engine.allocate_memory({ { 1, height / 2 , width / 2, 2 }, data_types::u8, format::bfyx});

    std::vector<uint8_t> input_y_data = rg.generate_random_1d<uint8_t>(width * height, 0, 255);
    std::vector<uint8_t> input_uv_data = rg.generate_random_1d<uint8_t>(width * height / 2, 0, 255);

    set_values(input_y, input_y_data);
    set_values(input_uv, input_uv_data);

    topology topology;
    topology.add(input_layout("input_y", input_y->get_layout()));
    topology.add(input_layout("input_uv", input_uv->get_layout()));
    topology.add(convert_color("convert_color",
                               { input_info("input_y"), input_info("input_uv") },
                               cldnn::convert_color::color_format::NV12,
                               cldnn::convert_color::color_format::RGB,
                               cldnn::convert_color::memory_type::buffer));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input_y", input_y);
    network.set_input_data("input_uv", input_uv);

    auto outputs = network.execute();

    std::vector<float> ref_res(width * height * 3);
    createReferenceDataNV12<uint8_t, float>(input_y_data.data(), input_uv_data.data(), ref_res.data(),
                                            1, height, width, height * width, height * width / 2, true);

    auto output = outputs.at("convert_color").get_memory();
    cldnn::mem_lock<uint8_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_res.size(); ++i) {
        ASSERT_NEAR(ref_res[i], static_cast<float>(output_ptr[i]), 1.001f);
    }
}

TEST(convert_color, nv12_to_rgb_two_planes_buffer_fp16) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    int width = 224;
    int height = 224;

    auto input_y = engine.allocate_memory({ { 1, height, width, 1 }, data_types::f16, format::bfyx });
    auto input_uv = engine.allocate_memory({ { 1, height / 2 , width / 2, 2 }, data_types::f16, format::bfyx});

    std::vector<ov::float16> input_y_data = rg.generate_random_1d<ov::float16>(width * height, 0, 255);
    std::vector<ov::float16> input_uv_data = rg.generate_random_1d<ov::float16>(width * height / 2, 0, 255);

    set_values(input_y, input_y_data);
    set_values(input_uv, input_uv_data);

    topology topology;
    topology.add(input_layout("input_y", input_y->get_layout()));
    topology.add(input_layout("input_uv", input_uv->get_layout()));
    topology.add(convert_color("convert_color",
                               { input_info("input_y"), input_info("input_uv") },
                               cldnn::convert_color::color_format::NV12,
                               cldnn::convert_color::color_format::RGB,
                               cldnn::convert_color::memory_type::buffer));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input_y", input_y);
    network.set_input_data("input_uv", input_uv);

    auto outputs = network.execute();

    std::vector<float> ref_res(width * height * 3);
    createReferenceDataNV12<ov::float16, float>(input_y_data.data(), input_uv_data.data(), ref_res.data(),
                                            1, height, width, height * width, height * width / 2, true);

    auto output = outputs.at("convert_color").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

     for (size_t i = 0; i < ref_res.size(); ++i) {
        ASSERT_NEAR(ref_res[i], half_to_float(output_ptr[i]), 1.001f);
    }
}

TEST(convert_color, nv12_to_rgb_single_plane_buffer_fp32) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    int width = 224;
    int height = 448;
    int input_height = height + height / 2;

    auto input = engine.allocate_memory({{ 1, input_height, width, 1 }, data_types::f32, format::bfyx});

    int data_size = width * (height + height / 2);
    std::vector<float> input_data = rg.generate_random_1d<float>(data_size, 0, 255);
    set_values(input, input_data);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(convert_color("convert_color",
                               { input_info("input") },
                               cldnn::convert_color::color_format::NV12,
                               cldnn::convert_color::color_format::RGB,
                               cldnn::convert_color::memory_type::buffer));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();

    std::vector<float> ref_res(width * height * 3);
    createReferenceDataNV12<float, float>(input_data.data(), input_data.data() + height * width, ref_res.data(),
                                          1, height, width, input_height * width, input_height * width, true);
    auto output = outputs.at("convert_color").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_res.size(); ++i) {
        ASSERT_NEAR(ref_res[i], output_ptr[i], 1.001f);
    }
}

TEST(convert_color, nv12_to_rgb_single_plane_buffer_u8) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();

    int width = 224;
    int height = 448;
    int input_height = height + height / 2;

    auto input = engine.allocate_memory({{ 1, input_height, width, 1 }, data_types::u8, format::bfyx});

    int data_size = width * (height + height / 2);
    std::vector<uint8_t> input_data = rg.generate_random_1d<uint8_t>(data_size, 0, 255);
    set_values(input, input_data);

    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(convert_color("convert_color",
                               { input_info("input") },
                               cldnn::convert_color::color_format::NV12,
                               cldnn::convert_color::color_format::RGB,
                               cldnn::convert_color::memory_type::buffer));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input", input);

    auto outputs = network.execute();

    std::vector<float> ref_res(width * height * 3);
    createReferenceDataNV12<uint8_t, float>(input_data.data(), input_data.data() + height * width, ref_res.data(),
                                            1, height, width, input_height * width, input_height * width, true);
    auto output = outputs.at("convert_color").get_memory();
    cldnn::mem_lock<uint8_t> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_res.size(); ++i) {
        ASSERT_NEAR(ref_res[i], static_cast<float>(output_ptr[i]), 1.001f);
    }
}

TEST(convert_color, nv12_to_rgb_two_planes_surface_u8) {
    tests::random_generator rg(GET_SUITE_NAME);
    int width = 224;
    int height = 448;

    device_query query(engine_types::ocl, runtime_types::ocl);
    auto devices = query.get_available_devices();
    ASSERT_TRUE(!devices.empty());
    auto iter = devices.find(std::to_string(device_query::device_id));
    auto& device = iter != devices.end() ? iter->second : devices.begin()->second;
    auto engine = engine::create(engine_types::ocl, runtime_types::ocl, device);
    auto ocl_instance = std::make_shared<OpenCL>(std::dynamic_pointer_cast<ocl::ocl_device>(device)->get_device());

    if (!engine->get_device_info().supports_image) {
        GTEST_SKIP() << "Device doesn't support images";
    }

    cl_int err;

    int data_size = width * (height + height / 2);
    std::vector<uint8_t> data = rg.generate_random_1d<uint8_t>(data_size, 0, 255);

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

    auto input = input_layout("input", { { 1, height, width, 1} , data_types::u8, format::nv12 });
    auto input2 = input_layout("input2", { { 1, height / 2, width / 2, 2}, data_types::u8, format::nv12 });
    auto input_memory = engine->share_image(input.layout, nv12_image_plane_y);
    auto input_memory2 = engine->share_image(input2.layout, nv12_image_plane_uv);

    topology topology;
    topology.add(input);
    topology.add(input2);
    topology.add(convert_color("convert_color",
                               { input_info("input"), input_info("input2") },
                               cldnn::convert_color::color_format::NV12,
                               cldnn::convert_color::color_format::RGB,
                               cldnn::convert_color::memory_type::image));

    network network(*engine, topology, get_test_default_config(*engine));
    network.set_input_data("input", input_memory);
    network.set_input_data("input2", input_memory2);

    auto outputs = network.execute();

    std::vector<uint8_t> reference_results(width * height * 3);
    createReferenceDataNV12<uint8_t, uint8_t>(data.data(), data.data() + height * width, reference_results.data(),
                                              1, height, width, height * width, height * width / 2, true);

    auto output_prim = outputs.begin()->second.get_memory();
    cldnn::mem_lock<uint8_t> output_ptr(output_prim, get_test_stream());
    for (size_t i = 0; i < reference_results.size(); i++) {
        ASSERT_EQ(reference_results[i], output_ptr[i]);
    }
    checkStatus(clReleaseMemObject(nv12_image_plane_uv), "clReleaseMemObject");
    checkStatus(clReleaseMemObject(nv12_image_plane_y), "clReleaseMemObject");
}

TEST(convert_color, nv12_to_rgb_single_plane_surface_u8) {
    tests::random_generator rg(GET_SUITE_NAME);
    int width = 224;
    int height = 448;
    int input_height = height + height / 2;

    device_query query(engine_types::ocl, runtime_types::ocl);
    auto devices = query.get_available_devices();
    ASSERT_TRUE(!devices.empty());
    auto iter = devices.find(std::to_string(device_query::device_id));
    auto& device = iter != devices.end() ? iter->second : devices.begin()->second;
    auto engine = engine::create(engine_types::ocl, runtime_types::ocl, device);
    auto ocl_instance = std::make_shared<OpenCL>(std::dynamic_pointer_cast<ocl::ocl_device>(device)->get_device());

    if (!engine->get_device_info().supports_image) {
        GTEST_SKIP() << "Device doesn't support images";
    }
    cl_int err;

    int data_size = width * (height + height / 2);
    std::vector<uint8_t> input_data = rg.generate_random_1d<uint8_t>(data_size, 0, 255);

    cl_image_format image_format;
    image_format.image_channel_order = CL_R;
    image_format.image_channel_data_type = CL_UNORM_INT8;
    cl_image_desc image_desc = { CL_MEM_OBJECT_IMAGE2D, (size_t)width, (size_t)input_height, 0,
                                 0, 0, 0, 0, 0, { nullptr } };

    cl_mem nv12_image = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
    checkStatus(err, "Creating nv12 image failed");

    size_t origin[3] = { 0, 0, 0 };
    size_t y_region[3] = { (size_t)width, (size_t)input_height, 1 };

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), nv12_image, true, origin, y_region, 0, 0, &input_data[0], 0, nullptr, nullptr);
    checkStatus(err, "Writing nv12 image failed");

    auto input = input_layout("input", {{ 1, input_height, width, 1 }, data_types::u8, format::nv12});
    auto input_memory = engine->share_image(input.layout, nv12_image);

    topology topology;
    topology.add(input);
    topology.add(convert_color("convert_color",
                               { input_info("input") },
                               cldnn::convert_color::color_format::NV12,
                               cldnn::convert_color::color_format::RGB,
                               cldnn::convert_color::memory_type::image));

    network network(*engine, topology, get_test_default_config(*engine));
    network.set_input_data("input", input_memory);

    auto outputs = network.execute();

    std::vector<uint8_t> reference_results(width * height * 3);
    createReferenceDataNV12<uint8_t, uint8_t>(input_data.data(), input_data.data() + height * width, reference_results.data(),
                                            1, height, width, input_height * width, input_height * width, true);

    auto output_prim = outputs.begin()->second.get_memory();
    cldnn::mem_lock<uint8_t> output_ptr(output_prim, get_test_stream());
    for (size_t i = 0; i < reference_results.size(); i++) {
        ASSERT_EQ(reference_results[i], output_ptr[i]);
    }
    checkStatus(clReleaseMemObject(nv12_image), "clReleaseMemObject");
}

template <typename T>
std::tuple<T, T, T> yuv_pixel_to_rgb(float y_val, float u_val, float v_val) {
    auto c = y_val - 16.f;
    auto d = u_val - 128.f;
    auto e = v_val - 128.f;
    auto clip = [](float a) -> T {
        if (std::is_integral<T>()) {
            return static_cast<T>(std::min(std::max(std::round(a), 0.f), 255.f));
        } else {
            return static_cast<T>(std::min(std::max(a, 0.f), 255.f));
        }
    };
    auto b = clip(1.164f * c + 2.018f * d);
    auto g = clip(1.164f * c - 0.391f * d - 0.813f * e);
    auto r = clip(1.164f * c + 1.596f * e);
    return std::tuple<T, T, T>{r, g, b};
}

template <typename T, typename U>
void createReferenceDataI420(const T* arg_y, const T* arg_u, const T* arg_v, U* out_ptr,
                             size_t batch_size, size_t image_h, size_t image_w,
                             size_t stride_y, size_t stride_uv, bool rgb_color_format) {
    for (size_t batch = 0; batch < batch_size; ++batch) {
        U* out = out_ptr + batch * image_w * image_h * 3;
        auto y_ptr = arg_y + batch * stride_y;
        auto u_ptr = arg_u + batch * stride_uv;
        auto v_ptr = arg_v + batch * stride_uv;
        for (size_t h = 0; h < image_h; ++h) {
            for (size_t w = 0; w < image_w; ++w) {
                auto y_index = h * image_w + w;
                auto y_val = static_cast<float>(y_ptr[y_index]);
                auto uv_index = (h / 2) * (image_w / 2) + (w / 2);
                auto u_val = static_cast<float>(u_ptr[uv_index]);
                auto v_val = static_cast<float>(v_ptr[uv_index]);
                T r, g, b;
                std::tie(r, g, b) = yuv_pixel_to_rgb<U>(y_val, u_val, v_val);
                if (rgb_color_format) {
                    out[y_index * 3] = r;
                    out[y_index * 3 + 1] = g;
                    out[y_index * 3 + 2] = b;
                } else {
                    out[y_index * 3] = b;
                    out[y_index * 3 + 1] = g;
                    out[y_index * 3 + 2] = r;
                }
            }
        }
    }
}

TEST(convert_color, i420_to_rgb_three_planes_buffer_fp32) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    int width = 224;
    int height = 448;

    auto input_y = engine.allocate_memory({ { 1, height, width, 1 }, data_types::f32, format::bfyx });
    auto input_u = engine.allocate_memory({ { 1, height / 2 , width / 2, 1 }, data_types::f32, format::bfyx });
    auto input_v = engine.allocate_memory({ { 1, height / 2 , width / 2, 1 }, data_types::f32, format::bfyx });

    std::vector<float> input_y_data = rg.generate_random_1d<float>(width * height, 0, 255);
    std::vector<float> input_u_data = rg.generate_random_1d<float>(width * height / 4, 0, 255);
    std::vector<float> input_v_data = rg.generate_random_1d<float>(width * height / 4, 0, 255);

    set_values(input_y, input_y_data);
    set_values(input_u, input_u_data);
    set_values(input_v, input_v_data);

    topology topology;
    topology.add(input_layout("input_y", input_y->get_layout()));
    topology.add(input_layout("input_u", input_u->get_layout()));
    topology.add(input_layout("input_v", input_v->get_layout()));
    topology.add(convert_color("convert_color",
                               { input_info("input_y"), input_info("input_u"), input_info("input_v") },
                               cldnn::convert_color::color_format::I420,
                               cldnn::convert_color::color_format::RGB,
                               cldnn::convert_color::memory_type::buffer));

    network network(engine, topology, get_test_default_config(engine));
    network.set_input_data("input_y", input_y);
    network.set_input_data("input_u", input_u);
    network.set_input_data("input_v", input_v);

    auto outputs = network.execute();

    std::vector<float> ref_res(width * height * 3);
    createReferenceDataI420<float, float>(input_y_data.data(), input_u_data.data(), input_v_data.data(), ref_res.data(),
                                          1, height, width, height * width, height * width / 2, true);
    auto output = outputs.at("convert_color").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());

    for (size_t i = 0; i < ref_res.size(); ++i) {
        ASSERT_NEAR(ref_res[i], output_ptr[i], 1.001f);
    }
}

template <typename T>
void test_convert_color_i420_to_rgb_three_planes_surface_u8(bool is_caching_test) {
    tests::random_generator rg(GET_SUITE_NAME);
    int width = 224;
    int height = 448;

    device_query query(engine_types::ocl, runtime_types::ocl);
    auto devices = query.get_available_devices();
    ASSERT_TRUE(!devices.empty());
    auto iter = devices.find(std::to_string(device_query::device_id));
    auto& device = iter != devices.end() ? iter->second : devices.begin()->second;
    auto engine = engine::create(engine_types::ocl, runtime_types::ocl, device);
    auto ocl_instance = std::make_shared<OpenCL>(std::dynamic_pointer_cast<ocl::ocl_device>(device)->get_device());

    if (!engine->get_device_info().supports_image) {
        GTEST_SKIP() << "Device doesn't support images";
    }

    int data_size = width * (height + height / 2);
    std::vector<uint8_t> data = rg.generate_random_1d<uint8_t>(data_size, 0, 255);

    cl_int err;
    cl_image_format image_format;
    image_format.image_channel_order = CL_R;
    image_format.image_channel_data_type = CL_UNORM_INT8;
    cl_image_desc image_desc = { CL_MEM_OBJECT_IMAGE2D, (size_t)width, (size_t)height, 0,
                                 0, 0, 0, 0, 0, { nullptr } };

    cl_mem i420_image_plane_y = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
    checkStatus(err, "Creating i420 image plane_y failed");

    image_desc.image_width = width / 2;
    image_desc.image_height = height / 2;

    cl_mem i420_image_plane_u = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
    checkStatus(err, "Creating i420 image plane_u failed");

    cl_mem i420_image_plane_v = clCreateImage(ocl_instance->_context.get(), CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
    checkStatus(err, "Creating i420 image plane_v failed");

    size_t origin[3] = { 0, 0, 0 };
    size_t y_region[3] = { (size_t)width, (size_t)height, 1 };
    size_t uv_region[3] = { (size_t)width / 2, (size_t)height / 2, 1 };

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), i420_image_plane_y, true, origin, y_region, 0, 0, &data[0], 0, nullptr, nullptr);
    checkStatus(err, "Writing i420 image plane_y failed");

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), i420_image_plane_u, true, origin, uv_region, 0, 0, &data[width * height], 0, nullptr, nullptr);
    checkStatus(err, "Writing i420 image plane_u failed");

    err = clEnqueueWriteImage(ocl_instance->_queue.get(), i420_image_plane_v, true, origin, uv_region, 0, 0, &data[width * (height + height / 4)], 0, nullptr, nullptr);
    checkStatus(err, "Writing i420 image plane_v failed");

    auto input = input_layout("input", { { 1, height, width, 1 }, data_types::u8, format::nv12, });
    auto input2 = input_layout("input2", { { 1, height / 2, width / 2, 1 }, data_types::u8, format::nv12 });
    auto input3 = input_layout("input3", { { 1, height / 2, width / 2, 1 }, data_types::u8, format::nv12 });

    auto input_memory = engine->share_image(input.layout, i420_image_plane_y);
    auto input_memory2 = engine->share_image(input2.layout, i420_image_plane_u);
    auto input_memory3 = engine->share_image(input3.layout, i420_image_plane_v);

    topology topology;
    topology.add(input);
    topology.add(input2);
    topology.add(input3);
    topology.add(convert_color("convert_color",
                               { input_info("input"), input_info("input2"), input_info("input3") },
                               cldnn::convert_color::color_format::I420,
                               cldnn::convert_color::color_format::RGB,
                               cldnn::convert_color::memory_type::image));

    cldnn::network::ptr network = get_network(*engine, topology, get_test_default_config(*engine), get_test_stream_ptr(), is_caching_test);

    network->set_input_data("input", input_memory);
    network->set_input_data("input2", input_memory2);
    network->set_input_data("input3", input_memory3);

    auto outputs = network->execute();

    std::vector<uint8_t> reference_results(width * height * 3);
    createReferenceDataI420<T, uint8_t>(data.data(), data.data() + height * width, data.data() + width * (height + height / 4), reference_results.data(),
                                            1, height, width, height * width, height * width / 2, true);

    auto output_prim = outputs.begin()->second.get_memory();
    cldnn::mem_lock<uint8_t> output_ptr(output_prim, get_test_stream());
    for (size_t i = 0; i < reference_results.size(); i++) {
        ASSERT_EQ(reference_results[i], output_ptr[i]) << " i = " << i;
    }
    checkStatus(clReleaseMemObject(i420_image_plane_y), "clReleaseMemObject");
    checkStatus(clReleaseMemObject(i420_image_plane_u), "clReleaseMemObject");
    checkStatus(clReleaseMemObject(i420_image_plane_v), "clReleaseMemObject");
}

TEST(convert_color, i420_to_rgb_three_planes_surface_u8) {
    test_convert_color_i420_to_rgb_three_planes_surface_u8<uint8_t>(false);
}

TEST(export_import_convert_color, i420_to_rgb_three_planes_surface_u8) {
    test_convert_color_i420_to_rgb_three_planes_surface_u8<uint8_t>(true);
}
