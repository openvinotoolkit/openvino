/*
// Copyright (c) 2016 Intel Corporation
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

#include <gtest/gtest.h>
#include "api/engine.hpp"
#include "test_utils/test_utils.h"
#include "api/network.hpp"
#include "api/topology.hpp"
#include "api/input_layout.hpp"
#include "api/activation.hpp"
#include "api/cldnn.hpp"

#include "test_utils.h"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120

#if defined __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-braces"
#elif defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include <cl2_wrapper.h>

using namespace cldnn;

class user_gpu_toolkit
{
public:
    user_gpu_toolkit()
    {
        get_platform_and_device(get_plaftorm());
        create_context_from_one_device();
    }

    cl_context get_gpu_context() const { return _gpu_context; }

private:
    cl_platform_id _platform_id;
    cl_device_id _gpu_device;
    cl_context _gpu_context;

    void create_context_from_one_device()
    {
        cl_int error = 0;
        _gpu_context = clCreateContext(0, 1, &_gpu_device, 0, 0, &error);
        if (error != CL_SUCCESS)
        {
            throw std::runtime_error("error creating context");
        }
    }

    cl_platform_id get_plaftorm()
    {
        static constexpr auto INTEL_PLATFORM_VENDOR = "Intel(R) Corporation";
        cl_uint n = 0;
        cl_int err = clGetPlatformIDs(0, NULL, &n);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("clGetPlatformIDs error " + std::to_string(err));
        }

        // Get platform list
        std::vector<cl_platform_id> platform_ids(n);
        err = clGetPlatformIDs(n, platform_ids.data(), NULL);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("clGetPlatformIDs error " + std::to_string(err));
        }

        // find Intel platform
        for (auto& id : platform_ids) {
            size_t infoSize;
            err = clGetPlatformInfo(id, CL_PLATFORM_VENDOR, 0, NULL, &infoSize);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("clGetPlatformInfo error " + std::to_string(err));
            }

            std::vector<char> tmp(infoSize);

            err = clGetPlatformInfo(id, CL_PLATFORM_VENDOR, infoSize, tmp.data(), NULL);
            if (err != CL_SUCCESS) {
                throw std::runtime_error("clGetPlatformInfo error " + std::to_string(err));
            }

            std::string vendor_id(tmp.data());
            if (vendor_id == std::string(INTEL_PLATFORM_VENDOR))
                return id;
        }
        return static_cast<cl_platform_id>(nullptr);
    }

    void get_platform_and_device(cl_platform_id platform_id)
    {
        _platform_id = platform_id;
        cl_int err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &_gpu_device, 0);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("clGetDeviceIDs error " + std::to_string(err));
        }
    }
};

TEST(gpu_engine, DISABLED_user_context)
{
    user_gpu_toolkit gpu_toolkit;
    cl_context user_context = gpu_toolkit.get_gpu_context();

    device_query query(static_cast<void*>(user_context));
    auto devices = query.get_available_devices();

    //[0] Check if the user engine config works.
    auto engine_config = cldnn::engine_configuration(false, false, false, "", "", true, "", "", cldnn::priority_mode_types::disabled, cldnn::throttle_mode_types::disabled, true, 1);

    //[1]Check if the engine creation works.
    engine engine(devices.begin()->second, engine_config);
    auto info = engine.get_info();
    EXPECT_GT(info.cores_count, 0u);
    EXPECT_GT(info.core_frequency, 0u);

    //[2]Now check if the queues works (run simple network).
    topology topo;
    auto inp_lay = cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, { 1,1,2,2 });
    auto input_mem = cldnn::memory::allocate(engine, inp_lay);
    tests::set_values<float>(input_mem, { 1.0f, 2.0f, 3.0f, 4.0f });
    auto inp = input_layout("input", inp_lay);
    auto activ = activation("this_needs_queue", "input", activation_func::abs);
    topo.add(inp, activ);
    network net(engine, topo);

    net.set_input_data("input", input_mem);
    auto out = net.execute();
    auto out_ptr = out.at("this_needs_queue").get_memory().pointer<float>();
    EXPECT_EQ(out.size(), size_t(1));
    for (uint32_t i = 0; i < 4; i++)
        EXPECT_EQ(out_ptr[i], float(i + 1));
}

void execute_simple_topology(cldnn::engine& engine) {
    auto batch_num = 1;
    auto feature_num = 4;
    auto x_size = 1;
    auto y_size = 1;
    auto input_tensor = cldnn::tensor(cldnn::spatial(x_size, y_size), cldnn::feature(feature_num), cldnn::batch(batch_num));
    auto topo = cldnn::topology(
        cldnn::input_layout("input", { cldnn::data_types::f32, cldnn::format::bfyx, input_tensor  }),
        cldnn::activation("relu", "input", cldnn::activation_func::relu));

    cldnn::network net(engine, topo);
    auto input_mem = memory::allocate(engine, { data_types::f32, format::bfyx, input_tensor });
    tests::set_values(input_mem, { -1.f, 2.f, -3.f, 4.f });
    net.set_input_data("input", input_mem);
    auto outs = net.execute();
    auto output = outs.at("relu");
    auto out_ptr = output.get_memory().pointer<float>();
    ASSERT_EQ(out_ptr[0], 0.0f);
    ASSERT_EQ(out_ptr[1], 2.0f);
    ASSERT_EQ(out_ptr[2], 0.0f);
    ASSERT_EQ(out_ptr[3], 4.0f);
}


TEST(gpu_device_query, get_device_info)
{
    cldnn::device_query query;
    auto devices = query.get_available_devices();
    auto device_id = devices.begin()->first;
    auto device = devices.begin()->second;
    auto device_info = device.get_info();

    //check key and few members, so we know that device info was returned properly
    ASSERT_EQ(device_id, "0");
    ASSERT_GT(device_info.cores_count, 0u);
    ASSERT_GT(device_info.core_frequency, 0u);
    ASSERT_NE(device_info.dev_name, "");
    ASSERT_NE(device_info.driver_version, "");
}


TEST(gpu_device_query, get_engine_info)
{
    const auto& engine = tests::get_test_engine();
    auto info = engine.get_info();
    EXPECT_GT(info.cores_count, 0u);
    EXPECT_GT(info.core_frequency, 0u);
}


TEST(gpu_device_query, simple)
{
    cldnn::device_query query;
    auto devices = query.get_available_devices();
    auto device = devices.begin()->second;

    cldnn::engine eng(device);
    //check if simple execution was finished correctly
    execute_simple_topology(eng);
}

TEST(gpu_device_query, DISABLED_release_query)
{
    cldnn::device_query query;
    auto devices = query.get_available_devices();
    auto device = devices.begin()->second;

    //destroy query
    query.~device_query();
    //create engine
    cldnn::engine eng(device);
    //check if simple execution was finished correctly
    execute_simple_topology(eng);
}

TEST(gpu_device_query, DISABLED_release_device)
{
    cldnn::device_query query;
    auto devices = query.get_available_devices();
    auto device = devices.begin()->second;

    //destroy query
    query.~device_query();
    //create engine
    cldnn::engine eng(device);
    //destroy device
    device.~device();
    //check if simple execution was finished correctly
    execute_simple_topology(eng);
}



