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
        return platform_ids[0];
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

TEST(gpu_engine, engine_info)
{
    const auto& engine = tests::get_test_engine();
    auto info = engine.get_info();
    EXPECT_GT(info.cores_count, 0u);
    EXPECT_GT(info.core_frequency, 0u);
}

TEST(gpu_engine, user_context)
{
    user_gpu_toolkit gpu_toolkit;
    cl_context user_context = gpu_toolkit.get_gpu_context();

    //[0] Check if the user engine config works.
    auto engine_config = cldnn::engine_configuration(false, false, false, "", "", true, "", "", cldnn::priority_mode_types::disabled, cldnn::throttle_mode_types::disabled, true, 1, &user_context);

    //[1]Check if the engine creation works.
    engine engine(engine_config);
    auto info = engine.get_info();
    EXPECT_GT(info.cores_count, 0u);
    EXPECT_GT(info.core_frequency, 0u);

    //[2]Now check if the queues works (run simple network).
    topology topo;
    auto inp_lay = cldnn::layout(cldnn::data_types::f32, cldnn::format::bfyx, { 1,1,2,2});
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
    for(uint32_t i = 0;i < 4; i++)
        EXPECT_EQ(out_ptr[i], float(i+1));
}
