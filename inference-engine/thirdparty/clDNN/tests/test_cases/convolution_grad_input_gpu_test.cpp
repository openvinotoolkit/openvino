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

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <gtest/gtest.h>
#include "api/memory.hpp"
#include <api/input_layout.hpp>
#include "api/convolution_grad_input.hpp"
#include <api/data.hpp>
#include <api/topology.hpp>
#include <api/network.hpp>
#include <api/engine.hpp>
#include "test_utils/test_utils.h"
#include "api/eltwise.hpp"

using namespace cldnn;
using namespace tests;

TEST(convolution_grad_input_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_stride2_pad1) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Output:
    //  -4    3.5    -0.5   21
    //   12  -18      4     -9

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::oiyx,{ 1, 1, 2, 2 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution_grad_input("deconv", "input", { "weights" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -4.f, 3.5f, 12.f, -18.f,
        -.5f, 21.f, 4.f, -8.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(convolution_grad_input_f32_fw_gpu, basic_wsiz2x2_in2x2x1x2_bfyx_stride2_pad1_output_size) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Output:
    //  -4    3.5    -0.5   21
    //   12  -18      4     -9

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::oiyx,{ 1, 1, 2, 2 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        convolution_grad_input("deconv", "input", { "weights" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }, { 2, 1, 2, 2 })
    );

    network network(engine, topology);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "deconv");

    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -4.f, 3.5f, 12.f, -18.f,
        -.5f, 21.f, 4.f, -8.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}

TEST(convolution_grad_input_f32_fw_gpu, DISABLED_basic_wsiz2x2_in2x2x1x2_bfyx_stride2_fusion) {
    //  Filter : 2x2
    //  Input  : 2x2x1x2
    //  Output : 2x2x1x2
    //  Stride : 2x2
    //
    //  Input:
    //  8  0.5    1   3
    //  6  9      2   4
    //
    //  Filter
    //  -2   2
    //   7  -0.5
    //
    //  Output:
    //  -4    3.5    -0.5   21
    //   12  -18      4     -9

    const auto& engine = get_test_engine();

    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 1, 2, 2 } });
    auto weights = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 2 } });
    auto scale_in = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto elt_data = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2, 2, 1, 2 } });

    set_values(input, { 8.f, 0.5f, 6.f, 9.f, 1.f, 3.f, 2.f, 4.f });
    set_values(weights, { -2.f, 2.f, 7.f, -0.5f });
    set_values(scale_in, { 1.0f });
    set_values(elt_data, { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f });

    topology topology(
        input_layout("input", input.get_layout()),
        data("weights", weights),
        data("scale_in", scale_in),
        data("elt_data", elt_data),
        convolution_grad_input("conv", "input", { "weights" }, { 1, 1, 2, 2 }, { 0, 0, -1, -1 }),
        eltwise("elt", "conv", "elt_data", eltwise_mode::sum),
        scale("scale", "elt", "scale_in")
    );

    build_options options;
    options.set_option(build_option::optimize_data(true));
       
    network network(engine, topology, options);
    network.set_input_data("input", input);

    auto outputs = network.execute();
    auto primitives = network.get_all_primitive_ids();
    auto exec_prim = network.get_executed_primitive_ids();
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "scale");
    EXPECT_TRUE(std::find(primitives.begin(), primitives.end(), "elt") == primitives.end());
    EXPECT_TRUE(std::find(exec_prim.begin(), exec_prim.end(), "elt") == exec_prim.end());
        
    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();

    std::vector<float> expected_output_vec = {
        -3.f, 5.5f, 14.f, -15.f,
        4.5f, 27.f, 10.f, -1.f
    };

    for (unsigned int i = 0; i < expected_output_vec.size(); i++)
    {
        EXPECT_FLOAT_EQ(expected_output_vec[i], output_ptr[i]);
    }
}