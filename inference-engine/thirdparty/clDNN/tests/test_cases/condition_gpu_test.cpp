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

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <gtest/gtest.h>

#include <api/CPP/engine.hpp>
#include <api/CPP/input_layout.hpp>
#include <api/CPP/memory.hpp>
#include <api/CPP/concatenation.hpp>
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/pooling.hpp>
#include <api/CPP/condition.hpp>
#include <api/CPP/softmax.hpp>
#include <api/CPP/scale.hpp>
#include <api/CPP/data.hpp>
#include "test_utils/test_utils.h"

#include <cstddef>


using namespace cldnn;
using namespace ::tests;


bool is_output_equal(const cldnn::memory& mem, const std::vector<float>& ref)
{
    auto ptr = mem.pointer<float>();
    for (size_t i = 0; i < mem.get_layout().count(); i++)
    {
        if (!are_equal(ptr[i], ref[i])) return false;
    }
    return true;
}

topology generate_simple_branch (bool branch_true_false, const primitive_id& input_id)
{
    topology branch;
    if (branch_true_false)
    {
        branch.add(
            pooling(input_id + "_when_true", input_id, cldnn::pooling_mode::max, { 0, 0, 2, 1 }, { 0, 0, 2, 1 })
        );
    }
    else
    {
        branch.add(
            pooling(input_id + "_when_false", input_id, cldnn::pooling_mode::average, { 0, 0, 2, 1 }, { 0, 0, 2, 1 })
        );
    }
    return branch;
}


TEST(condition_gpu, basic_equal_comp) {
    const auto& engine = get_test_engine();
    build_options bs;
    bs.set_option(build_option::optimize_data(true));
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
    auto compare = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto scale_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    
    topology branch_true = generate_simple_branch(true, "condi");
    topology branch_false = generate_simple_branch(false, "condi");

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        input_layout("compare", compare.get_layout())
    );    
    topology.add(
        input_layout("scale_data", scale_mem.get_layout())
    );
    topology.add(
        condition("condi", "input", branch_true, branch_false, "compare", cond_functions::EQUAL)
    );  
    topology.add(
        scale("output", "condi", "scale_data")
    );

    network net(engine, topology, bs);
    set_values(input, { 1.0f, 2.0f, 3.0f, 4.0f });
    set_values(scale_mem, { 10.0f });
    net.set_input_data("input", input);
    net.set_input_data("scale_data", scale_mem);

    decltype(net.execute()) out;

    //WHEN TRUE
    set_values(compare, { 1.0f });
    net.set_input_data("compare", compare);
    out = net.execute();
    auto out_data_true = out.at("output").get_memory();
    EXPECT_TRUE(is_output_equal(out_data_true, {20.0f, 40.0f}));

    //WHEN FALSE
    set_values(compare, { 4.0f });
    net.set_input_data("compare", compare);
    out = net.execute();
    auto out_data_false = out.at("output").get_memory();
    EXPECT_TRUE(is_output_equal(out_data_false, { 15.0f, 35.0f }));

}

TEST(condition_gpu, basic_range_equal_comp) {

    const auto& engine = get_test_engine();
    build_options bs;
    bs.set_option(build_option::optimize_data(true));
    auto input0 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });

    auto compare = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 3, 1 } });

    topology branch_true = generate_simple_branch(true, "condi");
    topology branch_false = generate_simple_branch(false, "condi");

    topology topology;
    topology.add(
        input_layout("input0", input0.get_layout())
    );
    topology.add(
        input_layout("input1", input1.get_layout())
    );
    topology.add(
        input_layout("compare", compare.get_layout())
    );
    topology.add(
        concatenation("concat", { "input0", "input1" }, concatenation::along_x)
    );
    topology.add( 
        condition("condi", "concat", branch_true, branch_false, "compare", cond_functions::EQUAL)
    );

    std::vector<float> input0_data = {
        1, 2, 3, 4
    };
    std::vector<float> input1_data = {
        5, 6, 7, 8
    };
    std::vector<float> compare_data_true = {
        1, 2, 3
    };
    std::vector<float> pooling_when_true_data = {
        2, 4, 6, 8
    };
    std::vector<float> compare_data_false = {
        1, 2, 10
    };
    std::vector<float> pooling_when_false_data = {
        1.5, 3.5, 5.5, 7.5
    };

    set_values(input0, input0_data);
    set_values(input1, input1_data);
    network net(engine, topology, bs);
    net.set_input_data("input0", input0);
    net.set_input_data("input1", input1);

    decltype(net.execute()) outputs;

    //CHECK TRUE
    set_values(compare, compare_data_true);
    net.set_input_data("compare", compare);
    outputs = net.execute();

    auto out_data_true = outputs.at("condi").get_memory();
    EXPECT_TRUE(is_output_equal(out_data_true, pooling_when_true_data));

    //CHECK FALSE
    set_values(compare, compare_data_false);
    net.set_input_data("compare", compare);
    outputs = net.execute();

    auto out_data_false = outputs.at("condi").get_memory();
    EXPECT_TRUE(is_output_equal(out_data_false, pooling_when_false_data));
}

std::pair<std::vector<float>, std::vector<float>> get_values_to_compare(const cldnn::tensor& offset, const cldnn::tensor& range, const std::vector<float>& values, const cldnn::layout& input_lay, const cond_functions& func)
{
    std::vector<float> ret_true;
    std::vector<float> ret_false;
    auto mem_desc = generic_test::get_linear_memory_desc(input_lay);
    for (int32_t b = 0; b < range.batch[0]; b++)
    {
        for (int32_t f = 0; f < range.feature[0]; f++)
        {
            for (int32_t y = 0; y < range.spatial[1]; y++)
            {
                for (int32_t x = 0; x < range.spatial[0]; x++)
                {
                    auto linear_idx = generic_test::get_linear_index(
                        input_lay,
                        offset.batch[0] + b,
                        offset.feature[0] + f,
                        offset.spatial[1] + y,
                        offset.spatial[0] + x,
                        mem_desc);

                    switch (func)
                    {
                    case cond_functions::EQUAL:
                        ret_true.push_back(values.at(linear_idx));
                        ret_false.push_back(-1.0f);
                        break;
                    case cond_functions::GREATER: 
                        ret_true.push_back(values.at(linear_idx) - 1.0f);
                        ret_false.push_back(99.0f);
                        break;
                    case cond_functions::LESS: 
                        ret_true.push_back(values.at(linear_idx) + 1.0f);
                        ret_false.push_back(-1.0f);
                        break;
                    }
                }
            }
        }
    }
    return { ret_true, ret_false };
}

TEST(DISABLED_condition_gpu, generic_test_true_false) {

    const auto& engine = get_test_engine();
    build_options bs;
    bs.set_option(build_option::optimize_data(true));
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 5, 2, 5, 1 } });
    std::vector<float> input_data(50);
    std::iota(input_data.begin(), input_data.end(), 0.0f);

    std::vector<cond_functions> functions = {
        cond_functions::EQUAL,
        cond_functions::GREATER,
        cond_functions::LESS,
    };

    // ranges, with data when condition is true or false
    std::vector<cldnn::tensor> ranges = {
        {1, 1, 1, 1},
        {1, 1, 3, 1},
        {2, 1, 1, 1},
        {2, 1, 1, 1}
    };

    std::vector<cldnn::tensor> offsets = {
        { 0, 0, 0, 0},
        { 0, 0, 1, 0},
        { 0, 0, 2, 0},
        { 2, 0, 0, 0},
        { 2, 1, 1, 0}
    };

    std::vector<float> pooling_when_true_data = {
        2, 4, 7, 9, 12, 14, 17,
        19, 22, 24, 27, 29, 32,
        34, 37, 39, 42, 44, 47, 49
    };

    std::vector<float> pooling_when_false_data = {
        1, 3, 6, 8, 11, 13, 16,
        18, 21, 23, 26, 28, 31,
        33, 36, 38, 41, 43, 46, 48
    };

    for (auto const& func : functions)
    {
        for (auto const& range : ranges)
        {
            for (auto const& offset : offsets)
            {
                auto comp_values = get_values_to_compare(offset, range, input_data, input.get_layout(), func);
                auto comp_values_true = comp_values.first;
                auto comp_values_false = comp_values.second;

                auto compare = memory::allocate(engine, { data_types::f32, format::bfyx, range });

                topology branch_true;
                topology branch_false;
                branch_true.add(
                    pooling("pooling_when_true", "condi", cldnn::pooling_mode::max, { 1, 1, 3, 1 }, { 1, 1, 2, 1 })
                );
                branch_false.add(
                    pooling("pooling_when_false", "condi", cldnn::pooling_mode::average, { 1, 1, 3, 1 }, { 1, 1, 2, 1 })
                );

                topology topology;
                topology.add(
                    input_layout("input", input.get_layout())
                );
                topology.add(
                    input_layout("compare", compare.get_layout())
                );
                topology.add(
                    condition("condi", "input", branch_true, branch_false, "compare", func, offset)
                );

                set_values(input, input_data);
                network net(engine, topology, bs);
                net.set_input_data("input", input);

                decltype(net.execute()) outputs;

                //CHECK TRUE
                set_values(compare, comp_values_true);
                net.set_input_data("compare", compare);
                outputs = net.execute();

                auto out_data_true = outputs.at("condi").get_memory();
                EXPECT_TRUE(is_output_equal(out_data_true, pooling_when_true_data));

                //CHECK FALSE
                set_values(compare, comp_values_false);
                net.set_input_data("compare", compare);
                outputs = net.execute();

                auto out_data_false = outputs.at("condi").get_memory();
                EXPECT_TRUE(is_output_equal(out_data_false, pooling_when_false_data));

            }
        }
    }
}

TEST(condition_gpu, basic_stacked_ifs) {

    /*   
        <prims...>
        <if>
        <...>
        <end_if>
        <...>
        <if>
        <...>
        <end_if>
        <prims...>    
    */
    const auto& engine = get_test_engine();
    build_options bs;
    bs.set_option(build_option::optimize_data(true));
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
    auto compare = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto compare2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });


    topology condi_1_true = generate_simple_branch(true, "condi");
    topology condi_1_false = generate_simple_branch(false, "condi");
    topology condi_2_true;
    condi_2_true.add(
        activation("activ_when_true", "condi2", cldnn_activation_func::activation_log2)
    );
    topology condi_2_false;
    condi_2_false.add(
        activation("activ_when_false", "condi2", cldnn_activation_func::activation_relu)
    );

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        input_layout("compare", compare.get_layout())
    );
    topology.add(
        condition("condi", "input", condi_1_true, condi_1_false, "compare", cond_functions::EQUAL)
    );
    topology.add(
        input_layout("compare2", compare2.get_layout())
    );
    topology.add(
        condition("condi2", "condi", condi_2_true, condi_2_false, "compare2", cond_functions::GREATER)
    );

    std::vector<float> input_data = {
        1, 2, 3, 4
    };
    std::vector<float> compare_data = {
        1
    };
    std::vector<float> compare_2_data = {
        0.0f, 0.0f
    };
    set_values(input, input_data);
    set_values(compare, compare_data);
    set_values(compare2, compare_2_data);

    network net(engine, topology, bs);
    net.set_input_data("input", input);
    net.set_input_data("compare", compare);
    net.set_input_data("compare2", compare2);
    auto outputs = net.execute();

    auto out_data = outputs.at("condi2").get_memory();
    EXPECT_TRUE(is_output_equal(out_data, {1.0f, 2.0f}));
}

TEST(condition_gpu, basic_nested_ifs) {

    /*
    <prims...>
    <if 0>
    <...>
    <if 1>
    <...>
    <end_if 1>
    <...>
    <end_if 0>
    <prims...>
    */
    const auto& engine = get_test_engine();
    build_options bs;
    bs.set_option(build_option::optimize_data(true));
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
    auto compare = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    auto compare2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 2, 1 } });
    auto scale_5_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    set_values(scale_5_mem, { 5.0f });
    auto scale_10_mem = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    set_values(scale_10_mem, { 10.0f });


    topology nested_true;
    {
        nested_true.add(scale("scale_5", "condi_nested", "scale_5_data"),
            data("scale_5_data", scale_5_mem));
    }
    topology nested_false;
    {
        nested_false.add(scale("scale_10", "condi_nested", "scale_10_data"),
            data("scale_10_data", scale_10_mem));
    }

    topology branch_true;
    branch_true.add(
        pooling("pooling_when_true", "condi", cldnn::pooling_mode::max, { 0, 0, 2, 1 }, { 0, 0, 2, 1 })
    );
    branch_true.add(
        input_layout("compare2", compare2.get_layout())
    );

    branch_true.add(
        condition(
        "condi_nested",
        "pooling_when_true",
        nested_true,
        nested_false,
        "compare2",
        cond_functions::EQUAL)
    );

    topology branch_false;
    branch_false.add(
        pooling("pooling_when_false", "condi", cldnn::pooling_mode::average, { 0, 0, 2, 1 }, { 0, 0, 2, 1 })
    );

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );

    topology.add(
        input_layout("compare", compare.get_layout())
    );

    topology.add(
        condition("condi", "input", branch_true, branch_false, "compare", cond_functions::EQUAL)
    );

    std::vector<float> input_data = {
        1.0f, 2.0f, 3.0f, 4.0f
    };
    std::vector<float> compare_data = {
        1.0f
    };
    std::vector<float> compare_2_data = {
        2.0f, 4.0f
    };
    set_values(input, input_data);
    set_values(compare, compare_data);
    set_values(compare2, compare_2_data);

    network net(engine, topology, bs);
    net.set_input_data("input", input);
    net.set_input_data("compare", compare);
    net.set_input_data("compare2", compare2);
    auto outputs = net.execute();

    auto out_data = outputs.at("condi").get_memory();
    EXPECT_TRUE(is_output_equal(out_data, { 10.0f, 20.0f }));
}


TEST(condition_gpu, negative_compare_wrong_layout) {
    const auto& engine = get_test_engine();
    build_options bs;
    bs.set_option(build_option::optimize_data(true));
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
    auto compare = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 5, 1 } });

    topology branch_true = generate_simple_branch(true, "condi");
    topology branch_false = generate_simple_branch(false, "condi");

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        input_layout("compare", compare.get_layout())
    );
    topology.add(
        condition("condi", "input", branch_true, branch_false, "compare", cond_functions::EQUAL)
    );

    EXPECT_ANY_THROW(network net(engine, topology, bs););
}

TEST(condition_gpu, negative_too_big_offset) {
    const auto& engine = get_test_engine();
    build_options bs;
    bs.set_option(build_option::optimize_data(true));
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
    auto compare = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 3, 1 } });

    topology branch_true = generate_simple_branch(true, "condi");
    topology branch_false = generate_simple_branch(false, "condi");

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        input_layout("compare", compare.get_layout())
    );
    topology.add(
        condition("condi", "input", branch_true, branch_false, "compare", cond_functions::EQUAL, {1, 1, 2, 1})
    );

    EXPECT_ANY_THROW(network net(engine, topology, bs););
}

TEST(condition_gpu, negative_not_same_layouts) {
    const auto& engine = get_test_engine();
    build_options bs;
    bs.set_option(build_option::optimize_data(true));
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
    auto compare = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });
    
    topology branch_true;
    branch_true.add(
        pooling("pooling_when_true", "condi", cldnn::pooling_mode::max, { 0, 0, 2, 1 }, { 0, 0, 2, 1 })
    );

    topology branch_false;
    branch_false.add(
        pooling("pooling_when_false", "condi", cldnn::pooling_mode::max, { 0, 0, 4, 1 }, { 0, 0, 4, 1 })
    );

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        input_layout("compare", compare.get_layout())
    );
    topology.add(
        condition("condi", "input", branch_true, branch_false, "compare", cond_functions::EQUAL)
    );

    EXPECT_ANY_THROW(network net(engine, topology, bs););
}

TEST(condition_gpu, negative_same_names_within_different_networks) {
    const auto& engine = get_test_engine();
    build_options bs;
    bs.set_option(build_option::optimize_data(true));
    auto input = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 4, 1 } });
    auto compare = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1, 1, 1, 1 } });

    topology branch_true;
    branch_true.add(
        pooling("pooling_check_name", "condi", cldnn::pooling_mode::max, { 0, 0, 2, 1 }, { 0, 0, 2, 1 })
    );

    topology branch_false;
    branch_false.add(
        pooling("pooling_when_false", "condi", cldnn::pooling_mode::max, { 0, 0, 2, 1 }, { 0, 0, 2, 1 })
    );

    topology topology;
    topology.add(
        input_layout("input", input.get_layout())
    );
    topology.add(
        input_layout("compare", compare.get_layout())
    );
    topology.add(
        condition("condi", "input", branch_true, branch_false, "compare", cond_functions::EQUAL)
    );
    topology.add(
        pooling("pooling_check_name", "condi", cldnn::pooling_mode::max, { 0, 0, 2, 1 }, { 0, 0, 2, 1 })
    );
    
    EXPECT_ANY_THROW(network net(engine, topology, bs););
}