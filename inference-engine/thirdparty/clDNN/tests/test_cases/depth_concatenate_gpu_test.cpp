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
#include "api/CPP/memory.hpp"
#include <api/CPP/input_layout.hpp>
#include "api/CPP/concatenation.hpp"
#include <api/CPP/topology.hpp>
#include <api/CPP/network.hpp>
#include <api/CPP/engine.hpp>
#include "test_utils/test_utils.h"

using namespace cldnn;
using namespace tests;

template<typename T>
std::vector<T> generate_random_input(size_t b, size_t f, size_t y, size_t x, int min, int max) {
    static std::default_random_engine generator(random_seed);
    int k = 8; // 1/k is the resolution of the floating point numbers
    std::uniform_int_distribution<int> distribution(k * min, k * max);
    std::vector<T> v(b*f*x*y);
    for (size_t i = 0; i < b*f*x*y; ++i) {
        v[i] = (T)distribution(generator);
        v[i] /= k;
    }
    return v;
}

TEST(depth_concatenate_f32_gpu, test01) {
    //  Input count : 2
    //  Input1 : 2x 1x1 x 2
    //  Input2 : 2x 1x1 x 3
    //
    //  Input1:
    //  0.5  0.7  :f0
    //  0.2  0.4  :f1
    //
    //  Input2:
    //  1    0.1  :f0
    //  0.3 -0.5  :f1
    //  0   -0.2  :f2
    //
    //  Output:
    //  0.5  0.7  :f0
    //  0.2  0.4  :f1
    //  1    0.1  :f2
    //  0.3 -0.5  :f3
    //  0   -0.2  :f4
    //

    engine engine;
    auto input1 = memory::allocate(engine, {data_types::f32, format::yxfb, { 2,2,1,1 }});
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2,3,1,1 }});

    set_values(input1, { 0.5f, 0.7f, 0.2f, 0.4f });
    set_values(input2, { 1.0f, 0.1f, 0.3f, -0.5f, 0.0f, -0.2f });

    topology topology;
    topology.add(input_layout("input1", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(concatenation("depth1", { "input1", "input2" }, concatenation::along_f));

    network network(engine, topology);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute({});
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "depth1");

    auto output = outputs.at("depth1").get_memory();

    auto output_ptr = output.pointer<float>();
    EXPECT_FLOAT_EQ(0.5f, output_ptr[0]);
    EXPECT_FLOAT_EQ(0.7f, output_ptr[1]);
    EXPECT_FLOAT_EQ(0.2f, output_ptr[2]);
    EXPECT_FLOAT_EQ(0.4f, output_ptr[3]);
    EXPECT_FLOAT_EQ(1.0f, output_ptr[4]);
    EXPECT_FLOAT_EQ(0.1f, output_ptr[5]);
    EXPECT_FLOAT_EQ(0.3f, output_ptr[6]);
    EXPECT_FLOAT_EQ(-0.5f, output_ptr[7]);
    EXPECT_FLOAT_EQ(0.0f, output_ptr[8]);
    EXPECT_FLOAT_EQ(-0.2f, output_ptr[9]);
}


TEST(depth_concatenate_f32_gpu, concat_basic_int8) {
    //  Input count : 2
    //  Input1 : 2x 1x1 x 2
    //  Input2 : 2x 1x1 x 3
    //
    //  Input1:
    //  2.5  3.7  :f0
    //  0.2  1.4  :f1
    //
    //  Input2:
    //  1    4.1  :f0
    // -4.3 -7.5  :f1
    //  0   -0.2  :f2
    //
    //  Output:
    //  2    3  :f0
    //  0    1  :f1
    //  1    4  :f2
    // -4   -7  :f3
    //  0    0  :f4
    //

    engine engine;
    auto input1 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2,2,1,1 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2,3,1,1 } });
    auto outs = { 2.0f, 3.0f, 0.0f, 1.0f, 1.0f, 4.0f, -4.0f, -7.0f, 0.0f, 0.0f };
    set_values(input1, { 2.5f, 3.7f, 0.2f, 1.4f });
    set_values(input2, { 1.0f, 4.1f, -4.3f, -7.5f, 0.0f, -0.2f });

    topology topology;
    topology.add(input_layout("input1", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(reorder("to_int1", "input1", { data_types::i8, format::yxfb,{ 2,2,1,1 } }));
    topology.add(reorder("to_int2", "input2", { data_types::i8, format::yxfb,{ 2,3,1,1 } }));
    topology.add(concatenation("depth1", { "to_int1", "to_int2" }, concatenation::along_f));
    topology.add(reorder("to_float", "depth1", { data_types::f32, format::yxfb,{ 2,5,1,1 } }));

    network network(engine, topology);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute({});
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "to_float");

    auto output = outputs.at("to_float").get_memory();

    auto output_ptr = output.pointer<float>();
    int ptr_cntr = 0;
    for (const auto& ref : outs)
    {
        EXPECT_FLOAT_EQ(ref, output_ptr[ptr_cntr++]);
    }
}

TEST(depth_concatenate_f32_gpu, test02) {
    //  Input count : 3 (yxfb, yxfb, bfyx)
    //  Input1 : 2x 1x1 x 2
    //  Input2 : 2x 1x1 x 3
    //  Input3 : 2x 1x1 x 3
    //
    //  Input1 (yxfb):
    //  0.5  0.7  :f0
    //  0.2  0.4  :f1
    //
    //  Input2 (yxfb):
    //  1    0.1  :f0
    //  0.3 -0.5  :f1
    //  0   -0.2  :f2
    //
    //  Input3 (bfyx):
    //  1    0.1  :f0
    //  0.3 -0.5  :f1
    //  0   -0.2  :f2
    //
    //  Output:
    //  0.5  0.7  :f0
    //  0.2  0.4  :f1
    //  1    0.1  :f2
    //  0.3 -0.5  :f3
    //  0   -0.2  :f4
    //  1    0.1  :f5
    //  0.3 -0.5  :f6
    //  0   -0.2  :f7
    //

    engine engine;
    auto input1 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2,2,1,1 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 2,3,1,1 } });
    auto input3 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 2,3,1,1 } });

    set_values(input1, { 0.5f, 0.7f, 0.2f, 0.4f });
    set_values(input2, { 1.0f, 0.1f, 0.3f, -0.5f, 0.0f, -0.2f });
    set_values(input3, { 1.0f, 0.3f, 0.0f, 0.1f, -0.5f, -0.2f });

    topology topology;
    topology.add(input_layout("input1", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(input_layout("input3", input3.get_layout()));
    topology.add(concatenation("depth1", { "input1", "input2", "input3" }, concatenation::along_f));

    network network(engine, topology);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);
    network.set_input_data("input3", input3);

    auto outputs = network.execute({});
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "depth1");

    auto output = outputs.at("depth1").get_memory();

    auto output_ptr = output.pointer<float>();
    EXPECT_FLOAT_EQ(0.5f, output_ptr[0]);
    EXPECT_FLOAT_EQ(0.7f, output_ptr[1]);
    EXPECT_FLOAT_EQ(0.2f, output_ptr[2]);
    EXPECT_FLOAT_EQ(0.4f, output_ptr[3]);
    EXPECT_FLOAT_EQ(1.0f, output_ptr[4]);
    EXPECT_FLOAT_EQ(0.1f, output_ptr[5]);
    EXPECT_FLOAT_EQ(0.3f, output_ptr[6]);
    EXPECT_FLOAT_EQ(-0.5f, output_ptr[7]);
    EXPECT_FLOAT_EQ(0.0f, output_ptr[8]);
    EXPECT_FLOAT_EQ(-0.2f, output_ptr[9]);
    EXPECT_FLOAT_EQ(1.0f, output_ptr[10]);
    EXPECT_FLOAT_EQ(0.1f, output_ptr[11]);
    EXPECT_FLOAT_EQ(0.3f, output_ptr[12]);
    EXPECT_FLOAT_EQ(-0.5f, output_ptr[13]);
    EXPECT_FLOAT_EQ(0.0f, output_ptr[14]);
    EXPECT_FLOAT_EQ(-0.2f, output_ptr[15]);
}

TEST(depth_concatenate_f32_gpu, test03_cascade_concat_opt) {
    //  Test for cascade concatenation optimization.
    //  Despite having concatenations one after another and connected to different non padded activation primitives,
    //  graph should remove all concatenations from execution.

    engine engine;
    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1,2,2,1 } });

    set_values(input1, { 16.0f, 32.0f, 128.0f, 256.0f });

    topology topology;
    topology.add(input_layout("input1", input1.get_layout()));
    topology.add(activation("relu1", "input1", activation_relu));
    topology.add(activation("relu2", "relu1", activation_sqrt));
    topology.add(concatenation("depth1", { "relu2", "relu1" }, concatenation::along_f));
    topology.add(activation("relu3", "depth1", activation_sqrt));
    topology.add(concatenation("depth2", { "relu3", "depth1" }, concatenation::along_f));
    topology.add(activation("relu4", "depth2", activation_sqrt));
    topology.add(concatenation("depth3", { "relu4", "depth2" }, concatenation::along_f));
    topology.add(activation("relu5", "depth3", activation_relu));

    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));
    network network(engine, topology, options);

    network.set_input_data("input1", input1);

    auto outputs = network.execute({});
    auto output_prim = outputs.begin()->second.get_memory();

    auto output_ptr = output_prim.pointer<float>();
    auto executed_primitives = network.get_executed_primitives();

    EXPECT_TRUE(executed_primitives.count("depth1") == 0);
    EXPECT_TRUE(executed_primitives.count("depth2") == 0);
    EXPECT_TRUE(executed_primitives.count("depth3") == 0);

    EXPECT_NEAR(1.4142f, output_ptr[0], 1e-3);
    EXPECT_NEAR(1.5422f, output_ptr[1], 1e-3);
    EXPECT_NEAR(1.8340f, output_ptr[2], 1e-3);
    EXPECT_NEAR(2.0f, output_ptr[3], 1e-3);
    EXPECT_NEAR(2.0f, output_ptr[4], 1e-3);
    EXPECT_NEAR(2.3784f, output_ptr[5], 1e-3);
    EXPECT_NEAR(3.3635f, output_ptr[6], 1e-3);
    EXPECT_NEAR(4.0f, output_ptr[7], 1e-3);
    EXPECT_NEAR(2.0f, output_ptr[8], 1e-3);
    EXPECT_NEAR(2.3784f, output_ptr[9], 1e-3);
    EXPECT_NEAR(3.3635f, output_ptr[10], 1e-3);
    EXPECT_NEAR(4.0f, output_ptr[11], 1e-3);
    EXPECT_NEAR(4.0f, output_ptr[12], 1e-3);
    EXPECT_NEAR(5.6568f, output_ptr[13], 1e-3);
    EXPECT_NEAR(11.3137f, output_ptr[14], 1e-3);
    EXPECT_NEAR(16.0f, output_ptr[15], 1e-3);

}

TEST(depth_concatenate_f32_gpu, test04_fused_relu) {
    // 2 inputs of size 3x10x10 concatenated on f axis with fused relu

    engine engine;
    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1,3,10,10 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1,3,10,10 } });

    std::vector<float> input1_vec = generate_random_input<float>(1, 3, 10, 10, -10, 10);
    set_values(input1, input1_vec);
    std::vector<float> input2_vec = generate_random_input<float>(1, 3, 10, 10, -10, 10);
    set_values(input2, input2_vec);

    topology topology;
    topology.add(input_layout("input1", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(concatenation("depth1", { "input1", "input2" }, concatenation::along_f));
    topology.add(activation("relu1", "depth1", activation_relu));

    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));
    network network(engine, topology, options);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute({});
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "relu1");

    auto output = outputs.at("relu1").get_memory();

    auto output_ptr = output.pointer<float>();
    unsigned int elements_count = 600;
    unsigned int input_element_count = 300;
    for (unsigned int i = 0; i < 600; i++)
    {
        if(i < input_element_count)
            EXPECT_FLOAT_EQ(input1_vec[i] < 0.0f ? 0.0f : input1_vec[i], output_ptr[i]);
        else
            EXPECT_FLOAT_EQ(input2_vec[i - input_element_count] < 0.0f ? 0.0f : input2_vec[i - input_element_count], output_ptr[i]);
    }
}


TEST(depth_concatenate_f32_gpu, test05_different_formats) {
    // 2 inputs of size 3x10x10 concatenated on f axis 

    engine engine;
    auto input1 = memory::allocate(engine, { data_types::f32, format::bfyx,{ 1,3,2,2 } });
    auto input2 = memory::allocate(engine, { data_types::f32, format::yxfb,{ 1,3,2,2 } });

    set_values(input1, { 1.0f, 1.0f, 1.0f, 1.0f,
                        2.0f, 2.0f, 2.0f, 2.0f,
                        3.0f, 3.0f, 3.0f, 3.0f });
    set_values(input2, { -1.0f, -2.0f, -3.0f,
                         -1.0f, -2.0f, -3.0f, 
                         -1.0f, -2.0f, -3.0f,
                        - 1.0f, -2.0f, -3.0f });

    std::vector<float> out_ref = {
        1.0f, 1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 3.0f, 3.0f,
        -1.0f, -1.0f, -1.0f, -1.0f,
        -2.0f, -2.0f, -2.0f, -2.0f,
        -3.0f, -3.0f, -3.0f, -3.0f
    };

    topology topology;
    topology.add(input_layout("input1", input1.get_layout()));
    topology.add(input_layout("input2", input2.get_layout()));
    topology.add(concatenation("depth1", { "input1", "input2" }, concatenation::along_f));

    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));
    network network(engine, topology, options);

    network.set_input_data("input1", input1);
    network.set_input_data("input2", input2);

    auto outputs = network.execute({});
    EXPECT_EQ(outputs.size(), size_t(1));
    EXPECT_EQ(outputs.begin()->first, "depth1");

    auto output = outputs.at("depth1").get_memory();
    auto output_ptr = output.pointer<float>();
    int cntr = 0;
    for (float val : output_ptr)
    {
        EXPECT_EQ(val, out_ref[cntr++]);
    }
    

}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Negative Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

//TODO: this should be done using TEST_P or some equivallent construct
static network setup_depth_concatatenate_network(const std::vector<data_types> dts, const std::vector<tensor> ts, const std::vector<cldnn::format> fmt)
{
    assert(dts.size() == ts.size());
    const size_t sz = ts.size();

    engine engine;
    topology topology;

    std::vector<std::string> input_names;
    input_names.resize(sz);

    for (size_t i = 0; i < sz; ++i)
    {
        auto input = memory::allocate(engine, { dts[i], fmt[i], ts[i] });

        input_names[i] = "input";
        input_names[i] += std::to_string(i);

        topology.add(input_layout(input_names[i], input.get_layout()));
    }
    //TODO: ask Uzi if something tests cases where there's missing input_names (nodes not present in the topology, etc.)
    topology.add(concatenation("depth_concat_node", input_names, concatenation::along_f));

    return network(engine, topology);
}

TEST(NegativeDepthConcatenateTest, DISABLED_TestAll) {
    auto d = data_types::f32;
    auto od = data_types::f16;

    auto f = format::bfyx;

    std::vector<int> t { 1, 2, 3, 4 };
    std::vector<int> t0 { 7, 2, 3, 4 };
    std::vector<int> t1 { 1, 2, 7, 4 };
    std::vector<int> t2 { 1, 2, 3, 7 };

    //TODO: should be ASSERT_THROW(statement, exception_type) - but what exception type?
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ }, { }, { }));

    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, od }, { tensor(t), tensor(t) }, { f, f }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d }, { tensor(t), tensor(t0) }, { f, f }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d }, { tensor(t), tensor(t1) }, { f, f }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d }, { tensor(t), tensor(t2) }, { f, f }));

    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, od, d }, { tensor(t), tensor(t), tensor(t) }, { f, f, f }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d, od }, { tensor(t), tensor(t), tensor(t) }, { f, f, f }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d, d }, { tensor(t), tensor(t0), tensor(t) }, { f, f, f }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d, d }, { tensor(t), tensor(t1), tensor(t) }, { f, f, f }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d, d }, { tensor(t), tensor(t2), tensor(t) }, { f, f, f }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d, d }, { tensor(t), tensor(t), tensor(t0) }, { f, f, f }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d, d }, { tensor(t), tensor(t), tensor(t1) }, { f, f, f }));
    ASSERT_ANY_THROW(setup_depth_concatatenate_network({ d, d, d }, { tensor(t), tensor(t), tensor(t2) }, { f, f, f }));
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Positive Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

using namespace cldnn;

class depth_concatenate_test : public tests::generic_test
{

public:

    static void TearDownTestCase()
    {
        for (auto generic_params : all_generic_params)
        {
            delete generic_params;
        }

        for (auto layer_params : all_layer_params)
        {
            delete layer_params;
        }
    }

    static std::vector<cldnn::primitive*> generate_specific_test_params(int i)
    {
        std::vector<cldnn::primitive*> all_layer_params;

        switch(i)
        {
            case 1 : all_layer_params.push_back(new concatenation("depth_concatenate", {"input0"}, concatenation::along_f)); break;
            case 2 : all_layer_params.push_back(new concatenation("depth_concatenate", {"input0", "input1"}, concatenation::along_f)); break;
            case 3 : all_layer_params.push_back(new concatenation("depth_concatenate", {"input0", "input1", "input2"}, concatenation::along_f)); break;
            default: assert(0);
        }

        return all_layer_params;
    }

    static std::vector<tests::test_params*> generate_generic_test_params(int input_count)
    {
        std::vector<tests::test_params*> all_generic_params;

        for (cldnn::data_types dt : test_data_types())
        for (int32_t b : test_batch_sizes)
        for (tensor & t : test_input_sizes)
        {
            const int w = t.spatial[0];
            const int h = t.spatial[1];

            switch(input_count)
            {
                case 1:
                    for(auto f0 : test_feature_sizes)
                    {
                        test_params * tp = new test_params();
                        tp->data_type = dt;

                        tp->input_layouts.push_back( cldnn::layout(tp->data_type, tp->fmt, cldnn::tensor(  b, f0, w, h )) );

                        all_generic_params.emplace_back(tp);
                    }
                    break;
                case 2:
                    for(auto f0 : test_feature_sizes)
                    for(auto f1 : test_feature_sizes)
                    {
                        test_params * tp = new test_params();
                        tp->data_type = dt;

                        tp->input_layouts.push_back(cldnn::layout(tp->data_type, tp->fmt, cldnn::tensor(  b, f0, w, h )) );
                      tp->input_layouts.push_back(cldnn::layout(tp->data_type, tp->fmt, cldnn::tensor(  b, f1, w, h )) );

                        all_generic_params.emplace_back(tp);
                    }
                    break;
                case 3:
                    for(auto f0 : test_feature_sizes)
                    for(auto f1 : test_feature_sizes)
                    for(auto f2 : test_feature_sizes)
                    {
                        test_params * tp = new test_params();
                        tp->data_type = dt;

                        tp->input_layouts.push_back(cldnn::layout(tp->data_type, tp->fmt, cldnn::tensor( b, f0, w, h )) );
                        tp->input_layouts.push_back(cldnn::layout(tp->data_type, tp->fmt, cldnn::tensor( b, f1, w, h )) );
                        tp->input_layouts.push_back(cldnn::layout(tp->data_type, tp->fmt, cldnn::tensor( b, f2, w, h )) );

                        all_generic_params.emplace_back(tp);
                    }
                    break;
                default:
                    assert(0);
            }
        }

        return all_generic_params;
    }

    static std::vector<std::tuple<test_params*, cldnn::primitive*>> generate_all_test_params()
    {
        std::vector<std::tuple<test_params*, cldnn::primitive*>> res;

        for (int i = 1; i <= 3; ++i)
        {
            auto tpv = generate_generic_test_params(i); 
            auto pv = generate_specific_test_params(i);

            all_generic_params.insert(all_generic_params.end(), tpv.begin(), tpv.end());
            all_layer_params.insert(all_layer_params.end(), pv.begin(), pv.end());

            for (auto & tp : tpv)
            for (auto & p: pv)
                res.emplace_back(tp, p);
        }

        return res;
    }

    virtual bool is_format_supported(cldnn::format format) override
    {
        return format == cldnn_format_type::cldnn_format_bfyx;
    }

    virtual cldnn::tensor get_expected_output_tensor() override
    {
        cldnn::tensor::value_type features = 0;
        for (const auto& t : generic_params->input_layouts)
        {
            features += t.size.feature[0];
        }

        const auto& t = generic_params->input_layouts[0].size;
        return{ t.batch[0], features, t.spatial[0], t.spatial[1] };
    }

    template<typename Type>
    memory generate_reference_typed(const std::vector<memory> & inputs)
    {
        assert(!inputs.empty());

        const int in_b = inputs[0].get_layout().size.batch[0];
        const int in_h = inputs[0].get_layout().size.spatial[1];
        const int in_w = inputs[0].get_layout().size.spatial[0];

        int out_f = 0;

        for (const memory & input : inputs)
        {
            assert(input.get_layout().size.batch[0] == in_b);
            assert(input.get_layout().size.spatial[1] == in_h);
            assert(input.get_layout().size.spatial[0] == in_w);

            out_f += input.get_layout().size.feature[0];

            assert(input.get_layout().data_type == inputs[0].get_layout().data_type);
            assert(input.get_layout().format == inputs[0].get_layout().format);
        }

        //Output is bfyx
        auto output = memory::allocate(engine, cldnn::layout(inputs[0].get_layout().data_type, cldnn::format::bfyx, tensor( in_b, out_f, in_w, in_h )));
        auto out_mem = output.pointer<Type>();

        int out_f_off = 0;
        for (const memory & input : inputs)
        {
            const auto input_desc = get_linear_memory_desc(input.get_layout());
            const auto output_desc = get_linear_memory_desc(output.get_layout());

            const int in_f = input.get_layout().size.feature[0];
            const auto in_mem = input.pointer<Type>();

            for (int n = 0; n < in_b; ++n)
            for (int f = 0; f < in_f; ++f)
            for (int y = 0; y < in_h; ++y)
            for (int x = 0; x < in_w; ++x)
            {
                const size_t in_idx = get_linear_index(input.get_layout(), n, f, y, x, input_desc);
                const size_t out_idx = get_linear_index(output.get_layout(), n, out_f_off + f, y, x, output_desc);

                out_mem[out_idx] = in_mem[in_idx];
            }

            out_f_off += in_f;
        }

        return output;
    }

    virtual memory generate_reference(const std::vector<memory> & inputs) override
    {
        if (generic_params->data_type == data_types::f32)
        {
            return generate_reference_typed<float>(inputs);
        }
        else
        {
            return generate_reference_typed<FLOAT16>(inputs);
        }
    }

    static std::string custom_param_name(const ::testing::TestParamInfo<std::tuple<test_params*, cldnn::primitive*>>& info)
    {
        std::stringstream res;

        const auto & p = std::get<0>(info.param);

        assert (p->data_type == data_types::f32 ||
                p->data_type == data_types::f16);

        res << info.index
            << "_" << (p->data_type == data_types::f32 ? "f32" : "f16");

        for (unsigned i = 0; i < p->input_layouts.size(); ++i)
        {
            const auto chans = p->fmt.order();

            res << "_" << "Input" << i;
            for (unsigned int j = 0; j < p->input_layouts[i].size.sizes(p->fmt).size(); ++j)
            {
                res << chans[j] << p->input_layouts[i].size.sizes(p->fmt)[j];
            }
        }

        return res.str();
    }

private:

    static std::vector<tests::test_params*> all_generic_params;
    static std::vector<cldnn::primitive*> all_layer_params;

};

std::vector<cldnn::primitive*> depth_concatenate_test::all_layer_params = {};
std::vector<tests::test_params*> depth_concatenate_test::all_generic_params = {};

TEST_P(depth_concatenate_test, DEPTHCONCATENATE)
{
    run_single_test();
}
 
INSTANTIATE_TEST_CASE_P(DISABLED_DEPTHCONCATENATE,
    depth_concatenate_test,
    ::testing::ValuesIn(depth_concatenate_test::generate_all_test_params()),
    depth_concatenate_test::custom_param_name);

