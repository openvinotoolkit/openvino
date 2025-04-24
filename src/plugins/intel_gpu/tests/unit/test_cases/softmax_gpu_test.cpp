// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include "openvino/reference/softmax.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/softmax.hpp>

#include <softmax_inst.h>

using namespace cldnn;
using namespace std;
using namespace ::tests;

class softmax_gpu_xb_f32_test_fixture: public ::testing::Test {
public:
    static const int32_t
        output_x  = 10, output_b  = 2,  // size of whole output buffer
        input_x   = 10, input_b   = 2,  // size of whole input buffer
        in_size   = input_x*input_b,
        out_size  = output_x*output_b;

    float in_buffer[in_size];
    float out_buffer[out_size];
    float expected_buffer[out_size];

    cldnn::engine& engine;
    cldnn::memory::ptr input;

    //neural::primitive output = memory::allocate({ memory::format::xb_f32, {output_b, {{output_x}}, 1}});

    softmax_gpu_xb_f32_test_fixture()
        : engine(get_test_engine())
        , input(engine.allocate_memory({ data_types::f32, format::yxfb, { input_b, 1, input_x, 1}}))
    {}

    void compare_out_buffer_with_expected() {
        for(size_t i = 0; i < out_size; ++i) {
            // does output have expected values
            ASSERT_TRUE(are_equal(out_buffer[i], expected_buffer[i]))
                << "At ["<< i <<  "] Expected : " << expected_buffer[i] << " actual : " << out_buffer[i];
        }
    }

    void compare_out_buffer_with_expected_batch_wise() {
        for(size_t b = 0; b < output_b; ++b) {
            float batch_wise_sum = 0;
            for(size_t x = 0; x < output_x; ++x) {
                auto idx = b+x*output_b;
                batch_wise_sum += out_buffer[idx];
                // does output have expected values
                ASSERT_TRUE(are_equal(out_buffer[idx], expected_buffer[idx]))
                    << "At ["<< idx <<  "] Expected : " << expected_buffer[idx] << " actual : " << out_buffer[idx];
            }
            // does it sum to 1 batch wise
            ASSERT_TRUE(are_equal(batch_wise_sum, 1.0f))
                << "Expected : " << 1.0f << " actual : " << batch_wise_sum;
        }
    }

    void test_input_same_values(bool is_caching_test) {
    // in_buffer filled with same value == 1.0f
        for(uint32_t i = 0; i < out_size; ++i) {
                in_buffer[i] = 1.0f;
            expected_buffer[i] = 0.1f;
        }
        std::vector<float> in_b(std::begin(in_buffer), std::end(in_buffer));

        set_values(input, in_b);

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(softmax("softmax", input_info("input"), 3));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "softmax");

        auto output_prim = outputs.begin()->second.get_memory();

        cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());
        for (uint32_t i = 0; i < out_size; i++) {
            out_buffer[i] = output_ptr[i];
        }
        compare_out_buffer_with_expected();

    }

    void test_input_same_values_batch_wise(bool is_caching_test) {
    // in_buffer filled with same value == 1..2 each batch accordingly (softmax can only xb_f32 )
        for(size_t i = 0; i < output_x; ++i) {
            for(size_t j = 0; j < output_b; ++j)
                in_buffer[j+i*output_b] = (j+i*output_b) % 2 +1.0f;
        }

        std::vector<float> in_b(std::begin(in_buffer), std::end(in_buffer));
        set_values(input, in_b);
        // fill buffer with the expected 0.1f value
        for(size_t i = 0; i < out_size; ++i)
            expected_buffer[i] = 0.1f;

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(softmax("softmax", input_info("input"), 3));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        network->set_input_data("input", input);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "softmax");

        auto output_prim = outputs.begin()->second.get_memory();

        cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());
        for (uint32_t i = 0; i < out_size; i++) {
            out_buffer[i] = output_ptr[i];
        }
        compare_out_buffer_with_expected_batch_wise();
    }

    void test_values_batch_wise(bool is_caching_test) {
        float in_buf[in_size] = {
        //b0  b1
            2.0f, 2.0f, //x0
            2.0f, 2.0f, //x1
            2.0f, 2.0f, //x2
            3.0f, 3.0f, //x3
            5.0f, 5.0f, //x4
            4.0f, 4.0f, //x5
            3.0f, 3.0f, //x6
            2.0f, 2.0f, //x7
            2.0f, 2.0f, //x8
            2.0f, 2.0f  //x9
        };

        float exp_buf[out_size] = {
            0.02569957f,     0.02569957f,
            0.02569957f,     0.02569957f,
            0.02569957f,     0.02569957f,
            0.069858674f,    0.069858674f,
            0.516189665f,    0.516189665f,
            0.189895565f,    0.189895565f,
            0.069858674f,    0.069858674f,
            0.02569957f,     0.02569957f,
            0.02569957f,     0.02569957f,
            0.02569957f,     0.02569957f

        };

        std::vector<float> in_b(std::begin(in_buf), std::end(in_buf));
        set_values(input, in_b);
        std::copy(exp_buf, exp_buf+in_size, expected_buffer);

        // out_buffer filled with non-signaling NaN
        for(size_t i = 0; i < out_size; ++i)
            out_buffer[i] = NAN;

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(softmax("softmax", input_info("input"), 3));

        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);

        auto outputs = network->execute();
        ASSERT_EQ(outputs.size(), size_t(1));
        ASSERT_EQ(outputs.begin()->first, "softmax");

        auto output_prim = outputs.begin()->second.get_memory();

        cldnn::mem_lock<float> output_ptr(output_prim, get_test_stream());
        for (uint32_t i = 0; i < out_size; i++) {
            out_buffer[i] = output_ptr[i];
        }
        compare_out_buffer_with_expected_batch_wise();
    }
};

TEST_F(softmax_gpu_xb_f32_test_fixture, input_same_values) {
    this->test_input_same_values(false);
}

TEST_F(softmax_gpu_xb_f32_test_fixture, input_same_values_batch_wise) {
    this->test_input_same_values_batch_wise(false);
}

TEST_F(softmax_gpu_xb_f32_test_fixture, values_batch_wise) {
    this->test_values_batch_wise(false);
}

TEST(softmax_gpu_bfyx_f32, normalize_y) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3,
        batch_num = 2, buf_size = x_size*y_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { batch_num, feature_num, x_size, y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", input_info("input"), 2));

    vector<float> input_vec = {
              //y0x0  y0x1   y1x0    y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   12.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
    };
    set_values(input, input_vec);

    float expected_max_values[12] = {
        0.689974481f,   //b=0, f=0, x=0
        0.832018385f,   //b=0, f=0, x=1

        0.999962831f,   //b=0, f=1, x=0
        0.993307149f,   //b=0, f=1, x=1

        0.999962831f,   //b=0, f=2, x=0
        0.993307149f,   //b=0, f=2, x=1

        0.98201379f,    //b=1, f=0, x=0
        0.99998987f,    //b=1, f=0, x=1

        0.98201379f,    //b=1, f=1, x=0
        0.999547378f,   //b=1, f=1, x=1

        0.999962831f,   //b=1, f=2, x=0
        0.993307149f    //b=1, f=2, x=1
    };

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++) {
        out_buffer[i] = output_ptr[i];
    }

    float temp_max = 0;
    float expected_sum = 1.0f;
    int max_value_buffer_index = 0;
    for (uint32_t i = 0; i < batch_num; i++) { //this for loops will sum results in a batch per feature, we expect that: sum = 1.0f
        for (uint32_t l = 0; l < feature_num; l++) {
            for (uint32_t k = 0; k < x_size; k++) {
                float sum = 0.0f;
                for (uint32_t j = 0; j < y_size; j++) {
                    int index = i * feature_num * x_size * y_size +
                                l * x_size * y_size +
                                j * x_size +
                                k;
                    if (out_buffer[index] >= temp_max) {
                        temp_max = out_buffer[index];
                    }
                    sum += out_buffer[index];
                }
                ASSERT_EQ(true, are_equal(temp_max, expected_max_values[max_value_buffer_index]));
                temp_max = 0;
                max_value_buffer_index++;

                ASSERT_EQ(true, are_equal(sum, expected_sum));
                sum = 0.0f;
            }
        }
    }
}

TEST(softmax_gpu_bfyx_f32, normalize_f) {
    //  Input  : 2x3x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 3,
        batch_num = 2, buf_size = x_size*y_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { batch_num, feature_num, x_size, y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", input_info("input"), 1));

    vector<float> input_vec = {
        //y0x0  y0x1   y1x0    y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   12.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
    };
    set_values(input, input_vec);

    float expected_max_values[8] = {
        0.344253346f, //b=0, y=0, x=0
        0.364854551f, //b=0, y=0, x=1

        0.999963085f, //b=0, y=1, x=0
        0.493894592f, //b=0, y=1, x=1

        0.719294981f, //b=1, y=0, x=0
        0.364854551f, //b=1, y=0, x=1

        0.73105857f, //b=1, y=1, x=0
        0.977054322f //b=1, y=1, x=1
    };

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    for (size_t i = 0; i < output->count(); i++) {
        std::cerr << "i = " << i << " v = " << output_ptr[i] << std::endl;
    }

    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++) {
        out_buffer[i] = output_ptr[i];
    }

    float temp_max = 0;
    float expected_sum = 1.0f;
    int max_value_buffer_index = 0;
    for (uint32_t i = 0; i < batch_num; i++) { //this for loops will sum results in a batch per feature, we expect that: sum = 1.0f
        for (uint32_t j = 0; j < y_size; j++) {
            for (uint32_t k = 0; k < x_size; k++) {
                float sum = 0.0f;
                for (uint32_t l = 0; l < feature_num; l++) {
                    int index = i * feature_num * x_size * y_size +
                                l * x_size * y_size +
                                j * x_size +
                                k;
                    if (out_buffer[index] >= temp_max) {
                        temp_max = out_buffer[index];
                    }
                    sum += out_buffer[index];
                }
                ASSERT_EQ(true, are_equal(temp_max, expected_max_values[max_value_buffer_index]));
                temp_max = 0;
                max_value_buffer_index++;

                ASSERT_EQ(true, are_equal(sum, expected_sum));
                sum = 0.0f;
            }
        }
    }
}

TEST(softmax_gpu_bfzyx_f32, normalize_z) {
    //  Input  : 2x3x2x2x2
    static const int32_t x_size = 2, y_size = 2, z_size = 2, feature_num = 3,
        batch_num = 2, buf_size = x_size  *y_size * z_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfzyx, { batch_num, feature_num, x_size, y_size, z_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", input_info("input"), 2));

    vector<float> input_vec = {
        //    z0y0x0 z0y0x1 z0y1x0 z0y1x1 z1y0x0 z1y0x1 z1y1x0 z1y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f, 0.2f, -0.2f, 0.9f,  2.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f, 0.3f, 0.1f,  -11.f, 6.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f, 0.1f, 0.3f,  -9.f,  4.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   12.f, 5.f,  0.1f,  6.f,   22.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f, 2.2f,  0.3f,  6.f,  5.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f, 1.2f, 0.3f,  -12.f,  2.2f
    };
    set_values(input, input_vec);

    float expected_max_values[24] = {
        0.524979f, 0.524979f,
        0.5f,      0.731059f,
        0.524979f, 0.524979f,
        0.731059f, 0.731059f,
        0.524979f, 0.524979f,
        0.731059f, 0.731059f,
        0.880797f, 0.598688f,
        0.731059f, 0.999955f,
        0.858149f, 0.549834f,
        0.880797f, 0.952574f,
        0.731059f, 0.524979f,
        0.880797f, 0.952574f,
    };

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++) {
        out_buffer[i] = output_ptr[i];
    }

    float temp_max = 0;
    float expected_sum = 1.0f;
    int max_value_buffer_index = 0;
    for (uint32_t i = 0; i < batch_num; i++) {
        for (uint32_t l = 0; l < feature_num; l++) {
            for (uint32_t j = 0; j < y_size; j++) {
                for (uint32_t k = 0; k < x_size; k++) {
                    float sum = 0.0f;
                    for (uint32_t m = 0; m < z_size; m++) {
                        int index = i * feature_num * x_size * y_size * z_size +
                                    l * x_size * y_size * z_size +
                                    m * x_size * y_size +
                                    j * x_size +
                                    k;
                        if (out_buffer[index] >= temp_max) {
                            temp_max = out_buffer[index];
                        }
                        sum += out_buffer[index];
                    }
                    ASSERT_EQ(true, are_equal(temp_max, expected_max_values[max_value_buffer_index]));
                    temp_max = 0;
                    max_value_buffer_index++;
                    ASSERT_EQ(true, are_equal(sum, expected_sum));
                    sum = 0.0f;
                }
            }
        }
    }
}

TEST(softmax_gpu_bfyx_f32, normalize_b) {
    //  Input  : 3x2x2x2
    static const int32_t x_size = 2, y_size = 2, feature_num = 2,
            batch_num = 3, buf_size = x_size*y_size * batch_num * feature_num;
    auto& engine = get_test_engine();

    auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { batch_num, feature_num, x_size, y_size } });
    topology topology;
    topology.add(input_layout("input", input->get_layout()));
    topology.add(softmax("softmax", input_info("input"), 0));

    vector<float> input_vec = {
        //      y0x0  y0x1   y1x0    y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/3.f,  0.5f,  7.f,   12.f,

        /*b1f0*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,

        /*b2f0*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b2f1*/0.2f, 0.2f,  -10.f, 5.2f
    };
    set_values(input, input_vec);

    float expected_max_values[8] = {
        0.344253346f, //f=0, y=0, x=0
        0.364854551f, //f=0, y=0, x=1

        0.999963085f, //f=0, y=1, x=0
        0.493894592f, //f=0, y=1, x=1

        0.719294981f, //f=1, y=0, x=0
        0.364854551f, //f=1, y=0, x=1

        0.73105857f, //f=1, y=1, x=0
        0.977054322f //f=1, y=1, x=1
    };

    network network(engine, topology, get_test_default_config(engine));

    network.set_input_data("input", input);
    auto outputs = network.execute();

    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++) {
        out_buffer[i] = output_ptr[i];
    }

    float temp_max = 0;
    float expected_sum = 1.0f;
    int max_value_buffer_index = 0;
    for (uint32_t i = 0; i < feature_num; i++) { //this for loops will sum results in a batch per feature, we expect that: sum = 1.0f
        for (uint32_t j = 0; j < y_size; j++) {
            for (uint32_t k = 0; k < x_size; k++) {
                float sum = 0.0f;
                for (uint32_t l = 0; l < batch_num; l++) {
                    int index = l * feature_num * x_size * y_size +
                                i * x_size * y_size +
                                j * x_size +
                                k;
                    if (out_buffer[index] >= temp_max) {
                        temp_max = out_buffer[index];
                    }
                    sum += out_buffer[index];
                }
                ASSERT_EQ(true, are_equal(temp_max, expected_max_values[max_value_buffer_index]));
                temp_max = 0;
                max_value_buffer_index++;

                ASSERT_EQ(true, are_equal(sum, expected_sum));
                sum = 0.0f;
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Negative Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

//TODO:
//TEST(NegativeSoftmaxTest, DISABLED_TestAll) {
//}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                      Exhaustive Positive Matrix tests                    //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

using namespace cldnn;

class softmax_test : public tests::generic_test
{

public:
    softmax_test() : tests::generic_test() {}

    void SetUp() override {
        max_ulps_diff_allowed = 6;
    }

    static void TearDownTestCase() {
        all_layer_params.clear();
        all_generic_params.clear();
    }

    static std::vector<std::shared_ptr<cldnn::primitive>> generate_specific_test_params() {
        all_layer_params.emplace_back(new softmax("softmax", input_info("input0"), 1));

        //The test checks only valid combinations.
        //TODO: add more combinations.

        return all_layer_params;
    }

    static std::vector<std::shared_ptr<tests::test_params>> generate_generic_test_params() {
        return generic_test::generate_generic_test_params(all_generic_params);
    }

    bool is_format_supported(cldnn::format format) override {
        return
            format == cldnn::format::yxfb ||
            format == cldnn::format::bfyx;
    }

    template<typename Type>
    memory::ptr generate_reference_typed(const std::vector<memory::ptr>& inputs) {
        assert(inputs.size() == 1);
        const memory::ptr input = inputs[0];

        //Output is bfyx
        auto output = engine.allocate_memory(cldnn::layout(input->get_layout().data_type, input->get_layout().format, input->get_layout().get_tensor()));

        cldnn::mem_lock<Type> in0_mem(input, get_test_stream());
        cldnn::mem_lock<Type> out_mem(output, get_test_stream());

        const int in0_b = input->get_layout().get_tensor().sizes()[0];
        const int in0_f = input->get_layout().get_tensor().sizes()[1];
        const int in0_h = input->get_layout().get_tensor().sizes()[3];
        const int in0_w = input->get_layout().get_tensor().sizes()[2];

//        const int out_b = output->get_layout().get_tensor().transform(cldnn::format::bfyx, 0).sizes()[0];
//        const int out_f = output->get_layout().get_tensor().transform(cldnn::format::bfyx, 0).sizes()[1];
//        const int out_h = output->get_layout().get_tensor().transform(cldnn::format::bfyx, 0).sizes()[2];
//        const int out_w = output->get_layout().get_tensor().transform(cldnn::format::bfyx, 0).sizes()[3];

//        assert(in0_b == out_b);
//        assert(in0_f == out_f);
//        assert(in0_h == out_h);
//        assert(in0_w == out_w);

        std::vector<float> cached_exp_vals;
        cached_exp_vals.resize(in0_f);

        const auto input_desc = get_linear_memory_desc(input->get_layout());

        for (int n = 0; n < in0_b; ++n)
        for (int y = 0; y < in0_h; ++y)
        for (int x = 0; x < in0_w; ++x) {
            float max_val = -std::numeric_limits<float>::infinity();

            for (int c = 0; c < in0_f; ++c) {
                const size_t in0_idx = get_linear_index(input->get_layout(), n, c, y, x, input_desc);

                max_val = std::max(max_val, static_cast<float>(in0_mem[in0_idx]));
            }

            float Z = 0;

            for (int c = 0; c < in0_f; ++c) {
                const size_t in0_idx = get_linear_index(input->get_layout(), n, c, y, x, input_desc);

                float tmp = static_cast<float>((Type)std::exp(static_cast<float>(in0_mem[in0_idx]) - max_val));
                Z += tmp;
                cached_exp_vals[c] = tmp;
            }

            for (int c = 0; c < in0_f; ++c) {
                const size_t out_idx = get_linear_index(output->get_layout(), n, c, y, x, input_desc);
                out_mem[out_idx] = (Type)(cached_exp_vals[c] / Z);
            }
        }

        return output;
    }

    virtual memory::ptr generate_reference(const std::vector<memory::ptr>& inputs) override {
        if (generic_params->data_type == data_types::f32) {
            return generate_reference_typed<float>(inputs);
        } else {
            return generate_reference_typed<ov::float16>(inputs);
        }
    }

    static std::string custom_param_name(const ::testing::TestParamInfo<std::tuple<std::shared_ptr<tests::test_params>, std::shared_ptr<cldnn::primitive>>>& info) {
        std::stringstream res;

        const auto& p = std::get<0>(info.param);

        assert (p->data_type == data_types::f32 ||
                p->data_type == data_types::f16);

        res << info.index
            << "_" << (p->data_type == data_types::f32 ? "f32" : "f16");

        for (unsigned i = 0; i < p->input_layouts.size(); ++i) {
            const auto chans = format::traits(p->fmt).order;

            res << "_" << "Input" << i;
            for (unsigned int j = 0; j < p->input_layouts[i].get_tensor().sizes(p->fmt).size(); ++j) {
                res << chans[j] << p->input_layouts[i].get_tensor().sizes(p->fmt)[j];
            }
        }

        return res.str();
    }

private:
    static std::vector<std::shared_ptr<tests::test_params>> all_generic_params;
    static std::vector<std::shared_ptr<cldnn::primitive>> all_layer_params;
};

std::vector<std::shared_ptr<cldnn::primitive>> softmax_test::all_layer_params = {};
std::vector<std::shared_ptr<tests::test_params>> softmax_test::all_generic_params = {};

TEST_P(softmax_test, SOFTMAX) {
    run_single_test();
}

INSTANTIATE_TEST_SUITE_P(DISABLED_SOFTMAX,
    softmax_test,
    ::testing::Combine(::testing::ValuesIn(softmax_test::generate_generic_test_params()), ::testing::ValuesIn(softmax_test::generate_specific_test_params())),
    softmax_test::custom_param_name);



namespace {
template<typename T>
struct SoftmaxParams {
    int64_t axis;
    tensor input_tensor;
    std::vector<T> input;
    std::vector<T> expected;
};

template<typename T>
using SoftmaxParamsWithFormat = std::tuple<
    SoftmaxParams<T>,
    format::type,     // source (plain) layout
    format::type      // target (blocked) layout
>;

const std::vector<format::type> formats2D{
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_yx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32
};

const std::vector<format::type> formats3D{
        format::bfzyx,
        format::b_fs_zyx_fsv16,
        format::bs_fs_zyx_bsv16_fsv16
};

template<typename T>
std::vector<T> getValues(const std::vector<float> &values) {
    std::vector<T> result(values.begin(), values.end());
    return result;
}

template<typename T>
std::vector<SoftmaxParams<T>> generateSoftmaxParams2D() {
    const std::vector<SoftmaxParams<T>> result = {
        {
            0,
            tensor(3, 2, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 3.f, 0.5f, 7.f, 12.f, 0.2f, 0.2f, -10.f, 5.2f,
                4.f, 0.5f, 8.f, 8.2f, 0.2f, 0.2f, -10.f, 5.2f, 0.2f, 0.2f, -10.f, 5.2f}),
            getValues<T>({
                0.311493f, 0.270291f, 0.999963f, 0.0122108f,
                0.264614f, 0.364855f, 0.268941f, 0.977054f,
                0.344253f, 0.364855f, 1.84576e-05f, 0.493895f,
                0.719295f, 0.364855f, 0.731059f, 0.0218575f,
                0.344253f, 0.364855f, 1.84576e-05f, 0.493895f,
                0.0160912f, 0.270291f, 1.1134e-08f, 0.00108822f})
        },
        {
            1,
            tensor(2, 3, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 0.2f, 0.2f, -10.f, 5.2f, 0.2f, 0.2f, -10.f, 5.2f,
                3.f, 0.5f, 7.f, 12.f, 4.f, 0.5f, 8.f, 8.2f, 0.2f, 0.2f, -10.f, 5.2f}),
            getValues<T>({
                0.311493f, 0.270291f, 0.999963f, 0.0122108f,
                0.344253f, 0.364855f, 1.84576e-05f, 0.493895f,
                0.344253f, 0.364855f, 1.84576e-05f, 0.493895f,
                0.264614f, 0.364855f, 0.268941f, 0.977054f,
                0.719295f, 0.364855f, 0.731059f, 0.0218575f,
                0.0160912f, 0.270291f, 1.1134e-08f, 0.00108822f})
        },
        {
            2,
            tensor(2, 3, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 0.2f, 0.2f, -10.f, 5.2f, 0.2f, 0.2f, -10.f, 5.2f,
                3.f, 0.5f, 7.f, 12.f, 4.f, 0.5f, 8.f, 8.2f, 0.2f, 0.2f, -10.f, 5.2f}),
            getValues<T>({
                0.310026f, 0.167982f, 0.689974f, 0.832018f,
                0.999963f, 0.00669285f, 3.71689e-05f, 0.993307f,
                0.999963f, 0.00669285f, 3.71689e-05f, 0.993307f,
                0.0179862f, 1.013e-05f, 0.982014f, 0.99999f,
                0.0179862f, 0.000452622f, 0.982014f, 0.999547f,
                0.999963f, 0.00669285f, 3.71689e-05f, 0.993307f})
        },
        {
            3,
            tensor(2, 3, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 0.2f, 0.2f, -10.f, 5.2f, 0.2f, 0.2f, -10.f, 5.2f,
                3.f, 0.5f, 7.f, 12.f, 4.f, 0.5f, 8.f, 8.2f, 0.2f, 0.2f, -10.f, 5.2f}),
            getValues<T>({
                0.549834f, 0.450166f, 0.354344f, 0.645656f,
                0.5f, 0.5f, 2.50452e-07f, 1.0f,
                0.5f, 0.5f, 2.50452e-07f, 1.0f,
                0.924142f, 0.0758582f, 0.00669285f, 0.993307f,
                0.970688f, 0.0293122f, 0.450166f, 0.549834f,
                0.5f, 0.5f, 2.50452e-07f, 1.0f})
        },

    };
    return result;
}

template<typename T>
std::vector<SoftmaxParams<T>> generateSoftmaxParams3D() {
    const std::vector<SoftmaxParams<T>> result = {
        {
            0,
            tensor(2, 3, 2, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 0.2f, -0.2f, 0.9f, 2.5f,
                0.2f, 0.2f, -10.f, 5.2f, 0.3f, 0.1f, -11.f, 6.2f,
                0.2f, 0.2f, -10.f, 5.2f, 0.1f, 0.3f, -9.f, 4.2f,
                3.f, 0.5f, 7.f, 12.f, 5.f, 0.1f, 6.f, 22.f,
                4.f, 0.5f, 8.f, 8.2f, 2.2f, 0.3f, 6.f, 5.2f,
                0.2f, 0.2f, -10.f, 5.2f, 1.2f, 0.3f, -12.f, 2.2f}),
            getValues<T>({
                0.0521536f, 0.354344f, 0.00223785f, 2.75357e-05f, 0.00816257f, 0.425557f, 0.0060598f, 3.39827e-09f,
                0.0218813f, 0.425557f, 1.523e-08f, 0.0474259f, 0.130108f, 0.450166f, 4.13994e-08f, 0.731059f,
                0.5f, 0.5f, 0.5f, 0.5f, 0.24974f, 0.5f, 0.952574f, 0.880797f,
                0.947846f, 0.645656f, 0.997762f, 0.999972f, 0.991837f, 0.574443f, 0.99394f, 1.0f,
                0.978119f, 0.574443f, 1.0f, 0.952574f, 0.869892f, 0.549834f, 1.0f, 0.268941f,
                0.5f, 0.5f, 0.5f, 0.5f, 0.75026f, 0.5f, 0.0474259f, 0.119203f})
        },
        {
            1,
            tensor(2, 3, 2, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 0.2f, -0.2f, 0.9f, 2.5f,
                0.2f, 0.2f, -10.f, 5.2f, 0.3f, 0.1f, -11.f, 6.2f,
                0.2f, 0.2f, -10.f, 5.2f, 0.1f, 0.3f, -9.f, 4.2f,
                3.f, 0.5f, 7.f, 12.f, 5.f, 0.1f, 6.f, 22.f,
                4.f, 0.5f, 8.f, 8.2f, 2.2f, 0.3f, 6.f, 5.2f,
                0.2f, 0.2f, -10.f, 5.2f, 1.2f, 0.3f, -12.f, 2.2f}),
            getValues<T>({
                0.311493f, 0.270291f, 0.999963f, 0.0122108f, 0.332225f, 0.250089f, 0.999943f, 0.0213123f,
                0.344253f, 0.364855f, 1.84576e-05f, 0.493895f, 0.367165f, 0.337585f, 6.79002e-06f, 0.862025f,
                0.344253f, 0.364855f, 1.84576e-05f, 0.493895f, 0.30061f, 0.412327f, 5.01718e-05f, 0.116662f,
                0.264614f, 0.364855f, 0.268941f, 0.977054f, 0.923207f, 0.290461f, 0.5f, 1.0f,
                0.719295f, 0.364855f, 0.731059f, 0.0218575f, 0.0561403f, 0.35477f, 0.5f, 5.05653e-08f,
                0.0160912f, 0.270291f, 1.1134e-08f, 0.00108822f, 0.0206528f, 0.35477f, 7.615e-09f, 2.5175e-09f})
        },
        {
            2,
            tensor(2, 3, 2, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 0.2f, -0.2f, 0.9f, 2.5f,
                0.2f, 0.2f, -10.f, 5.2f, 0.3f, 0.1f, -11.f, 6.2f,
                0.2f, 0.2f, -10.f, 5.2f, 0.1f, 0.3f, -9.f, 4.2f,
                3.f, 0.5f, 7.f, 12.f, 5.f, 0.1f, 6.f, 22.f,
                4.f, 0.5f, 8.f, 8.2f, 2.2f, 0.3f, 6.f, 5.2f,
                0.2f, 0.2f, -10.f, 5.2f, 1.2f, 0.3f, -12.f, 2.2f}),
            getValues<T>({
                0.475021f, 0.524979f, 0.5f, 0.268941f, 0.524979f, 0.475021f, 0.5f, 0.731059f,
                0.475021f, 0.524979f, 0.731059f, 0.268941f, 0.524979f, 0.475021f, 0.268941f, 0.731059f,
                0.524979f, 0.475021f, 0.268941f, 0.731059f, 0.475021f, 0.524979f, 0.731059f, 0.268941f,
                0.119203f, 0.598688f, 0.731059f, 4.53979e-05f, 0.880797f, 0.401312f, 0.268941f, 0.999955f,
                0.858149f, 0.549834f, 0.880797f, 0.952574f, 0.141851f, 0.450166f, 0.119203f, 0.0474259f,
                0.268941f, 0.475021f, 0.880797f, 0.952574f, 0.731059f, 0.524979f, 0.119203f, 0.0474259f})
        },
        {
            3,
            tensor(2, 3, 2, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 0.2f, -0.2f, 0.9f, 2.5f,
                0.2f, 0.2f, -10.f, 5.2f, 0.3f, 0.1f, -11.f, 6.2f,
                0.2f, 0.2f, -10.f, 5.2f, 0.1f, 0.3f, -9.f, 4.2f,
                3.f, 0.5f, 7.f, 12.f, 5.f, 0.1f, 6.f, 22.f,
                4.f, 0.5f, 8.f, 8.2f, 2.2f, 0.3f, 6.f, 5.2f,
                0.2f, 0.2f, -10.f, 5.2f, 1.2f, 0.3f, -12.f, 2.2f}),
            getValues<T>({
                0.310026f, 0.167982f, 0.689974f, 0.832018f, 0.331812f, 0.0629734f, 0.668188f, 0.937027f,
                0.999963f, 0.00669285f, 3.71689e-05f, 0.993307f, 0.999988f, 0.00223785f, 1.23728e-05f, 0.997762f,
                0.999963f, 0.00669285f, 3.71689e-05f, 0.993307f, 0.999888f, 0.0198403f, 0.000111653f, 0.98016f,
                0.0179862f, 1.013e-05f, 0.982014f, 0.99999f, 0.268941f, 3.08284e-10f, 0.731059f, 1.0f,
                0.0179862f, 0.000452622f, 0.982014f, 0.999547f, 0.0218813f, 0.00739154f, 0.978119f, 0.992609f,
                0.999963f, 0.00669285f, 3.71689e-05f, 0.993307f, 0.999998f, 0.130108f, 1.8506e-06f, 0.869892f})
        },
        {
            4,
            tensor(2, 3, 2, 2, 2),
            getValues<T>({
                0.1f, -0.1f, 0.9f, 1.5f, 0.2f, -0.2f, 0.9f, 2.5f,
                0.2f, 0.2f, -10.f, 5.2f, 0.3f, 0.1f, -11.f, 6.2f,
                0.2f, 0.2f, -10.f, 5.2f, 0.1f, 0.3f, -9.f, 4.2f,
                3.f, 0.5f, 7.f, 12.f, 5.f, 0.1f, 6.f, 22.f,
                4.f, 0.5f, 8.f, 8.2f, 2.2f, 0.3f, 6.f, 5.2f,
                0.2f, 0.2f, -10.f, 5.2f, 1.2f, 0.3f, -12.f, 2.2f}),
            getValues<T>({
                0.549834f, 0.450166f, 0.354344f, 0.645656f, 0.598688f, 0.401312f, 0.167982f, 0.832018f,
                0.5f, 0.5f, 2.50452e-07f, 1.0f, 0.549834f, 0.450166f, 3.38949e-08f, 1.0f,
                0.5f, 0.5f, 2.50452e-07f, 1.0f, 0.450166f, 0.549834f, 1.8506e-06f, 0.999998f,
                0.924142f, 0.0758582f, 0.00669285f, 0.993307f, 0.992609f, 0.00739154f, 1.12535e-07f, 1.0f,
                0.970688f, 0.0293122f, 0.450166f, 0.549834f, 0.869892f, 0.130108f, 0.689974f, 0.310025f,
                0.5f, 0.5f, 2.50452e-07f, 1.0f, 0.710949f, 0.28905f, 6.80798e-07f, 0.999999f})
        }
    };
    return result;
}

template<typename T>
float getError();

template<>
float getError<float>() {
    return 0.001;
}

template<>
float getError<ov::float16>() {
    return 0.2;
}

struct PrintToStringParamName {
    template<class T>
    std::string operator()(const testing::TestParamInfo<SoftmaxParamsWithFormat<T> > &param) {
        std::stringstream buf;
        SoftmaxParams<T> p;
        format::type plain_format;
        format::type target_format;
        std::tie(p, plain_format, target_format) = param.param;
        buf << "_inputTensor=" << p.input_tensor.to_string()
            << "_axis=" << p.axis
            << "_plainFormat=" << fmt_to_str(plain_format)
            << "_targetFormat=" << fmt_to_str(target_format);
        return buf.str();
    }
};
}; // namespace



template<typename T>
struct softmax_gpu_formats_test
        : public ::testing::TestWithParam<SoftmaxParamsWithFormat<T> > {
public:
    void test(bool is_caching_test) {
        const auto data_type = ov::element::from<T>();
        SoftmaxParams<T> params;
        format::type plain_format;
        format::type target_format;

        std::tie(params, plain_format, target_format) = this->GetParam();

        auto& engine = get_test_engine();
        const auto input = engine.allocate_memory({data_type, plain_format, params.input_tensor});

        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(reorder("reordered_input", input_info("input"), target_format, data_type));
        topology.add(softmax("blocked_softmax", input_info("reordered_input"), params.axis));
        topology.add(reorder("softmax", input_info("blocked_softmax"), plain_format, data_type));

        set_values(input, params.input);

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::optimize_data(false));
        cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);

        network->set_input_data("input", input);
        const auto outputs = network->execute();
        const auto output = outputs.at("softmax").get_memory();
        const cldnn::mem_lock<T> output_ptr(output, get_test_stream());

        ASSERT_EQ(params.input_tensor.count(), output_ptr.size());
        for (uint32_t i = 0; i < output_ptr.size(); i++) {
            ASSERT_NEAR(output_ptr[i], params.expected[i], getError<T>()) << "target_format=" << target_format << ", i=" << i;
        }
    }
};

using softmax_gpu_formats_test_f32 = softmax_gpu_formats_test<float>;
using softmax_gpu_formats_test_f16 = softmax_gpu_formats_test<ov::float16>;

TEST_P(softmax_gpu_formats_test_f32, softmax_gpu_formats_test_f32) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

TEST_P(softmax_gpu_formats_test_f16, softmax_gpu_formats_test_f16) {
    ASSERT_NO_FATAL_FAILURE(test(false));
}

INSTANTIATE_TEST_SUITE_P(softmax_gpu_formats_test_f32_2d,
                         softmax_gpu_formats_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateSoftmaxParams2D<float>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(formats2D)
                                 ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(softmax_gpu_formats_test_f16_2d,
                         softmax_gpu_formats_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateSoftmaxParams2D<ov::float16>()),
                                 ::testing::Values(format::bfyx),
                                 ::testing::ValuesIn(formats2D)
                                 ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(softmax_gpu_formats_test_f32_3d,
                         softmax_gpu_formats_test_f32,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateSoftmaxParams3D<float>()),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(formats3D)
                                 ),
                         PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(softmax_gpu_formats_test_f16_3d,
                         softmax_gpu_formats_test_f16,
                         ::testing::Combine(
                                 ::testing::ValuesIn(generateSoftmaxParams3D<ov::float16>()),
                                 ::testing::Values(format::bfzyx),
                                 ::testing::ValuesIn(formats3D)
                                 ),
                         PrintToStringParamName());

TEST(softmax_gpu_bfyx_f32, normalize_f_dynamic) {
    auto& engine = get_test_engine();

    const int64_t x = 2, y = 2, f = 3, b = 2;
    const int64_t buf_size = b*f*y*x;
    auto input_layout_dynamic = layout{ov::PartialShape::dynamic(4), data_types::f32, format::bfyx};
    auto input_layout_static = layout{ov::PartialShape{b, f, y, x}, data_types::f32, format::bfyx};

    auto input = engine.allocate_memory(input_layout_static);
    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(softmax("softmax", input_info("input"), 1));

    vector<float> input_vec = {
        //y0x0  y0x1   y1x0    y1x1
        /*b0f0*/0.1f, -0.1f, 0.9f,  1.5f,
        /*b0f1*/0.2f, 0.2f,  -10.f, 5.2f,
        /*b0f2*/0.2f, 0.2f,  -10.f, 5.2f,

        /*b1f0*/3.f,  0.5f,  7.f,   12.f,
        /*b1f1*/4.f,  0.5f,  8.f,   8.2f,
        /*b1f2*/0.2f, 0.2f,  -10.f, 5.2f
    };
    set_values(input, input_vec);

    float expected_max_values[8] = {
        0.344253346f, //b=0, y=0, x=0
        0.364854551f, //b=0, y=0, x=1

        0.999963085f, //b=0, y=1, x=0
        0.493894592f, //b=0, y=1, x=1

        0.719294981f, //b=1, y=0, x=0
        0.364854551f, //b=1, y=0, x=1

        0.73105857f, //b=1, y=1, x=0
        0.977054322f //b=1, y=1, x=1
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("softmax");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++) {
        out_buffer[i] = output_ptr[i];
    }

    float temp_max = 0;
    float expected_sum = 1.0f;
    int max_value_buffer_index = 0;
    for (uint32_t i = 0; i < b; i++) { //this for loops will sum results in a batch per feature, we expect that: sum = 1.0f
        for (uint32_t j = 0; j < y; j++) {
            for (uint32_t k = 0; k < x; k++) {
                float sum = 0.0f;
                for (uint32_t l = 0; l < f; l++) {
                    int index = i * f * x * y +
                                l * x * y +
                                j * x +
                                k;
                    if (out_buffer[index] >= temp_max) {
                        temp_max = out_buffer[index];
                    }
                    sum += out_buffer[index];
                }
                ASSERT_TRUE(are_equal(temp_max, expected_max_values[max_value_buffer_index]));
                temp_max = 0;
                max_value_buffer_index++;

                ASSERT_TRUE(are_equal(sum, expected_sum));
                sum = 0.0f;
            }
        }
    }
}

TEST_F(softmax_gpu_xb_f32_test_fixture, input_same_values_cached) {
    this->test_input_same_values(true);
}

#ifdef RUN_ALL_MODEL_CACHING_TESTS
TEST_F(softmax_gpu_xb_f32_test_fixture, input_same_values_batch_wise_cached) {
    this->test_input_same_values_batch_wise(true);
}

TEST_F(softmax_gpu_xb_f32_test_fixture, values_batch_wise_cached) {
    this->test_values_batch_wise(true);
}

TEST_P(softmax_test, SOFTMAX_cached) {
    run_single_test(true);
}

TEST_P(softmax_gpu_formats_test_f32, softmax_gpu_formats_test_f32_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}

TEST_P(softmax_gpu_formats_test_f16, softmax_gpu_formats_test_f16_cached) {
    ASSERT_NO_FATAL_FAILURE(test(true));
}
#endif

TEST(softmax_gpu_bfyx_f32, bf_opt_normalize_f_dynamic) {
    auto& engine = get_test_engine();

    const int64_t x = 1, y = 1, f = 3, b = 2;
    const int64_t buf_size = b*f*y*x;
    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), ov::Dimension::dynamic(), y, x},
                                       data_types::f32, format::bfyx};
    auto input_layout_static = layout{ov::PartialShape{b, f, y, x}, data_types::f32, format::bfyx};

    auto input = engine.allocate_memory(input_layout_static);
    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(softmax("softmax", input_info("input"), 1));

    vector<float> input_vec = {
              //y0x0
        /*b0f0*/0.1f,
        /*b0f1*/0.2f,
        /*b0f2*/0.2f,
        /*b1f0*/3.f,
        /*b1f1*/4.f,
        /*b1f2*/0.2f,
    };
    set_values(input, input_vec);

    float expected_max_values[2] = {
        0.344253346f, //b=0, y=0, x=0
        0.719294981f  //b=1, y=0, x=0
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    network network(engine, topology, config);
    network.set_input_data("input", input);

    auto inst = network.get_primitive("softmax");
    auto impl = inst->get_impl();
    ASSERT_TRUE(impl != nullptr);
    ASSERT_TRUE(impl->is_dynamic());

    auto outputs = network.execute();
    ASSERT_EQ(outputs.size(), size_t(1));
    ASSERT_EQ(outputs.begin()->first, "softmax");

    auto output = outputs.at("softmax").get_memory();
    cldnn::mem_lock<float> output_ptr(output, get_test_stream());
    float out_buffer[buf_size];
    for (uint32_t i = 0; i < buf_size; i++) {
        out_buffer[i] = output_ptr[i];
    }

    float temp_max = 0;
    float expected_sum = 1.0f;
    int max_value_buffer_index = 0;
    for (uint32_t i = 0; i < b; i++) { //this for loops will sum results in a batch per feature, we expect that: sum = 1.0f
        for (uint32_t j = 0; j < y; j++) {
            for (uint32_t k = 0; k < x; k++) {
                float sum = 0.0f;
                for (uint32_t l = 0; l < f; l++) {
                    int index = i * f * x * y +
                                l * x * y +
                                j * x +
                                k;
                    if (out_buffer[index] >= temp_max) {
                        temp_max = out_buffer[index];
                    }
                    sum += out_buffer[index];
                }
                ASSERT_TRUE(are_equal(temp_max, expected_max_values[max_value_buffer_index]));
                temp_max = 0;
                max_value_buffer_index++;

                ASSERT_TRUE(are_equal(sum, expected_sum));
                sum = 0.0f;
            }
        }
    }
}

static void run_softmax_bfyx_opt(const int64_t b, const int64_t f, const int64_t y, const int64_t x, const uint64_t axis) {
    tests::random_generator rg(GET_SUITE_NAME);
    auto& engine = get_test_engine();
    auto config = get_test_default_config(engine);
    config.set_property(ov::intel_gpu::optimize_data(true));
    config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
    ov::intel_gpu::ImplementationDesc softmax_bf_kernel = {format::bfyx, "softmax_gpu_bf"};
    config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{{"softmax", softmax_bf_kernel}}));

    const int64_t buf_size = b * f * y * x;
    auto input_layout_dynamic = layout{ov::PartialShape{ov::Dimension::dynamic(), f, ov::Dimension::dynamic(), ov::Dimension::dynamic()},
                                        data_types::f16, format::bfyx};
    auto input_layout_static = layout{ov::PartialShape{b, f, y, x}, data_types::f16, format::bfyx};

    std::string softmax_id = "softmax";
    topology topology;
    topology.add(input_layout("input", input_layout_dynamic));
    topology.add(softmax(softmax_id, input_info("input"), axis));

    cldnn::network::ptr network = get_network(engine, topology, config, get_test_stream_ptr(), false);

    auto input_mem = engine.allocate_memory(input_layout_static);

    auto input_data = rg.generate_random_1d<ov::float16>(buf_size, -20, 20);
    set_values(input_mem, input_data);

    std::map<cldnn::primitive_id, cldnn::network_output> outputs;
    cldnn::memory::ptr output = nullptr;

    network->set_input_data("input", input_mem);
    outputs = network->execute();
    output = outputs.at(softmax_id).get_memory();
    ASSERT_NE(output, nullptr);

    std::vector<ov::float16> output_ref(buf_size);
    ov::reference::softmax<ov::float16>(input_data.data(), output_ref.data(), input_layout_static.get_shape(), ov::AxisSet{axis});
    ASSERT_NE(output, nullptr);
    const float threshold_fp16 = 1e-1;
    cldnn::mem_lock<ov::float16> output_ptr(output, get_test_stream());
    for (size_t idx = 0; idx < static_cast<size_t>(buf_size); idx++) {
        ASSERT_NEAR(float(output_ptr[idx]), float(output_ref[idx]), threshold_fp16) << idx << ", " << std::fixed << setprecision(8) << output_ptr[idx] << " vs " << output_ref[idx];
    }
}

TEST(softmax_gpu_bfyx_f16, opt_softmax_bf_axis_3) {
    run_softmax_bfyx_opt(1, 4, 2, 3083, 3);
}
