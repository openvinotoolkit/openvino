// Copyright (C) 2023-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>

#include "test_utils.h"

#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/fully_connected.hpp>
#include <intel_gpu/primitives/crop.hpp>
#include <intel_gpu/primitives/gather.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/mvn.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/reshape.hpp>
#include <intel_gpu/primitives/quantize.hpp>


#include "primitive_inst.h"

using namespace cldnn;
using namespace tests;

class check_hash_value: public ::testing::Test {
public:
    void test_eltwise_basic(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ { 2, 2, 2, 2 }, data_types::f32, format::bfyx });
        auto input2 = engine.allocate_memory({ { 2, 2, 2, 2 }, data_types::f32, format::bfyx });

        auto key_prim_id = "eltwise";
        topology topology;
        topology.add(input_layout("input", input1->get_layout()));
        topology.add(input_layout("input2", input2->get_layout()));
        topology.add(eltwise(key_prim_id, { input_info("input"), input_info("input2") }, eltwise_mode::sum));

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        const auto  prim_inst = net->get_primitive(key_prim_id);
        const auto  primitve  = prim_inst->desc();

        const auto primitive_hash = primitve->hash();
        const auto params_hash = prim_inst->get_impl_params()->hash();

        ASSERT_EQ(primitive_hash, 4145865612957978777UL);
        ASSERT_EQ(params_hash, 13330229854511334999UL);
    }

    void test_fc_basic(bool is_caching_test) {
        auto& engine = get_test_engine();

        const int32_t b = 1, in_f = 128, out_f = 65;

        auto input_prim = engine.allocate_memory({ { b, in_f }, data_types::f32, format::bfyx });
        auto weights_prim = engine.allocate_memory({ { out_f, in_f }, data_types::f32, format::bfyx });
        auto bias_prim = engine.allocate_memory({ { out_f }, data_types::f32, format::bfyx });

        const auto key_prim_id = "fc";
        topology topology(
            input_layout("input", input_prim->get_layout()),
            data("weights", weights_prim),
            data("bias", bias_prim),
            fully_connected(key_prim_id, input_info("input"), "weights", "bias")
        );

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        const auto  prim_inst = net->get_primitive(key_prim_id);
        const auto  primitve  = prim_inst->desc();

        const auto primitive_hash = primitve->hash();
        const auto params_hash = primitve->type->get_fake_aligned_params(*prim_inst->get_impl_params()).hash();
        if (!engine.get_device_info().supports_immad) {
            ASSERT_EQ(primitive_hash, 14259723886449306729UL);
            ASSERT_EQ(params_hash, 3365957578641948513UL);
        } else {
            ASSERT_EQ(primitive_hash, 14259723886449306729UL);
            ASSERT_EQ(params_hash, 9831190959346679696UL);
        }
    }

    void test_gather_basic(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ { 3, 2, 2, 4, 3}, data_types::f16, format::bfzyx }); // Dictionary
        auto input2 = engine.allocate_memory({ { 3, 2, 1, 3 }, data_types::f32, format::bfyx }); // Indexes

        int64_t axis = 3;
        int64_t batch_dim = -1;
        bool negative_indexes = true;

        auto key_prim_id = "gather";
        topology topology;
        topology.add(input_layout("InputDictionary", input1->get_layout()));
        topology.add(input_layout("InputText", input2->get_layout()));
        topology.add(
            gather(key_prim_id, input_info("InputDictionary"), input_info("InputText"), axis, 5, ov::Shape{3, 2, 3, 3, 2}, batch_dim, negative_indexes)
        );

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        const auto  prim_inst = net->get_primitive(key_prim_id);
        const auto  primitve  = prim_inst->desc();

        const auto primitive_hash = primitve->hash();
        const auto params_hash = prim_inst->get_impl_params()->hash();

        ASSERT_EQ(primitive_hash, 8439414674502129643UL);
        ASSERT_EQ(params_hash, 9235751886952244871UL);
    }

    void test_gemm_basic(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto input = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 2, 4, 3 } });
        auto input2 = engine.allocate_memory({ data_types::f32, format::bfyx, { 1, 1, 4, 1 } });

        auto key_prim_id = "gemm";
        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        topology.add(input_layout("input2", input2->get_layout()));
        topology.add(crop("crop.1", input_info("input"), { 1, 1, 4, 3 }, { 0, 1, 0, 0 }));
        topology.add(gemm(key_prim_id, { input_info("crop.1"), input_info("input2") }, data_types::f32, false, true));

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        const auto  prim_inst = net->get_primitive(key_prim_id);
        const auto  primitve  = prim_inst->desc();

        const auto primitive_hash = primitve->hash();
        const auto params_hash = prim_inst->get_impl_params()->hash();
        ASSERT_EQ(primitive_hash, 15839977233203008631UL);
        ASSERT_EQ(params_hash, 15375157605915685928UL);
    }

    void test_permute_basic(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto input = engine.allocate_memory({ { 2, 2, 3, 2 }, data_types::f32, format::bfyx });

        auto key_prim_id = "permute";
        topology topology(
            input_layout("input", input->get_layout()),
            permute(key_prim_id, input_info("input"), { 0, 1, 2, 3 }));

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        const auto  prim_inst = net->get_primitive(key_prim_id);
        const auto  primitve  = prim_inst->desc();

        const auto primitive_hash = primitve->hash();
        const auto params_hash = prim_inst->get_impl_params()->hash();

        ASSERT_EQ(primitive_hash, 4658575237077439700UL);
        ASSERT_EQ(params_hash, 10588150284756843899UL);
    }

    void test_reorder_basic(bool is_caching_test) {
        auto& engine = get_test_engine();

        const int32_t b_in = 1;
        const int32_t f_in = 8 * 4;
        const int32_t y_in = 4;
        const int32_t x_in = 8 * 2;

        auto input = engine.allocate_memory({ { b_in,f_in,y_in,x_in }, data_types::f32, format::bfyx });
        layout output_layout({ b_in,f_in,y_in,x_in }, data_types::f32, format::b_fs_yx_fsv16);

        auto key_prim_id = "reorder";
        topology topology(
            input_layout("input", input->get_layout()),
            reorder(key_prim_id, input_info("input"), output_layout));

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        const auto  prim_inst = net->get_primitive(key_prim_id);
        const auto  primitve  = prim_inst->desc();

        const auto primitive_hash = primitve->hash();
        const auto params_hash = prim_inst->get_impl_params()->hash();

        ASSERT_EQ(primitive_hash, 16293979194373117693UL);
        ASSERT_EQ(params_hash, 15950979219660866859UL);
    }

    void test_reshape_basic(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto input = engine.allocate_memory({ { 1, 1, 2, 2 }, data_types::f32, format::bfyx });

        auto key_prim_id = "reshape";
        topology topology;
        topology.add(input_layout("input", input->get_layout()));
        auto padded_input_layout = input->get_layout();
        padded_input_layout.data_padding = padding();
        topology.add(reorder("reorder", input_info("input"), padded_input_layout));
        topology.add(reshape(key_prim_id, input_info("reorder"), tensor( 1, 1, 4, 1 ), cldnn::reshape::reshape_mode::base, padding({0, 0, 2, 2})));

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        const auto  prim_inst = net->get_primitive(key_prim_id);
        const auto  primitve  = prim_inst->desc();

        const auto primitive_hash = primitve->hash();
        const auto params_hash = prim_inst->get_impl_params()->hash();

        ASSERT_EQ(primitive_hash, 1534749073560581535UL);
        ASSERT_EQ(params_hash, 4349925423879269352UL);
    }

    void test_conv_basic(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto input = engine.allocate_memory({ { 1, 1, 4, 4, 4 }, data_types::f32, format::bfzyx });
        auto weights = engine.allocate_memory({ { 1, 1, 2, 2, 2 }, data_types::f32, format::bfzyx });
        auto biases = engine.allocate_memory({ { 1, 1, 1, 1 }, data_types::f32, format::bfyx });

        auto key_prim_id = "convolution";
        topology topology(
            input_layout("input", input->get_layout()),
            data("weights", weights),
            data("biases", biases),
            convolution(key_prim_id, input_info("input"), "weights", "biases", 1, {1, 1, 1}, {1, 1, 1}, {0, 0, 0}, {0, 0, 0}, false));

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        const auto  prim_inst = net->get_primitive(key_prim_id);
        const auto  primitve  = prim_inst->desc();

        const auto primitive_hash = primitve->hash();
        const auto params_hash = prim_inst->get_impl_params()->hash();

        ASSERT_EQ(primitive_hash, 13549661972131371304UL);
        ASSERT_EQ(params_hash, 7127098854451559675UL);
    }

    void test_quantize_basic(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto input = engine.allocate_memory({ {1, 16, 2, 2}, data_types::f32, format::bfyx });
        auto input_low = engine.allocate_memory({ { 1, 16, 1, 1 }, data_types::f32,format::bfyx });
        auto input_high = engine.allocate_memory({ { 1, 16, 1, 1 }, data_types::f32,format::bfyx });
        auto output_low = engine.allocate_memory({ { 1, 1, 1, 1 }, data_types::f32,format::bfyx });
        auto output_high = engine.allocate_memory({ { 1, 1, 1, 1 }, data_types::f32,format::bfyx });

        auto key_prim_id = "quantize";
        topology topology;
        topology.add(
            input_layout("input", input->get_layout()),
            data("input_low", input_low),
            data("input_high", input_high),
            data("output_low", output_low),
            data("output_high", output_high),
            quantize(key_prim_id, input_info("input"), input_info("input_low"), input_info("input_high"), input_info("output_low"), input_info("output_high"), 256, data_types::u8)
        );

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        const auto  prim_inst = net->get_primitive(key_prim_id);
        const auto  primitve  = prim_inst->desc();

        const auto primitive_hash = primitve->hash();
        const auto params_hash = prim_inst->get_impl_params()->hash();
        ASSERT_EQ(primitive_hash, 4135863035456568493UL);
        ASSERT_EQ(params_hash, 5990757629995899044UL);
    }
};

TEST_F(check_hash_value, eltwise_basic) {
    this->test_eltwise_basic(false);
}

TEST_F(check_hash_value, eltwise_basic_cached) {
    this->test_eltwise_basic(true);
}

TEST_F(check_hash_value, fc_basic) {
    this->test_fc_basic(false);
}

TEST_F(check_hash_value, fc_basic_cached) {
    this->test_fc_basic(true);
}

TEST_F(check_hash_value, gather_basic) {
    this->test_gather_basic(false);
}

TEST_F(check_hash_value, gather_basic_cached) {
    this->test_gather_basic(true);
}

TEST_F(check_hash_value, gemm_basic) {
    this->test_gemm_basic(false);
}

TEST_F(check_hash_value, gemm_basic_cached) {
    this->test_gemm_basic(true);
}

TEST_F(check_hash_value, permute_basic) {
    this->test_permute_basic(false);
}

TEST_F(check_hash_value, permute_basic_cached) {
    this->test_permute_basic(true);
}

TEST_F(check_hash_value, reorder_basic) {
    this->test_reorder_basic(false);
}

TEST_F(check_hash_value, reorder_basic_cached) {
    this->test_reorder_basic(true);
}

TEST_F(check_hash_value, reshape_basic) {
    this->test_reshape_basic(false);
}

TEST_F(check_hash_value, reshape_basic_cached) {
    this->test_reshape_basic(true);
}

TEST_F(check_hash_value, conv_basic) {
    this->test_conv_basic(false);
}

TEST_F(check_hash_value, conv_basic_cached) {
    this->test_conv_basic(true);
}

TEST_F(check_hash_value, quantize_basic) {
    this->test_quantize_basic(false);
}

TEST_F(check_hash_value, quantize_basic_cached) {
    this->test_quantize_basic(true);
}
