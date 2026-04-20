// Copyright (C) 2018-2026 Intel Corporation
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
    // These tests verify that primitive and parameter hashes are non-zero and
    // consistent between cached and non-cached networks.
    //
    // Limitation: because cldnn::hash_combine (runtime/utils.hpp) relies on std::hash<T>,
    // whose output is implementation-defined and varies across compilers and versions,
    // these tests cannot compare against fixed golden values. As a consequence, they will
    // not detect silent changes to the set of fields fed into a hash function (e.g. a new
    // field accidentally omitted from primitive::hash()). Sensitivity tests that mutate
    // individual fields and assert hash changes would cover that gap.
    struct hash_pair {
        size_t primitive_hash;
        size_t params_hash;
    };

    hash_pair get_hashes(cldnn::network& net, const std::string& key_prim_id) {
        const auto prim_inst = net.get_primitive(key_prim_id);
        const auto prim = prim_inst->desc();
        return { prim->hash(), prim_inst->get_impl_params()->hash() };
    }

    hash_pair get_fc_hashes(cldnn::network& net, const std::string& key_prim_id) {
        const auto prim_inst = net.get_primitive(key_prim_id);
        const auto prim = prim_inst->desc();
        return { prim->hash(), prim->type->get_fake_aligned_params(*prim_inst->get_impl_params()).hash() };
    }

    void verify_hash(const hash_pair& h, const std::string& label) {
        ASSERT_NE(h.primitive_hash, 0UL) << label << ": primitive hash should be non-zero";
        ASSERT_NE(h.params_hash, 0UL) << label << ": params hash should be non-zero";
    }

    void verify_hash_consistency(const hash_pair& ref, const hash_pair& cached, const std::string& label) {
        ASSERT_EQ(cached.primitive_hash, ref.primitive_hash)
            << label << ": primitive hash must be preserved across caching";
        ASSERT_EQ(cached.params_hash, ref.params_hash)
            << label << ": params hash must be preserved across caching";
    }

    void test_eltwise_basic(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto input1 = engine.allocate_memory({ { 2, 2, 2, 2 }, data_types::f32, format::bfyx });
        auto input2 = engine.allocate_memory({ { 2, 2, 2, 2 }, data_types::f32, format::bfyx });

        auto key_prim_id = "eltwise";
        topology topology;
        topology.add(input_layout("input", input1->get_layout()));
        topology.add(input_layout("input2", input2->get_layout()));
        topology.add(eltwise(key_prim_id, { input_info("input"), input_info("input2") }, eltwise_mode::sum));

        auto ref_net = std::make_shared<cldnn::network>(engine, topology, get_test_default_config(engine));
        auto ref = get_hashes(*ref_net, key_prim_id);
        verify_hash(ref, "eltwise");

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        auto h = get_hashes(*net, key_prim_id);
        verify_hash_consistency(ref, h, "eltwise");
    }

    void test_fc_basic(bool is_caching_test) {
        auto& engine = get_test_engine();

        const int32_t b = 1, in_f = 128, out_f = 65;

        auto input_prim = engine.allocate_memory({ { b, in_f }, data_types::f16, format::bfyx });
        auto weights_prim = engine.allocate_memory({ { out_f, in_f }, data_types::f16, format::bfyx });
        auto bias_prim = engine.allocate_memory({ { out_f }, data_types::f16, format::bfyx });

        const auto key_prim_id = "fc";
        topology topology(
            input_layout("input", input_prim->get_layout()),
            data("weights", weights_prim),
            data("bias", bias_prim),
            fully_connected(key_prim_id, input_info("input"), "weights", "bias")
        );

        auto ref_net = std::make_shared<cldnn::network>(engine, topology, get_test_default_config(engine));
        auto ref = get_fc_hashes(*ref_net, key_prim_id);
        verify_hash(ref, "fc");

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        auto h = get_fc_hashes(*net, key_prim_id);
        verify_hash_consistency(ref, h, "fc");
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

        auto ref_net = std::make_shared<cldnn::network>(engine, topology, get_test_default_config(engine));
        auto ref = get_hashes(*ref_net, key_prim_id);
        verify_hash(ref, "gather");

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        auto h = get_hashes(*net, key_prim_id);
        verify_hash_consistency(ref, h, "gather");
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

        auto ref_net = std::make_shared<cldnn::network>(engine, topology, get_test_default_config(engine));
        auto ref = get_hashes(*ref_net, key_prim_id);
        verify_hash(ref, "gemm");

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        auto h = get_hashes(*net, key_prim_id);
        verify_hash_consistency(ref, h, "gemm");
    }

    void test_permute_basic(bool is_caching_test) {
        auto& engine = get_test_engine();

        auto input = engine.allocate_memory({ { 2, 2, 3, 2 }, data_types::f32, format::bfyx });

        auto key_prim_id = "permute";
        topology topology(
            input_layout("input", input->get_layout()),
            permute(key_prim_id, input_info("input"), { 0, 1, 2, 3 }));

        auto ref_net = std::make_shared<cldnn::network>(engine, topology, get_test_default_config(engine));
        auto ref = get_hashes(*ref_net, key_prim_id);
        verify_hash(ref, "permute");

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        auto h = get_hashes(*net, key_prim_id);
        verify_hash_consistency(ref, h, "permute");
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

        auto ref_net = std::make_shared<cldnn::network>(engine, topology, get_test_default_config(engine));
        auto ref = get_hashes(*ref_net, key_prim_id);
        verify_hash(ref, "reorder");

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        auto h = get_hashes(*net, key_prim_id);
        verify_hash_consistency(ref, h, "reorder");
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
        auto reshape_prim = reshape(key_prim_id, input_info("reorder"), tensor( 1, 1, 4, 1 ), cldnn::reshape::reshape_mode::base);
        reshape_prim.output_paddings = {padding({0, 0, 2, 2})};
        topology.add(reshape_prim);

        auto ref_net = std::make_shared<cldnn::network>(engine, topology, get_test_default_config(engine));
        auto ref = get_hashes(*ref_net, key_prim_id);
        verify_hash(ref, "reshape");

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        auto h = get_hashes(*net, key_prim_id);
        verify_hash_consistency(ref, h, "reshape");
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

        auto ref_net = std::make_shared<cldnn::network>(engine, topology, get_test_default_config(engine));
        auto ref = get_hashes(*ref_net, key_prim_id);
        verify_hash(ref, "conv");

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        auto h = get_hashes(*net, key_prim_id);
        verify_hash_consistency(ref, h, "conv");
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

        auto ref_net = std::make_shared<cldnn::network>(engine, topology, get_test_default_config(engine));
        auto ref = get_hashes(*ref_net, key_prim_id);
        verify_hash(ref, "quantize");

        cldnn::network::ptr net = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), is_caching_test);
        auto h = get_hashes(*net, key_prim_id);
        verify_hash_consistency(ref, h, "quantize");
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
