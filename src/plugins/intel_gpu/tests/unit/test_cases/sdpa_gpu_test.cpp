// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/runtime/debug_configuration.hpp>

#include "openvino/util/file_util.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <iostream>

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scaled_dot_product_attention.hpp>
#include "scaled_dot_product_attention_inst.h"

#include <cstddef>
#include <vector>

using namespace cldnn;
using namespace ::tests;

namespace  {
#ifdef ENABLE_ONEDNN_FOR_GPU
struct sdpa_test_params {
    int head_size;
    int num_heads;
    int sequence_length_q;
    int sequence_length_kv;
    int batch;
};

struct sdpa_gpu_test : public ::testing::TestWithParam<sdpa_test_params> {
    tests::random_generator rg;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void load_input(cldnn::memory::ptr mem, size_t idx) {
        auto shapes = mem->get_layout().get_shape();
        size_t size = ov::shape_size(shapes);
        auto input_data = rg.generate_random_1d<ov::float16>(size, -1.0f, 1.0f);
        set_values(mem, input_data);
    }

    cldnn::memory::ptr run_network(bool is_caching_test, bool use_micro_sdpa,
            cldnn::layout input0_dyn_layout,
            cldnn::layout input1_dyn_layout,
            cldnn::layout input2_dyn_layout,
            cldnn::layout input3_dyn_layout,
            cldnn::memory::ptr input0,
            cldnn::memory::ptr input1,
            cldnn::memory::ptr input2,
            cldnn::memory::ptr input3) {
        auto& engine = get_test_engine();
        topology topo;
        topo.add(input_layout("input0", input0_dyn_layout));
        topo.add(input_layout("input1", input1_dyn_layout));
        topo.add(input_layout("input2", input2_dyn_layout));
        topo.add(input_layout("input3", input3_dyn_layout));
        topo.add(scaled_dot_product_attention("sdpa", {input_info("input0"), input_info("input1"), input_info("input2"), input_info("input3")},
            false, -1, {0,2,1,3}, {0,2,1,3}, {0,2,1,3}, {0,1,2,3}, {}, false));
        topo.add(reorder("result",input_info("sdpa"), format::bfyx, data_types::f16));

        ExecutionConfig config = get_test_default_config(engine);
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        if (use_micro_sdpa) {
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"sdpa", {format::type::bfyx, "sdpa_micro"}} }));
        } else {
            config.set_property(ov::intel_gpu::force_implementations(ov::intel_gpu::ImplForcingMap{ {"sdpa", {format::type::bfyx, "sdpa_ref"}} }));
        }

        cldnn::network::ptr net = get_network(engine, topo, config, get_test_stream_ptr(), is_caching_test);

        net->set_input_data("input0", input0);
        net->set_input_data("input1", input1);
        net->set_input_data("input2", input2);
        net->set_input_data("input3", input3);

        auto outputs = net->execute();
        auto output = outputs.at("result").get_memory();
        return output;
    }

    void execute(sdpa_test_params& p, bool is_caching_test = false) {
        const auto head_size = p.head_size;
        const auto num_heads = p.num_heads;
        const auto seq_length_q = p.sequence_length_q;
        const auto seq_length_kv = p.sequence_length_kv;
        const auto batch = p.batch;

        auto& engine = get_test_engine();
        cldnn::layout input0_dyn_layout({-1, -1, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout input1_dyn_layout({-1, -1, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout input2_dyn_layout({-1, -1, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout input3_dyn_layout({-1, num_heads, -1, -1}, data_types::f16, format::bfyx);

        cldnn::layout input0_static_layout({batch, seq_length_q,  num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout input1_static_layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout input2_static_layout({batch, seq_length_kv, num_heads, head_size}, data_types::f16, format::bfyx);
        cldnn::layout input3_static_layout({batch, num_heads,     1,     seq_length_kv}, data_types::f16, format::bfyx);

        auto input0 = engine.allocate_memory(input0_static_layout);
        auto input1 = engine.allocate_memory(input1_static_layout);
        auto input2 = engine.allocate_memory(input2_static_layout);
        auto input3 = engine.allocate_memory(input3_static_layout);

        load_input(input0, 0);
        load_input(input1, 1);
        load_input(input2, 2);
        load_input(input3, 3);

        auto mem_ref_ptr = run_network(is_caching_test, false,
                                        input0_dyn_layout, input1_dyn_layout, input2_dyn_layout, input3_dyn_layout,
                                        input0, input1, input2, input3);
        auto mem_opt_ptr = run_network(is_caching_test, true,
                                        input0_dyn_layout, input1_dyn_layout, input2_dyn_layout, input3_dyn_layout,
                                        input0, input1, input2, input3);
        cldnn::mem_lock<ov::float16, mem_lock_type::read> ref_data(mem_ref_ptr, get_test_stream());
        cldnn::mem_lock<ov::float16, mem_lock_type::read> opt_data(mem_opt_ptr, get_test_stream());
        {
            for (size_t idx = 0; idx < ref_data.size(); idx++) {
                ASSERT_FALSE(std::isnan(opt_data[idx]) || std::isnan(ref_data[idx])) << "NaN found at index " << idx;
            }
            auto ret = cosineSimilarity(ref_data, opt_data);
            ASSERT_GE(ret, 0.95f);
        }
    }

    float cosineSimilarity(cldnn::mem_lock<ov::float16, mem_lock_type::read>& vec1, cldnn::mem_lock<ov::float16, mem_lock_type::read>& memLockVec2) {
        if (vec1.size() != memLockVec2.size()) {
            return -1.0f;
        }

        float dotProduct = std::inner_product(vec1.begin(), vec1.end(), memLockVec2.begin(), 0.0f);

        float magnitude1 = std::sqrt(std::inner_product(vec1.begin(), vec1.end(), vec1.begin(), 0.0f));
        float magnitude2 = std::sqrt(std::inner_product(memLockVec2.begin(), memLockVec2.end(), memLockVec2.begin(), 0.0f));

        if (magnitude1 == 0.0f || magnitude2 == 0.0f) {
            return -1.0f;
        }

        return dotProduct / (magnitude1 * magnitude2);
    }

    static std::string
    PrintToStringParamName(const testing::TestParamInfo<sdpa_test_params>& info) {
        return "sdpa_gpu_test_" + std::to_string(info.param.head_size) + "_" +
               std::to_string(info.param.num_heads) + "_" +
               std::to_string(info.param.sequence_length_q) + "_" +
               std::to_string(info.param.sequence_length_kv) + "_" +
               std::to_string(info.param.batch);
    }
};

INSTANTIATE_TEST_SUITE_P(
    smoke_sdpa_gpu_test,
    sdpa_gpu_test,
    ::testing::Values(
        sdpa_test_params{64, 32, 990, 128, 2}
    ),
    sdpa_gpu_test::PrintToStringParamName
);

TEST_P(sdpa_gpu_test, basic) {
    auto& engine = get_test_engine();
    if (!engine.get_device_info().supports_immad)
        return;
    auto p = GetParam();
    execute(p);
}
#endif
} // namespace
