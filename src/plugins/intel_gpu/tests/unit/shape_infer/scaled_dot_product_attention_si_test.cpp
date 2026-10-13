// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/scaled_dot_product_attention.hpp>

#include "program_wrapper.h"
#include "scaled_dot_product_attention_inst.h"
#include "test_utils.h"

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

struct sdpa_compressed_kv_test_params {
    ov::element::Type quantization_dt;
    ov::element::Type kv_cache_precision;
    int64_t kv_head_size;
    int64_t expected_head_size;
};

class sdpa_compressed_kv_test : public testing::TestWithParam<sdpa_compressed_kv_test_params> {};

TEST_P(sdpa_compressed_kv_test, shape_infer) {
    const auto& p = GetParam();

    auto& engine = get_test_engine();

    constexpr int64_t head_size = 64;
    const std::vector<layout> input_layouts = {
        layout{ov::PartialShape{1, 4, 8, head_size}, data_types::f16, format::bfyx},
        layout{ov::PartialShape{1, 2, 16, p.kv_head_size}, data_types::i8, format::bfyx},
        layout{ov::PartialShape{1, 2, 16, p.kv_head_size}, data_types::i8, format::bfyx},
        layout{ov::PartialShape{1, 2, 1, head_size}, data_types::f16, format::bfyx},
        layout{ov::PartialShape{1, 2, 1, head_size}, data_types::f16, format::bfyx},
    };

    ExecutionConfig config = get_test_default_config(engine);
    config.set_property(ov::hint::kv_cache_precision(p.kv_cache_precision));
    cldnn::program prog(engine, config);
    std::vector<std::shared_ptr<input_layout>> input_prims;
    std::vector<input_info> input_prim_ids;
    for (size_t i = 0; i < input_layouts.size(); ++i) {
        const auto prim_id = "data" + std::to_string(i);
        input_prims.push_back(std::make_shared<input_layout>(prim_id, input_layouts[i]));
        input_prim_ids.emplace_back(prim_id);
    }

    scaled_dot_product_attention::QuantizationAttributes qa;
    qa.quantization_type = ov::op::internal::DynamicQuantize::QuantizationType::Symmetric;
    qa.quantization_dt = p.quantization_dt;
    qa.scale_dt = ov::element::f16;
    qa.group_sizes = {1, 1, UINT64_MAX, 1};
    qa.scales_zp_output_order = {0, 1, 2, 3};
    const std::vector<int64_t> order = {0, 1, 2, 3};
    auto sdpa_prim = std::make_shared<scaled_dot_product_attention>("output", input_prim_ids, true, -1, order, order, order, order, qa, true);
    auto& sdpa_node = prog.get_or_create(sdpa_prim);
    for (const auto& input_prim : input_prims) {
        auto& input_node = prog.get_or_create(input_prim);
        program_wrapper::add_connection(prog, input_node, sdpa_node);
    }

    auto params = sdpa_node.get_kernel_impl_params();
    const auto result = scaled_dot_product_attention_inst::calc_output_layouts<ov::PartialShape>(sdpa_node, *params);

    ASSERT_EQ(result.size(), 1);
    ASSERT_EQ(result[0].get_partial_shape(), ov::PartialShape({1, 4, 8, p.expected_head_size}));
}

// Packed INT4 KV stores head_size / 2 bytes; the logical head size is restored from the primitive's
// quantization_dt even when the plugin KV-cache precision is not 4-bit (e.g. GQA u4 cache).
INSTANTIATE_TEST_SUITE_P(smoke,
                         sdpa_compressed_kv_test,
                         testing::ValuesIn(std::vector<sdpa_compressed_kv_test_params>{
                             {ov::element::u4, ov::element::f16, 32, 64},
                             {ov::element::i4, ov::element::f16, 32, 64},
                             {ov::element::i4, ov::element::i4, 32, 64},
                             {ov::element::i8, ov::element::f16, 64, 64},
                             {ov::element::i8, ov::element::i8, 64, 64},
                         }));

}  // namespace shape_infer_tests
