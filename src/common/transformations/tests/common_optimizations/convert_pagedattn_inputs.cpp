// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/convert_pagedattn_inputs.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/op/paged_causal_conv1d.hpp"
#include "openvino/op/paged_gated_delta_net.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/paged_selective_ssm.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/sdpa_to_paged_attention.hpp"
#include "openvino/runtime/properties.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/paged_attention/paged_qsa_fusion.hpp"
#include "transformations/pattern_blocks/qsa_index_query_block.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::test;
using namespace ov;
using namespace ov::gen_pattern;
using ConvertPagedAttnInputsParams = std::tuple<std::vector<ov::element::Type>,  // cache_precision
                                                std::vector<size_t>,             // cache_group_size
                                                std::vector<size_t>,             // block_size
                                                ov::element::Type,               // infer_precsion
                                                bool,                            // quant_key_by_channel
                                                bool,                            // accuracy_mode
                                                bool                             // is_ir_kv_cache_f16
                                                >;

namespace v0 = ov::op::v0;
namespace {
class ConvertPagedAttnInputsTest : public TransformationTestsF,
                                   public testing::WithParamInterface<ConvertPagedAttnInputsParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertPagedAttnInputsParams>& obj) {
        const auto& [cachePrecision,
                     cacheGroupSize,
                     blcokSize,
                     inferPrec,
                     quantKeybychannel,
                     isAccuracyMode,
                     isIRKVCacheF16] = obj.param;
        std::ostringstream result;
        result << "KeyPrc=" << cachePrecision[0] << "_";
        result << "ValuePrc=" << cachePrecision[1] << "_";
        result << "KeyBlockSize=" << blcokSize[0] << "_";
        result << "ValueBlockSize=" << blcokSize[1] << "_";
        result << "InferPrec=" << inferPrec << "_";
        result << "KeyGS=" << cacheGroupSize[0] << "_";
        result << "ValueGS=" << cacheGroupSize[1] << "_";
        result << "KeyByChannel=" << quantKeybychannel << "_";
        result << "isAccuracyMode=" << isAccuracyMode << "_";
        result << "isIRKVCacheF16=" << isIRKVCacheF16;
        return result.str();
    }

public:
    size_t keyHeadSize;
    size_t valueHeadSize;
    size_t numKeyHeads;
    size_t numValueHeads;
    ov::element::Type inferPrec;
    ov::element::Type keyCachePrecision;
    ov::element::Type valueCachePrecision;
    std::vector<size_t> blockSize;
    size_t keyCacheGroupSize;
    size_t valueCacheGroupSize;
    bool quantKeybychannel;
    bool isAccuracyMode;
    bool isIRKVCacheF16;
};

TEST_P(ConvertPagedAttnInputsTest, checkPrecisionAndShape) {
    const auto& [cachePrecision,
                 cacheGroupSize,
                 _blockSize,
                 _inferPrec,
                 _quantKeybychannel,
                 _isAccuracyMode,
                 _isIRKVCacheF16] = this->GetParam();
    blockSize = _blockSize;
    inferPrec = _inferPrec;
    quantKeybychannel = _quantKeybychannel;
    isAccuracyMode = _isAccuracyMode;
    isIRKVCacheF16 = _isIRKVCacheF16;
    keyCachePrecision = cachePrecision[0];
    valueCachePrecision = cachePrecision[1];
    keyCacheGroupSize = cacheGroupSize[0];
    valueCacheGroupSize = cacheGroupSize[1];
    numKeyHeads = 2;
    keyHeadSize = 32;

    numValueHeads = 2;
    valueHeadSize = 64;
    {
        auto Q = std::make_shared<v0::Parameter>(ov::element::f32, PartialShape{-1, 4 * 32});
        auto K = std::make_shared<v0::Parameter>(
            ov::element::f32,
            PartialShape{-1, static_cast<ov::Dimension::value_type>(numKeyHeads * keyHeadSize)});
        auto V = std::make_shared<v0::Parameter>(
            ov::element::f32,
            PartialShape{-1, static_cast<ov::Dimension::value_type>(numValueHeads * valueHeadSize)});
        auto max_context_len = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{});
        auto block_indices_begins = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto block_indices = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto subsequence_begins = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto past_lens = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto key_cache_0 = std::make_shared<v0::Parameter>(ov::element::dynamic, PartialShape::dynamic(4));
        auto value_cache_0 = std::make_shared<v0::Parameter>(ov::element::dynamic, PartialShape::dynamic(4));
        auto scale = std::make_shared<v0::Constant>(element::f32, Shape{}, 0.5f);
        auto sliding_window = std::make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto alibi_slopes = std::make_shared<v0::Constant>(element::f32, Shape{0});
        auto score_aggregation_window = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto rotated_block_indices = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto rotation_deltas = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto rotation_trig_lut = std::make_shared<v0::Parameter>(ov::element::f32, PartialShape{DYN});
        auto xattention_threshold = std::make_shared<v0::Parameter>(ov::element::f32, PartialShape{DYN});
        auto xattention_block_size = std::make_shared<v0::Parameter>(ov::element::i32, Shape{});
        auto xattention_stride = std::make_shared<v0::Parameter>(ov::element::i32, Shape{});
        auto sinks = std::make_shared<v0::Constant>(element::f32, Shape{0, 0, 0, 0});
        auto adaptive_rkv_start_size = std::make_shared<v0::Parameter>(ov::element::i32, Shape{});
        auto adaptive_rkv_evictable_sizes = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto adaptive_rkv_diversity_block_set_indices =
            std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto adaptive_rkv_diversity_block_set_indices_begins =
            std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto token_type_ids = std::make_shared<op::v0::Parameter>(ov::element::i32, ov::Shape{0});

        auto qq_bias = std::make_shared<v0::Parameter>(ov::element::u8, PartialShape{DYN});
        auto qq_bias_begins = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto pa =
            std::make_shared<op::PagedAttentionExtension>(OutputVector{Q,
                                                                       K,
                                                                       V,
                                                                       key_cache_0,
                                                                       value_cache_0,
                                                                       past_lens,
                                                                       subsequence_begins,
                                                                       block_indices,
                                                                       block_indices_begins,
                                                                       scale,
                                                                       sliding_window,
                                                                       alibi_slopes,
                                                                       max_context_len,
                                                                       score_aggregation_window,
                                                                       rotated_block_indices,
                                                                       rotation_deltas,
                                                                       rotation_trig_lut,
                                                                       xattention_threshold,
                                                                       xattention_block_size,
                                                                       xattention_stride,
                                                                       sinks,
                                                                       adaptive_rkv_start_size,
                                                                       adaptive_rkv_evictable_sizes,
                                                                       adaptive_rkv_diversity_block_set_indices,
                                                                       adaptive_rkv_diversity_block_set_indices_begins,
                                                                       token_type_ids,
                                                                       qq_bias,
                                                                       qq_bias_begins});
        pa->get_rt_info()["num_k_heads"] = numKeyHeads;
        pa->get_rt_info()["k_head_size"] = keyHeadSize;
        pa->get_rt_info()["num_v_heads"] = numValueHeads;
        pa->get_rt_info()["v_head_size"] = valueHeadSize;

        model = std::make_shared<ov::Model>(ov::OutputVector{pa},
                                            ov::ParameterVector{Q,
                                                                K,
                                                                V,
                                                                key_cache_0,
                                                                value_cache_0,
                                                                past_lens,
                                                                subsequence_begins,
                                                                block_indices,
                                                                block_indices_begins,
                                                                max_context_len,
                                                                score_aggregation_window,
                                                                rotated_block_indices,
                                                                rotation_deltas,
                                                                rotation_trig_lut,
                                                                xattention_threshold,
                                                                xattention_block_size,
                                                                xattention_stride,
                                                                adaptive_rkv_start_size,
                                                                adaptive_rkv_evictable_sizes,
                                                                adaptive_rkv_diversity_block_set_indices,
                                                                adaptive_rkv_diversity_block_set_indices_begins,
                                                                token_type_ids,
                                                                qq_bias,
                                                                qq_bias_begins});

        if (isIRKVCacheF16) {
            model->set_rt_info("f16", "runtime_options", ov::hint::kv_cache_precision.name());
        }
    }

    {
        auto getCachePrec = [&](ov::element::Type tensorPrec) {
            tensorPrec =
                tensorPrec == ov::element::f16 && inferPrec == ov::element::bf16 ? ov::element::bf16 : tensorPrec;
            if (isAccuracyMode)
                tensorPrec = ov::element::f32;
            return tensorPrec;
        };
        auto getCacheShape = [&](ov::element::Type cachePrec,
                                 size_t headNums,
                                 size_t headSize,
                                 size_t groupSize,
                                 size_t blockSize,
                                 bool quantBychannel) {
            auto targeShape = PartialShape::dynamic(4);
            targeShape[1] = headNums;
            groupSize = groupSize ? groupSize : headSize;
            const size_t paramSize = 2 * sizeof(float) * 8 / cachePrec.bitwidth();
            if (!cachePrec.is_integral()) {
                targeShape[2] = blockSize;
                targeShape[3] = headSize;
            } else if (quantBychannel) {
                targeShape[2] = paramSize + blockSize;
                targeShape[3] = headSize;
            } else {
                size_t groupNum = headSize / groupSize;
                targeShape[2] = blockSize;
                targeShape[3] = headSize + paramSize * groupNum;
            }
            return targeShape;
        };

        keyCachePrecision = getCachePrec(keyCachePrecision);
        valueCachePrecision = getCachePrec(valueCachePrecision);
        auto Q = std::make_shared<v0::Parameter>(ov::element::f32, PartialShape{-1, 4 * 32});
        auto K = std::make_shared<v0::Parameter>(
            ov::element::f32,
            PartialShape{-1, static_cast<ov::Dimension::value_type>(numKeyHeads * keyHeadSize)});
        auto V = std::make_shared<v0::Parameter>(
            ov::element::f32,
            PartialShape{-1, static_cast<ov::Dimension::value_type>(numValueHeads * valueHeadSize)});
        auto max_context_len = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{});
        auto block_indices_begins = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto block_indices = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto subsequence_begins = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto past_lens = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto key_cache_0 = std::make_shared<v0::Parameter>(keyCachePrecision,
                                                           getCacheShape(keyCachePrecision,
                                                                         numKeyHeads,
                                                                         keyHeadSize,
                                                                         keyCacheGroupSize,
                                                                         blockSize[0],
                                                                         quantKeybychannel));
        auto value_cache_0 = std::make_shared<v0::Parameter>(
            valueCachePrecision,
            getCacheShape(valueCachePrecision, numKeyHeads, valueHeadSize, valueCacheGroupSize, blockSize[1], false));
        auto scale = std::make_shared<v0::Constant>(element::f32, Shape{}, 0.5f);
        auto sliding_window = std::make_shared<v0::Constant>(element::i32, Shape{}, 0);
        auto alibi_slopes = std::make_shared<v0::Constant>(element::f32, Shape{0});
        auto score_aggregation_window = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto rotated_block_indices = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto rotation_deltas = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto rotation_trig_lut = std::make_shared<v0::Parameter>(ov::element::f32, PartialShape{DYN});
        auto xattention_threshold = std::make_shared<v0::Parameter>(ov::element::f32, PartialShape{DYN});
        auto xattention_block_size = std::make_shared<v0::Parameter>(ov::element::i32, Shape{});
        auto xattention_stride = std::make_shared<v0::Parameter>(ov::element::i32, Shape{});
        auto sinks = std::make_shared<v0::Constant>(element::f32, Shape{0, 0, 0, 0});
        auto adaptive_rkv_start_size = std::make_shared<v0::Parameter>(ov::element::i32, Shape{});
        auto adaptive_rkv_evictable_sizes = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto adaptive_rkv_diversity_block_set_indices =
            std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto adaptive_rkv_diversity_block_set_indices_begins =
            std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto token_type_ids = std::make_shared<v0::Parameter>(ov::element::i32, ov::Shape{0});

        auto qq_bias = std::make_shared<v0::Parameter>(ov::element::u8, PartialShape{DYN});
        auto qq_bias_begins = std::make_shared<v0::Parameter>(ov::element::i32, PartialShape{DYN});
        auto pa =
            std::make_shared<op::PagedAttentionExtension>(OutputVector{Q,
                                                                       K,
                                                                       V,
                                                                       key_cache_0,
                                                                       value_cache_0,
                                                                       past_lens,
                                                                       subsequence_begins,
                                                                       block_indices,
                                                                       block_indices_begins,
                                                                       scale,
                                                                       sliding_window,
                                                                       alibi_slopes,
                                                                       max_context_len,
                                                                       score_aggregation_window,
                                                                       rotated_block_indices,
                                                                       rotation_deltas,
                                                                       rotation_trig_lut,
                                                                       xattention_threshold,
                                                                       xattention_block_size,
                                                                       xattention_stride,
                                                                       sinks,
                                                                       adaptive_rkv_start_size,
                                                                       adaptive_rkv_evictable_sizes,
                                                                       adaptive_rkv_diversity_block_set_indices,
                                                                       adaptive_rkv_diversity_block_set_indices_begins,
                                                                       token_type_ids,
                                                                       qq_bias,
                                                                       qq_bias_begins});
        pa->get_rt_info()["num_k_heads"] = numKeyHeads;
        pa->get_rt_info()["k_head_size"] = keyHeadSize;
        pa->get_rt_info()["num_v_heads"] = numValueHeads;
        pa->get_rt_info()["v_head_size"] = valueHeadSize;

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{pa},
                                                ov::ParameterVector{Q,
                                                                    K,
                                                                    V,
                                                                    key_cache_0,
                                                                    value_cache_0,
                                                                    past_lens,
                                                                    subsequence_begins,
                                                                    block_indices,
                                                                    block_indices_begins,
                                                                    max_context_len,
                                                                    score_aggregation_window,
                                                                    rotated_block_indices,
                                                                    rotation_deltas,
                                                                    rotation_trig_lut,
                                                                    xattention_threshold,
                                                                    xattention_block_size,
                                                                    xattention_stride,
                                                                    adaptive_rkv_start_size,
                                                                    adaptive_rkv_evictable_sizes,
                                                                    adaptive_rkv_diversity_block_set_indices,
                                                                    adaptive_rkv_diversity_block_set_indices_begins,
                                                                    token_type_ids,
                                                                    qq_bias,
                                                                    qq_bias_begins});
    }
    ov::pass::ConvertPagedAttnInputs::KVCacheConfig cacheConfig;
    cacheConfig.keyCacheBlockSize = blockSize[0];
    cacheConfig.valueCacheBlockSize = blockSize[1];
    if (isAccuracyMode) {
        cacheConfig.inferencePrecision = ov::element::f32;
        cacheConfig.keyCachePrecision = ov::element::f32;
        cacheConfig.valueCachePrecision = ov::element::f32;
    } else {
        cacheConfig.keyCachePrecision = keyCachePrecision;
        cacheConfig.valueCachePrecision = valueCachePrecision;
        cacheConfig.keyCacheGroupSize = keyCacheGroupSize;
        cacheConfig.valueCacheGroupSize = valueCacheGroupSize;
    }
    auto update_paged_attention_shape_func = [](const ov::element::Type& precision,
                                                const bool bychannel,
                                                const size_t group_num,
                                                int64_t& head_size,
                                                int64_t& block_size) {
        if (precision == ov::element::u8) {
            if (bychannel) {
                block_size += 2 * sizeof(float);
            } else {
                head_size += sizeof(float) * 2 * group_num;
            }
        } else if (precision == ov::element::u4) {
            head_size += sizeof(float) * 2 * group_num * 2;
        }
    };

    manager.register_pass<ov::pass::ConvertPagedAttnInputs>(cacheConfig, update_paged_attention_shape_func);
    comparator.disable(FunctionsComparator::ACCURACY);
    comparator.disable(FunctionsComparator::RUNTIME_KEYS);
    disable_result_friendly_names_check();
    disable_rt_info_check();
}

std::vector<std::vector<ov::element::Type>> get_cache_prec() {
    return {
        {ov::element::f32, ov::element::f32},
        {ov::element::f16, ov::element::f16},
        {ov::element::u8, ov::element::u8},
        {ov::element::u8, ov::element::u4},
        {ov::element::u4, ov::element::u4},
    };
}

// group size
const std::vector<std::vector<size_t>> cache_gs = {{32, 16}, {0, 0}};

// block size
const std::vector<std::vector<size_t>> cache_bs = {{32, 16}};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertPagedAttnInputsTest,
                         ConvertPagedAttnInputsTest,
                         ::testing::Combine(::testing::ValuesIn(get_cache_prec()),
                                            ::testing::ValuesIn(cache_gs),
                                            ::testing::ValuesIn(cache_bs),
                                            ::testing::Values(ov::element::f32, ov::element::bf16),
                                            ::testing::Values(false),
                                            ::testing::Values(true, false),
                                            ::testing::Values(true, false)),
                         ConvertPagedAttnInputsTest::getTestCaseName);

class ConvertPagedAttnInputsStateTableTest : public testing::Test {};

// The tests bellow replicate the behavior of ConvertPagedAttnInputs pass in a GPU pipeline.
// Testing it in isolation does not really make sense because in a real pipeline the model is
// different and it's important to replicate the same conditions that a model is going through.
// TODO: consider moving ConvertPagedAttnInputs above in the pipeline so we avoid taking into account
// the ConvertPrecision pass that much.
TEST_F(ConvertPagedAttnInputsStateTableTest, ConvertPagedCausalConv1DInputsPrecision) {
    auto input_embeds = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 256});
    auto conv_state_table = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{-1, 256, 4});
    conv_state_table->set_friendly_name("conv_state_table.0");
    ov::enable_keep_const_precision(conv_state_table);
    auto conv_weight = std::make_shared<v0::Parameter>(element::f32, PartialShape{256, 1, 4});
    auto conv_bias = std::make_shared<v0::Parameter>(element::f32, PartialShape{256});
    auto subsequence_begins = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto block_indices = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto block_indices_begins = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto past_lens = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto cache_interval = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});

    auto paged_conv = std::make_shared<op::internal::PagedCausalConv1D>(input_embeds,
                                                                        conv_state_table,
                                                                        conv_weight,
                                                                        conv_bias,
                                                                        subsequence_begins,
                                                                        block_indices,
                                                                        block_indices_begins,
                                                                        past_lens,
                                                                        cache_interval);
    auto local_model = std::make_shared<Model>(OutputVector{paged_conv},
                                               ParameterVector{input_embeds,
                                                               conv_state_table,
                                                               conv_weight,
                                                               conv_bias,
                                                               subsequence_begins,
                                                               block_indices,
                                                               block_indices_begins,
                                                               past_lens,
                                                               cache_interval});

    // Replicate GPU pipeline: clear keep_const_precision on all nodes
    for (auto& node : local_model->get_ops()) {
        ov::disable_keep_const_precision(node);
    }

    ov::pass::ConvertPagedAttnInputs::KVCacheConfig cacheConfig;
    cacheConfig.inferencePrecision = ov::element::f16;

    ov::pass::Manager local_manager;

    precisions_map fp_convert_precision_map = {{ov::element::f64, ov::element::f32},
                                               {ov::element::f32, ov::element::f16}};
    type_to_fuse_map empty_fuse_map = {};
    local_manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map,
                                                            empty_fuse_map,
                                                            /*keep_precision_sensitive_in_fp32*/ true,
                                                            /*convert_input_output_precision*/ false,
                                                            /*store_original_precision_as_rt_attribute*/ true);

    auto update_paged_attention_shape_func =
        [](const ov::element::Type&, const bool, const size_t, int64_t&, int64_t&) {};
    local_manager.register_pass<ov::pass::ConvertPagedAttnInputs>(cacheConfig, update_paged_attention_shape_func);
    local_manager.run_passes(local_model);

    // Verify no Convert node between Parameter and PagedCausalConv1D
    EXPECT_TRUE(ov::is_type<v0::Parameter>(paged_conv->get_input_node_shared_ptr(1)));
    EXPECT_EQ(conv_state_table->get_element_type(), ov::element::f16);
}

TEST_F(ConvertPagedAttnInputsStateTableTest, ConvertPagedGatedDeltaNetPrecision) {
    auto query = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 2, 64});
    auto key = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 2, 64});
    auto value = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 2, 64});
    auto gated_delta_state_table = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{-1, 2, 64, 64});
    gated_delta_state_table->set_friendly_name("gated_delta_state_table.0");
    enable_keep_const_precision(gated_delta_state_table);
    auto gate = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 2});
    auto beta = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 2});
    auto subsequence_begins = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto block_indices = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto block_indices_begins = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto past_lens = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto cache_interval = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});

    auto paged_gdn = std::make_shared<op::internal::PagedGatedDeltaNet>(query,
                                                                        key,
                                                                        value,
                                                                        gated_delta_state_table,
                                                                        gate,
                                                                        beta,
                                                                        subsequence_begins,
                                                                        block_indices,
                                                                        block_indices_begins,
                                                                        past_lens,
                                                                        cache_interval);
    auto local_model = std::make_shared<Model>(OutputVector{paged_gdn},
                                               ParameterVector{query,
                                                               key,
                                                               value,
                                                               gated_delta_state_table,
                                                               gate,
                                                               beta,
                                                               subsequence_begins,
                                                               block_indices,
                                                               block_indices_begins,
                                                               past_lens,
                                                               cache_interval});

    // Replicate GPU pipeline: clear keep_const_precision on all nodes
    for (auto& node : local_model->get_ops()) {
        ov::disable_keep_const_precision(node);
    }

    ov::pass::ConvertPagedAttnInputs::KVCacheConfig cacheConfig;
    cacheConfig.inferencePrecision = ov::element::f16;

    ov::pass::Manager local_manager;

    precisions_map fp_convert_precision_map = {{ov::element::f64, ov::element::f32},
                                               {ov::element::f32, ov::element::f16}};
    type_to_fuse_map empty_fuse_map = {};
    local_manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map,
                                                            empty_fuse_map,
                                                            /*keep_precision_sensitive_in_fp32*/ true,
                                                            /*convert_input_output_precision*/ false,
                                                            /*store_original_precision_as_rt_attribute*/ true);

    auto update_paged_attention_shape_func =
        [](const ov::element::Type&, const bool, const size_t, int64_t&, int64_t&) {};
    local_manager.register_pass<ov::pass::ConvertPagedAttnInputs>(cacheConfig, update_paged_attention_shape_func);
    local_manager.run_passes(local_model);

    // Verify no Convert node between Parameter and PagedGatedDeltaNet
    EXPECT_TRUE(ov::is_type<v0::Parameter>(paged_gdn->get_input_node_shared_ptr(3)));
    EXPECT_EQ(gated_delta_state_table->get_element_type(), ov::element::f16);
}

TEST_F(ConvertPagedAttnInputsStateTableTest, ConvertPagedSelectiveSSMPrecision) {
    auto A = std::make_shared<v0::Parameter>(element::f32, PartialShape{4});
    auto dt = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 4});
    auto B = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 2, 16});
    auto x = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 4, 8});
    auto C = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 2, 16});
    auto selective_ssm_state_table = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{-1, 4, 8, 16});
    selective_ssm_state_table->set_friendly_name("selective_ssm_state_table.0");
    enable_keep_const_precision(selective_ssm_state_table);
    auto subsequence_begins = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto block_indices = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto block_indices_begins = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto num_processed_tokens = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto cache_interval = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});

    auto paged_ssm = std::make_shared<op::internal::PagedSelectiveSSM>(A,
                                                                       dt,
                                                                       B,
                                                                       x,
                                                                       C,
                                                                       selective_ssm_state_table,
                                                                       subsequence_begins,
                                                                       block_indices,
                                                                       block_indices_begins,
                                                                       num_processed_tokens,
                                                                       cache_interval);
    auto local_model = std::make_shared<Model>(OutputVector{paged_ssm},
                                               ParameterVector{A,
                                                               dt,
                                                               B,
                                                               x,
                                                               C,
                                                               selective_ssm_state_table,
                                                               subsequence_begins,
                                                               block_indices,
                                                               block_indices_begins,
                                                               num_processed_tokens,
                                                               cache_interval});

    // Replicate GPU pipeline: clear keep_const_precision on all nodes
    for (auto& node : local_model->get_ops()) {
        ov::disable_keep_const_precision(node);
    }

    ov::pass::ConvertPagedAttnInputs::KVCacheConfig cacheConfig;
    cacheConfig.inferencePrecision = ov::element::f16;

    ov::pass::Manager local_manager;

    precisions_map fp_convert_precision_map = {{ov::element::f64, ov::element::f32},
                                               {ov::element::f32, ov::element::f16}};
    type_to_fuse_map empty_fuse_map = {};
    local_manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map,
                                                            empty_fuse_map,
                                                            /*keep_precision_sensitive_in_fp32*/ true,
                                                            /*convert_input_output_precision*/ false,
                                                            /*store_original_precision_as_rt_attribute*/ true);

    auto update_paged_attention_shape_func =
        [](const ov::element::Type&, const bool, const size_t, int64_t&, int64_t&) {};
    local_manager.register_pass<ov::pass::ConvertPagedAttnInputs>(cacheConfig, update_paged_attention_shape_func);
    local_manager.run_passes(local_model);

    for (size_t input = 0; input < 5; ++input) {
        EXPECT_EQ(paged_ssm->get_input_element_type(input), element::f16);
    }
    // Verify no Convert node between Parameter and PagedSelectiveSSM
    EXPECT_TRUE(ov::is_type<v0::Parameter>(paged_ssm->get_input_node_shared_ptr(5)));
    EXPECT_EQ(selective_ssm_state_table->get_element_type(), ov::element::f16);
    EXPECT_EQ(paged_ssm->get_output_element_type(0), element::f16);
}

TEST_F(ConvertPagedAttnInputsStateTableTest, ConvertPagedSelectiveSSMIndependentStatePrecision) {
    auto A = std::make_shared<v0::Parameter>(element::f32, PartialShape{4});
    auto dt = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 4});
    auto B = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 2, 16});
    auto x = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 4, 8});
    auto C = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, 2, 16});
    auto selective_ssm_state_table = std::make_shared<v0::Parameter>(element::dynamic, PartialShape{-1, 4, 8, 16});
    selective_ssm_state_table->set_friendly_name("selective_ssm_state_table.0");
    for (const auto& parameter : {A, dt, B, x, C, selective_ssm_state_table}) {
        enable_keep_const_precision(parameter);
    }
    auto subsequence_begins = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto block_indices = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto block_indices_begins = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto num_processed_tokens = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto cache_interval = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});

    auto paged_ssm = std::make_shared<op::internal::PagedSelectiveSSM>(A,
                                                                       dt,
                                                                       B,
                                                                       x,
                                                                       C,
                                                                       selective_ssm_state_table,
                                                                       subsequence_begins,
                                                                       block_indices,
                                                                       block_indices_begins,
                                                                       num_processed_tokens,
                                                                       cache_interval);
    auto local_model = std::make_shared<Model>(OutputVector{paged_ssm},
                                               ParameterVector{A,
                                                               dt,
                                                               B,
                                                               x,
                                                               C,
                                                               selective_ssm_state_table,
                                                               subsequence_begins,
                                                               block_indices,
                                                               block_indices_begins,
                                                               num_processed_tokens,
                                                               cache_interval});

    ov::pass::ConvertPagedAttnInputs::KVCacheConfig cacheConfig;
    cacheConfig.inferencePrecision = ov::element::f16;

    ov::pass::Manager local_manager;

    precisions_map fp_convert_precision_map = {{ov::element::f64, ov::element::f32},
                                               {ov::element::f32, ov::element::f16}};
    type_to_fuse_map empty_fuse_map = {};
    local_manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map,
                                                            empty_fuse_map,
                                                            /*keep_precision_sensitive_in_fp32*/ true,
                                                            /*convert_input_output_precision*/ false,
                                                            /*store_original_precision_as_rt_attribute*/ true);

    auto update_paged_attention_shape_func =
        [](const ov::element::Type&, const bool, const size_t, int64_t&, int64_t&) {};
    local_manager.register_pass<ov::pass::ConvertPagedAttnInputs>(cacheConfig, update_paged_attention_shape_func);
    local_manager.run_passes(local_model);

    for (size_t input = 0; input < 5; ++input) {
        EXPECT_EQ(paged_ssm->get_input_element_type(input), element::f32);
    }
    EXPECT_TRUE(ov::is_type<v0::Parameter>(paged_ssm->get_input_node_shared_ptr(5)));
    EXPECT_EQ(selective_ssm_state_table->get_element_type(), ov::element::f16);
    EXPECT_EQ(paged_ssm->get_output_element_type(0), element::f32);
}

// PagedSparseAttention caches get the same precision and padded shapes as PagedAttentionExtension, and the op
// stays valid with the padded shapes.
TEST(ConvertPagedAttnInputsSparseTest, PagedSparseAttentionQuantizedCaches) {
    constexpr size_t num_heads = 4;
    constexpr size_t num_kv_heads = 2;
    constexpr size_t key_head_size = 32;
    constexpr size_t value_head_size = 16;
    constexpr size_t key_block_size = 32;
    constexpr size_t value_block_size = 32;

    auto query = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, num_heads * key_head_size});
    auto key = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, num_kv_heads * key_head_size});
    auto value = std::make_shared<v0::Parameter>(element::f32, PartialShape{-1, num_kv_heads * value_head_size});
    auto key_cache = std::make_shared<v0::Parameter>(element::dynamic, PartialShape::dynamic(4));
    auto value_cache = std::make_shared<v0::Parameter>(element::dynamic, PartialShape::dynamic(4));
    auto past_lens = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto subsequence_begins = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto block_indices = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto block_indices_begins = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1});
    auto selected_block_indices = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1, 1, 8});
    auto selected_block_count = std::make_shared<v0::Parameter>(element::i32, PartialShape{-1, 1});
    auto scale = std::make_shared<v0::Constant>(element::f32, Shape{}, 0.5f);
    auto sinks = std::make_shared<v0::Constant>(element::f32, Shape{0});
    auto psa = std::make_shared<op::internal::PagedSparseAttention>(OutputVector{query,
                                                                                 key,
                                                                                 value,
                                                                                 key_cache,
                                                                                 value_cache,
                                                                                 past_lens,
                                                                                 subsequence_begins,
                                                                                 block_indices,
                                                                                 block_indices_begins,
                                                                                 selected_block_indices,
                                                                                 selected_block_count,
                                                                                 scale,
                                                                                 sinks},
                                                                    4);
    psa->get_rt_info()["num_k_heads"] = num_kv_heads;
    psa->get_rt_info()["k_head_size"] = key_head_size;
    psa->get_rt_info()["num_v_heads"] = num_kv_heads;
    psa->get_rt_info()["v_head_size"] = value_head_size;
    auto model = std::make_shared<ov::Model>(OutputVector{psa},
                                             ParameterVector{query,
                                                             key,
                                                             value,
                                                             key_cache,
                                                             value_cache,
                                                             past_lens,
                                                             subsequence_begins,
                                                             block_indices,
                                                             block_indices_begins,
                                                             selected_block_indices,
                                                             selected_block_count});

    ov::pass::ConvertPagedAttnInputs::KVCacheConfig cacheConfig;
    cacheConfig.inferencePrecision = element::f32;
    cacheConfig.keyCachePrecision = element::u8;
    cacheConfig.valueCachePrecision = element::u8;
    cacheConfig.keyCacheBlockSize = key_block_size;
    cacheConfig.valueCacheBlockSize = value_block_size;
    // u8 by-token: scale and zero point are appended to the head size.
    auto update_shape = [](const ov::element::Type& precision, const bool, const size_t, int64_t& head_size, int64_t&) {
        if (precision == ov::element::u8) {
            head_size += 2 * sizeof(float);
        }
    };

    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConvertPagedAttnInputs>(cacheConfig, update_shape);
    manager.run_passes(model);

    EXPECT_EQ(key_cache->get_element_type(), element::u8);
    EXPECT_EQ(value_cache->get_element_type(), element::u8);
    EXPECT_EQ(key_cache->get_partial_shape(),
              (PartialShape{-1, num_kv_heads, key_block_size, key_head_size + 2 * sizeof(float)}));
    EXPECT_EQ(value_cache->get_partial_shape(),
              (PartialShape{-1, num_kv_heads, value_block_size, value_head_size + 2 * sizeof(float)}));

    // The padded caches must still validate against the unpadded key/value widths.
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
    EXPECT_EQ(psa->get_output_partial_shape(0), (PartialShape{-1, num_heads * value_head_size}));
}

TEST(ConvertPagedAttnInputsSparseTest, SeparateSparseCachesKeepLogicalShapeAndInferencePrecision) {
    ParameterVector params;
    auto parameter = [&](const element::Type& type, const PartialShape& shape) {
        auto node = std::make_shared<v0::Parameter>(type, shape);
        params.push_back(node);
        return node;
    };
    auto query = parameter(element::f32, {-1, 128});
    auto key = parameter(element::f32, {-1, 64});
    auto value = parameter(element::f32, {-1, 32});
    auto key_cache = parameter(element::dynamic, {-1, 2, -1, 32});
    auto value_cache = parameter(element::dynamic, {-1, 2, -1, 16});
    auto past = parameter(element::i32, {-1});
    auto subsequences = parameter(element::i32, {-1});
    auto blocks = parameter(element::i32, {-1});
    auto block_begins = parameter(element::i32, {-1});
    auto selected = parameter(element::i32, {-1, 1, 8});
    auto count = parameter(element::i32, {-1, 1});
    auto scale = v0::Constant::create(element::f32, Shape{}, {0.5f});
    auto sinks = v0::Constant::create(element::f32, Shape{0}, {});
    auto sparse = std::make_shared<op::internal::PagedSparseAttention>(OutputVector{query,
                                                                                    key,
                                                                                    value,
                                                                                    key_cache,
                                                                                    value_cache,
                                                                                    past,
                                                                                    subsequences,
                                                                                    blocks,
                                                                                    block_begins,
                                                                                    selected,
                                                                                    count,
                                                                                    scale,
                                                                                    sinks},
                                                                       4);
    auto model = std::make_shared<Model>(OutputVector{sparse}, params);

    ov::pass::ConvertPagedAttnInputs::KVCacheConfig config;
    config.inferencePrecision = element::f32;
    config.keyCachePrecision = element::u8;
    config.valueCachePrecision = element::u8;
    config.keyCacheBlockSize = 32;
    config.valueCacheBlockSize = 16;
    config.separateSparseCacheFormat = true;
    bool shape_callback_called = false;
    auto update_shape = [&](const element::Type&, bool, size_t, int64_t&, int64_t&) {
        shape_callback_called = true;
    };
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConvertPagedAttnInputs>(config, update_shape);
    manager.run_passes(model);

    EXPECT_FALSE(shape_callback_called);
    EXPECT_EQ(key_cache->get_element_type(), element::f32);
    EXPECT_EQ(value_cache->get_element_type(), element::f32);
    EXPECT_EQ(key_cache->get_partial_shape(), (PartialShape{-1, 2, 32, 32}));
    EXPECT_EQ(value_cache->get_partial_shape(), (PartialShape{-1, 2, 16, 16}));
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());
}

TEST(ConvertPagedAttnInputsSparseTest, PagedQSAIndexerCachePageSizes) {
    ParameterVector params;
    auto parameter = [&](const element::Type& type, const PartialShape& shape) {
        auto node = std::make_shared<v0::Parameter>(type, shape);
        params.push_back(node);
        return node;
    };
    OutputVector args{parameter(element::f32, {-1, 8}),
                      parameter(element::f32, {-1, 4}),
                      parameter(element::f32, {-1, 2}),
                      parameter(element::f32, {-1, 2}),
                      parameter(element::f32, {4}),
                      parameter(element::f32, {4}),
                      parameter(element::i32, {-1}),
                      parameter(element::i32, {-1}),
                      parameter(element::i32, {-1}),
                      parameter(element::i32, {-1}),
                      parameter(element::f32, {-1, 1, -1, 4}),
                      parameter(element::f32, {-1, -1, 4}),
                      parameter(element::f32, {-1, 1, -1, 4})};
    op::internal::PagedQSAIndexer::Config indexer_config;
    indexer_config.key_block_size = 4;
    indexer_config.block_topk = 3;
    indexer_config.num_selection_heads = 1;
    indexer_config.indexer_rotary_dim = 2;
    indexer_config.indexer_head_dim = 4;
    auto indexer = std::make_shared<op::internal::PagedQSAIndexer>(args, indexer_config);
    auto model = std::make_shared<Model>(OutputVector{indexer->output(0), indexer->output(1)}, params);

    ov::pass::ConvertPagedAttnInputs::KVCacheConfig config;
    config.inferencePrecision = element::f32;
    config.keyCacheBlockSize = 32;
    auto update_shape = [](const element::Type&, bool, size_t, int64_t&, int64_t&) {};
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConvertPagedAttnInputs>(config, update_shape);
    manager.run_passes(model);

    EXPECT_EQ(params[10]->get_partial_shape(), (PartialShape{-1, 1, 32, 4}));
    EXPECT_EQ(params[11]->get_partial_shape(), (PartialShape{-1, 8, 4}));
    EXPECT_EQ(params[12]->get_partial_shape(), (PartialShape{-1, 1, 8, 4}));
    EXPECT_NO_THROW(model->validate_nodes_and_infer_types());

    config.keyCacheBlockSize = 10;
    ov::pass::Manager invalid_manager;
    invalid_manager.register_pass<ov::pass::ConvertPagedAttnInputs>(config, update_shape);
    EXPECT_THROW(invalid_manager.run_passes(model), ov::Exception);
}

struct PagedQSAFusionGraph {
    std::shared_ptr<Model> model;
    ov::pass::paged_attention::PaParams pa_params;
    std::unordered_set<std::string> removed_variables;
};

PagedQSAFusionGraph make_paged_qsa_fusion_graph(bool per_layer, bool expose_key_cache) {
    ParameterVector params;
    auto parameter = [&](const element::Type& type, const PartialShape& shape) {
        auto node = std::make_shared<v0::Parameter>(type, shape);
        params.push_back(node);
        return node;
    };
    auto variable = [](const std::string& name, const PartialShape& shape) {
        return std::make_shared<op::util::Variable>(op::util::VariableInfo{shape, element::f32, name});
    };
    auto key_var = variable("key", {1, 2, -1, 16});
    auto value_var = variable("value", {1, 2, -1, 16});
    auto block_var = variable("block", {1, -1, 16});
    auto raw_var = variable("raw", {1, 3, 16});
    auto rope_var = variable("rope", {1, 16});
    auto key_read = std::make_shared<op::v6::ReadValue>(key_var);
    auto value_read = std::make_shared<op::v6::ReadValue>(value_var);
    auto block_read = std::make_shared<op::v6::ReadValue>(block_var);
    auto raw_read = std::make_shared<op::v6::ReadValue>(raw_var);
    auto rope_read = std::make_shared<op::v6::ReadValue>(rope_var);

    auto raw_query = parameter(element::f32, {1, -1, 2, 16});
    auto raw_keys = parameter(element::f32, {1, -1, 16});
    auto key_length = parameter(element::i32, {1});
    auto cos = parameter(element::f32, {1, -1, 8});
    auto sin = parameter(element::f32, {1, -1, 8});
    auto gain = v0::Constant::create(element::f32, Shape{16}, std::vector<float>(16, 1.0f));
    auto norm = std::make_shared<op::internal::RMS>(raw_query, gain, 1e-5f);
    auto axis = v0::Constant::create(element::i64, Shape{1}, {2});
    auto cos_expanded = std::make_shared<op::v0::Unsqueeze>(cos, axis);
    auto sin_expanded = std::make_shared<op::v0::Unsqueeze>(sin, axis);
    op::internal::RoPE::Config rope_config;
    rope_config.rotary_ndims = 8;
    rope_config.cos_sin_ndims = 8;
    rope_config.head_cnt = 2;
    rope_config.head_size = 16;
    auto rope = std::make_shared<op::internal::RoPE>(OutputVector{norm, cos_expanded, sin_expanded}, rope_config);
    auto order = v0::Constant::create(element::i64, Shape{4}, {0, 2, 1, 3});
    auto index_query = std::make_shared<op::v1::Transpose>(rope, order);
    auto prepare = std::make_shared<
        op::v17::QSAPrepare>(raw_keys, block_read, key_length, cos, sin, gain, raw_read, rope_read, 4, 1e-5f);
    auto indexer =
        std::make_shared<op::v17::SparseAttentionIndexer>(OutputVector{index_query, prepare->output(0), key_length},
                                                          4,
                                                          8,
                                                          1);
    auto query = parameter(element::f32, {1, 4, -1, 16});
    auto current_key = parameter(element::f32, {1, 2, -1, 16});
    auto current_value = parameter(element::f32, {1, 2, -1, 16});
    auto input_ids = parameter(element::i32, {1, -1});
    input_ids->set_friendly_name("input_ids");
    input_ids->output(0).set_names({"input_ids"});
    auto key = std::make_shared<op::v0::Concat>(OutputVector{key_read, current_key}, 2);
    auto value = std::make_shared<op::v0::Concat>(OutputVector{value_read, current_value}, 2);
    auto attention = std::make_shared<op::v17::SparseAttention>(
        OutputVector{query, key, value, indexer->output(0), indexer->output(1), key_length},
        4,
        1,
        true);
    SinkVector sinks{std::make_shared<op::v6::Assign>(key, key_var),
                     std::make_shared<op::v6::Assign>(value, value_var),
                     std::make_shared<op::v6::Assign>(prepare->output(0), block_var),
                     std::make_shared<op::v6::Assign>(prepare->output(1), raw_var),
                     std::make_shared<op::v6::Assign>(prepare->output(2), rope_var)};
    OutputVector outputs{attention};
    if (expose_key_cache) {
        outputs.push_back(key);
    }

    PagedQSAFusionGraph graph;
    graph.model = std::make_shared<Model>(outputs, sinks, params);
    for (const auto* name : {"past_lens", "subsequence_begins", "block_indices_begins"}) {
        graph.pa_params.add(name, element::i32, {-1});
    }
    if (!per_layer) {
        graph.pa_params.add("block_indices", element::i32, {-1});
    }
    return graph;
}

TEST(PagedQSAFusionTest, ConvertsStatefulLayerWithSharedOrPerLayerBlockIndices) {
    for (bool per_layer : {false, true}) {
        auto graph = make_paged_qsa_fusion_graph(per_layer, false);
        std::shared_ptr<op::v17::SparseAttentionIndexer> original_indexer;
        for (const auto& node : graph.model->get_ordered_ops()) {
            if (auto match = ov::as_type_ptr<op::v17::SparseAttentionIndexer>(node)) {
                original_indexer = match;
            }
        }
        ASSERT_NE(original_indexer, nullptr);
        auto index_query = std::make_shared<ov::pass::pattern::op::QSAIndexQueryBlock>();
        auto indexer_pattern = ov::pass::pattern::wrap_type<op::v17::SparseAttentionIndexer>(
            {index_query, ov::pass::pattern::wrap_type<op::v17::QSAPrepare>(), ov::pass::pattern::any_input()});
        ov::pass::pattern::Matcher matcher(indexer_pattern, "QSAIndexerTest");
        ASSERT_TRUE(matcher.match(original_indexer->output(0)));
        ASSERT_TRUE(index_query->get_match(matcher.get_pattern_value_map(), original_indexer));
        ov::pass::Manager manager;
        manager.set_per_pass_validation(false);
        manager.register_pass<ov::pass::PagedQSAFusion>(graph.pa_params, graph.removed_variables, per_layer);
        manager.run_passes(graph.model);
        std::shared_ptr<op::internal::PagedSparseAttention> sparse;
        std::shared_ptr<op::internal::PagedQSAIndexer> indexer;
        for (const auto& node : graph.model->get_ordered_ops()) {
            if (auto match = ov::as_type_ptr<op::internal::PagedSparseAttention>(node)) {
                sparse = match;
            } else if (auto match = ov::as_type_ptr<op::internal::PagedQSAIndexer>(node)) {
                indexer = match;
            }
        }
        ASSERT_NE(sparse, nullptr);
        ASSERT_NE(indexer, nullptr);
        EXPECT_EQ(graph.removed_variables.size(), 5);
        EXPECT_TRUE(graph.pa_params.exists(per_layer ? "block_indices.0" : "block_indices"));
        EXPECT_EQ(sparse->input_value(7), indexer->input_value(op::internal::PagedQSAIndexer::InputIdx::BLOCK_INDICES));
        EXPECT_EQ(sparse->get_rt_info().at("num_k_heads").as<int64_t>(), 2);
        EXPECT_EQ(sparse->get_rt_info().at("k_head_size").as<int64_t>(), 16);
    }
}

TEST(PagedQSAFusionTest, DoesNotRemoveStateWithOtherConsumers) {
    auto graph = make_paged_qsa_fusion_graph(false, true);
    ov::pass::Manager manager;
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::pass::PagedQSAFusion>(graph.pa_params, graph.removed_variables, false);
    manager.run_passes(graph.model);
    EXPECT_TRUE(graph.removed_variables.empty());
    EXPECT_FALSE(graph.pa_params.exists("key_cache.0"));
    EXPECT_TRUE(ov::op::util::has_op_with_type<op::v17::SparseAttention>(graph.model));
}

TEST(SDPAToPagedAttentionSparseTest, ConvertsStatefulSparseModel) {
    for (bool per_layer : {false, true}) {
        auto graph = make_paged_qsa_fusion_graph(per_layer, false);
        ov::pass::SDPAToPagedAttention pass(per_layer);
        EXPECT_TRUE(pass.run_on_model(graph.model));
        EXPECT_TRUE(graph.model->get_sinks().empty());
        EXPECT_TRUE(ov::op::util::has_op_with_type<op::internal::PagedQSAIndexer>(graph.model));
        EXPECT_TRUE(ov::op::util::has_op_with_type<op::internal::PagedSparseAttention>(graph.model));
        EXPECT_FALSE(ov::op::util::has_op_with_type<op::v17::SparseAttention>(graph.model));
        const auto inputs = graph.model->inputs();
        EXPECT_NE(std::find_if(inputs.begin(),
                               inputs.end(),
                               [&](const auto& input) {
                                   return input.get_names().count(per_layer ? "block_indices.0" : "block_indices") != 0;
                               }),
                  inputs.end());
    }
}

TEST(SDPAToPagedAttentionSparseTest, RejectsSparseStateWithOtherConsumers) {
    auto graph = make_paged_qsa_fusion_graph(false, true);
    ov::pass::SDPAToPagedAttention pass;
    EXPECT_THROW(pass.run_on_model(graph.model), ov::Exception);
}

}  // namespace
