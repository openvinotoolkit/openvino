// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_pagedattn_inputs.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/paged_attention.hpp"
#include "openvino/runtime/properties.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace ov::test;
using namespace ov;
using namespace ov::op;
using namespace ov::gen_pattern;
using ConvertPagedAttnInputsParams = std::tuple<std::vector<ov::element::Type>,  // cache_precision
                                                std::vector<size_t>,             // cache_group_size
                                                std::vector<size_t>,             // block_size
                                                ov::element::Type,               // infer_precsion
                                                bool,                            // quant_key_by_channel
                                                bool,                            // accuracy_mode
                                                bool                             // is_ir_kv_cache_f16
                                                >;
namespace {
class ConvertPagedAttnInputsTest : public TransformationTestsF,
                                   public testing::WithParamInterface<ConvertPagedAttnInputsParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertPagedAttnInputsParams>& obj) {
        std::vector<ov::element::Type> cachePrecision;
        std::vector<size_t> cacheGroupSize;
        std::vector<size_t> blcokSize;
        bool quantKeybychannel;
        bool isAccuracyMode;
        bool isIRKVCacheF16;
        ov::element::Type inferPrec;
        std::tie(cachePrecision,
                 cacheGroupSize,
                 blcokSize,
                 inferPrec,
                 quantKeybychannel,
                 isAccuracyMode,
                 isIRKVCacheF16) = obj.param;
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
    std::vector<ov::element::Type> cachePrecision;
    std::vector<size_t> cacheGroupSize;
    std::tie(cachePrecision, cacheGroupSize, blockSize, inferPrec, quantKeybychannel, isAccuracyMode, isIRKVCacheF16) =
        this->GetParam();
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

        auto pa = std::make_shared<op::PagedAttentionExtension>(OutputVector{Q,
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
                                                                             max_context_len});
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
                                                                max_context_len});
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

        auto pa = std::make_shared<op::PagedAttentionExtension>(OutputVector{Q,
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
                                                                             max_context_len});
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
                                                                    max_context_len});
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

}  // namespace