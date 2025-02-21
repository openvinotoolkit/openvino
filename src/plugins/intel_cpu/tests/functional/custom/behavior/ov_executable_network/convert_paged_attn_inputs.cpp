// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/paged_attention.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace ov::test;
using namespace ov;
using namespace ov::op;
using namespace ov::gen_pattern;
using ConvertPagedAttnInputsParams = std::tuple<std::vector<ov::element::Type>,  // cache_precision
                                                std::vector<size_t>,             // cache_group_size
                                                bool,                            // quant_key_by_channel
                                                bool,                            // accuracy_mode
                                                bool                             // is_ir_kv_cache_f16
                                                >;
namespace {
class ConvertPagedAttnInputsTest : public testing::WithParamInterface<ConvertPagedAttnInputsParams>,
                                   public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertPagedAttnInputsParams>& obj) {
        std::vector<ov::element::Type> cachePrecision;
        std::vector<size_t> cacheGroupSize;
        bool quantKeybychannel;
        bool isAccuracyMode;
        bool isIRKVCacheF16;
        std::tie(cachePrecision, cacheGroupSize, quantKeybychannel, isAccuracyMode, isIRKVCacheF16) = obj.param;
        std::ostringstream result;
        result << "KeyPrc=" << cachePrecision[0] << "_";
        result << "ValuePrc=" << cachePrecision[1] << "_";
        result << "KeyGS=" << cacheGroupSize[0] << "_";
        result << "ValueGS=" << cacheGroupSize[1] << "_";
        result << "KeyByChannel=" << quantKeybychannel << "_";
        result << "isAccuracyMode=" << isAccuracyMode << "_";
        result << "isIRKVCacheF16=" << isIRKVCacheF16;
        return result.str();
    }

public:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        std::vector<ov::element::Type> cachePrecision;
        std::vector<size_t> cacheGroupSize;
        targetDevice = utils::DEVICE_CPU;
        std::tie(cachePrecision, cacheGroupSize, quantKeybychannel, isAccuracyMode, isIRKVCacheF16) = this->GetParam();
        keyCachePrecision = cachePrecision[0];
        valueCachePrecision = cachePrecision[1];
        keyCacheGroupSize = cacheGroupSize[0];
        valueCacheGroupSize = cacheGroupSize[1];
        numKeyHeads = 2;
        keyHeadSize = 32;

        numValueHeads = 2;
        valueHeadSize = 64;
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
        key_cache_0->get_output_tensor(0).set_names({"key_cache.0"});
        auto value_cache_0 = std::make_shared<v0::Parameter>(ov::element::dynamic, PartialShape::dynamic(4));
        value_cache_0->get_output_tensor(0).set_names({"value_cache.0"});
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

        function = std::make_shared<ov::Model>(ov::OutputVector{pa},
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
            function->set_rt_info("f16", "runtime_options", ov::hint::kv_cache_precision.name());
        }
    }
    size_t keyHeadSize;
    size_t valueHeadSize;
    size_t numKeyHeads;
    size_t numValueHeads;
    ov::element::Type keyCachePrecision;
    ov::element::Type valueCachePrecision;
    size_t keyCacheGroupSize;
    size_t valueCacheGroupSize;
    bool quantKeybychannel;
    bool isAccuracyMode;
    bool isIRKVCacheF16;
};

TEST_P(ConvertPagedAttnInputsTest, checkPrecisionAndShape) {
    if (!(with_cpu_sve() || with_cpu_x86_avx2())) {
        GTEST_SKIP();
    }
    auto inferPrec = with_cpu_x86_bfloat16() ? ov::element::bf16 : ov::element::f32;
    if (isAccuracyMode) {
        configuration[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
        inferPrec = ov::element::f32;
    } else {
        configuration[ov::key_cache_precision.name()] = keyCachePrecision;
        configuration[ov::value_cache_precision.name()] = valueCachePrecision;
        configuration[ov::key_cache_group_size.name()] = keyCacheGroupSize;
        configuration[ov::value_cache_group_size.name()] = valueCacheGroupSize;
    }
    configuration[ov::hint::inference_precision.name()] = inferPrec;
    compile_model();
    for (const auto& input : compiledModel.inputs()) {
        for (auto& name : input.get_names()) {
            auto cachePrec = input.get_element_type();
            ov::PartialShape pshape = input.get_partial_shape();
            auto checkCachePrec = [&](ov::element::Type tensorPrec, ov::element::Type desizredPrec) {
                desizredPrec = desizredPrec == ov::element::f16 && inferPrec == ov::element::bf16 ? ov::element::bf16
                                                                                                  : desizredPrec;
                if (isAccuracyMode)
                    desizredPrec = ov::element::f32;
                return tensorPrec == desizredPrec;
            };
            auto checkCacheShape =
                [&](ov::element::Type cachePrec, size_t headSize, size_t groupSize, bool quantBychannel) {
                    const size_t blockSize = 32;
                    const size_t paramSize = 2 * sizeof(float) * 8 / cachePrec.bitwidth();
                    if (quantBychannel) {
                        ASSERT_EQ(pshape[2].get_length(), paramSize + blockSize);
                        ASSERT_EQ(pshape[3].get_length(), headSize);
                    } else {
                        size_t groupNum = headSize / groupSize;
                        ASSERT_EQ(pshape[2].get_length(), blockSize);
                        ASSERT_EQ(pshape[3].get_length(), headSize + paramSize * groupNum);
                    }
                    ASSERT_EQ(pshape[0], ov::Dimension::dynamic());
                };

            if (name.find("key_cache.") == 0) {
                ASSERT_TRUE(checkCachePrec(cachePrec, keyCachePrecision));
                if (keyCachePrecision.is_integral() && !isAccuracyMode) {
                    checkCacheShape(keyCachePrecision, keyHeadSize, keyCacheGroupSize, quantKeybychannel);
                }
                break;
            } else if (name.find("value_cache.") == 0) {
                ASSERT_TRUE(checkCachePrec(cachePrec, valueCachePrecision));
                if (valueCachePrecision.is_integral() && !isAccuracyMode) {
                    checkCacheShape(valueCachePrecision, valueHeadSize, valueCacheGroupSize, false);
                }
                break;
            }
        }
    }
}

std::vector<std::vector<ov::element::Type>> get_cache_prec() {
    if (with_cpu_x86_bfloat16()) {
        return {
            {ElementType::f16, ElementType::f16},
            {ElementType::u8, ElementType::u8},
            {ElementType::u8, ElementType::u4},
        };
    } else {
        return {
            {ElementType::f32, ElementType::f32},
            {ElementType::f16, ElementType::f16},
            {ElementType::u8, ElementType::u8},
            {ElementType::u8, ElementType::u4},
        };
    }
}

const std::vector<std::vector<size_t>> cache_gs = {{32, 32}};

INSTANTIATE_TEST_SUITE_P(smoke_ConvertPagedAttnInputsTest,
                         ConvertPagedAttnInputsTest,
                         ::testing::Combine(::testing::ValuesIn(get_cache_prec()),
                                            ::testing::ValuesIn(cache_gs),
                                            ::testing::Values(false),
                                            ::testing::Values(true, false),
                                            ::testing::Values(true, false)),
                         ConvertPagedAttnInputsTest::getTestCaseName);

}  // namespace