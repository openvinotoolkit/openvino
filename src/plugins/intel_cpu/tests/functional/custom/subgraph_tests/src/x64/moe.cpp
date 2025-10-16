// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <vector>

#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/subgraph/moe_builders.hpp"
#include "shared_test_classes/subgraph/weights_decompression_builders.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov::test {

enum class MoEType { MoE2GeMM, MoE3GeMM };

inline std::ostream& operator<<(std::ostream& os, const MoEType& type) {
    switch (type) {
    case MoEType::MoE2GeMM:
        return os << "MoE2GeMM";
    case MoEType::MoE3GeMM:
        return os << "MoE3GeMM";
    default:
        OPENVINO_THROW("Unsupported MoEType");
    }
}

typedef std::tuple<MoePatternParams,
                   MoEType,                // MoE builder type
                   ov::test::ElementType,  // weights precision
                   ov::test::ElementType,  // decompression precision
                   ov::test::ElementType,  // scale precision
                   bool,                   // use weight decompression
                   DecompressionType,      // decompression multiply type
                   DecompressionType,      // decompression subtract type
                   bool,                   // reshape on decompression constants
                   int,                    // decompression_group_size
                   ov::AnyMap>             // additional config
    MoeTestParams;

class MoeSubgraphTest : public testing::WithParamInterface<MoeTestParams>,
                        virtual public SubgraphBaseTest,
                        public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoeTestParams>& obj) {
        const auto& [moe_params,
                     moe_type,
                     weights_precision,
                     decompression_precision,
                     scale_precision,
                     use_weight_decompression,
                     decompression_multiply_type,
                     decompression_subtract_type,
                     reshape_on_decompression,
                     decompression_group_size,
                     additional_config] = obj.param;

        std::ostringstream result;
        result << "IS=" << ov::test::utils::partialShape2str({moe_params.data_shape.first}) << "_";
        result << "TS=";
        for (const auto& static_shape : moe_params.data_shape.second) {
            result << ov::test::utils::vec2str(static_shape) << ",";
        }
        result << "top_k_experts=" << moe_params.topk << "_";
        result << "total_experts=" << moe_params.number_of_experts << "_";
        result << "intermediate_size=" << moe_params.intermediate_size << "_";
        result << "moe_type=" << moe_type << "_";

        if (use_weight_decompression) {
            result << "WP=" << weights_precision << "_";
            result << "DP=" << decompression_precision << "_";
            result << "SP=" << scale_precision << "_";
            result << "DM=" << decompression_multiply_type << "_";
            result << "DS=" << decompression_subtract_type << "_";
            result << "RD=" << reshape_on_decompression << "_";
            result << "GS=" << decompression_group_size << "_";
        }

        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << "=" << configEntry.second.as<std::string>() << "_";
        }
        result << ")";

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        const auto& [shape_params,
                     moe_type,
                     weights_precision,
                     decompression_precision,
                     scale_precision,
                     use_weight_decompression,
                     decompression_multiply_type,
                     decompression_subtract_type,
                     reshape_on_decompression,
                     decompression_group_size,
                     additional_config] = GetParam();

        configuration.insert(additional_config.begin(), additional_config.end());
        init_input_shapes({shape_params.data_shape});
        inType = outType = ov::element::f32;

        if (moe_type == MoEType::MoE2GeMM) {
            function = initMoE2GeMMSubgraph(shape_params,
                                            ov::element::f32,
                                            weights_precision,
                                            decompression_precision,
                                            ov::element::f32,
                                            use_weight_decompression,
                                            decompression_multiply_type,
                                            decompression_subtract_type,
                                            reshape_on_decompression,
                                            decompression_group_size);
        } else if (moe_type == MoEType::MoE3GeMM) {
            function = initMoE3GeMMSubgraph(shape_params,
                                            ov::element::f32,
                                            weights_precision,
                                            decompression_precision,
                                            ov::element::f32,
                                            use_weight_decompression,
                                            decompression_multiply_type,
                                            decompression_subtract_type,
                                            reshape_on_decompression,
                                            decompression_group_size);
        } else {
            OPENVINO_THROW("Unsupported MoEType");
        }
    }
};

namespace {
// Test parameter generation
const std::vector<ov::test::ElementType> decompression_precisions = {ov::element::f32};
const std::vector<ov::test::ElementType> weights_precisions = {ov::element::u8, ov::element::u4, ov::element::i4};
const std::vector<MoEType> moe_types = {MoEType::MoE2GeMM, MoEType::MoE3GeMM};

const std::vector<MoePatternParams> moe_params = {
    {
        {{-1, -1, 2048}, {{2, 15, 2048}, {2, 1, 2048}, {3, 8, 2048}}},  // data_shape,
                                                                        // seq_len=dynamic, hidden_size=2048
        4,                                                              // topk
        32,                                                             // number_of_experts
        4096                                                            // intermediate_size
    },
    {
        {{-1, -1, 1024}, {{1, 32, 1024}, {1, 1, 1024}, {1, 16, 1024}}},  // Different seq length
        6,                                                               // topk
        64,                                                              // number_of_experts
        2048                                                             // intermediate_size
    }};

const ov::AnyMap additional_config_basic = {{ov::hint::inference_precision.name(), ov::element::f32}};
const ov::AnyMap additional_config_bf16 = {{ov::hint::inference_precision.name(), ov::element::bf16}};

}  // namespace

// Basic FP32 tests
INSTANTIATE_TEST_SUITE_P(smoke_MoeSubgraph_basic,
                         MoeSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(false),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(additional_config_basic)),
                         MoeSubgraphTest::getTestCaseName);

// BF16 inference precision tests
INSTANTIATE_TEST_SUITE_P(smoke_MoeSubgraph_bf16,
                         MoeSubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(false),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(DecompressionType::full),
                                            ::testing::Values(false),
                                            ::testing::Values(0),
                                            ::testing::Values(additional_config_bf16)),
                         MoeSubgraphTest::getTestCaseName);

TEST_P(MoeSubgraphTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

}  // namespace ov::test
