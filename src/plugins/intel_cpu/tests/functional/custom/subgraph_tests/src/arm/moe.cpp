// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <vector>

#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/subgraph/moe_builders.hpp"
#include "shared_test_classes/subgraph/weights_decompression_params.hpp"
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

using MoeTestParams = std::tuple<MoePatternParams,
                                 MoEType,      // MoE builder type
                                 ov::AnyMap>;  // additional config

class MoESubgraphTest : public testing::WithParamInterface<MoeTestParams>,
                        virtual public SubgraphBaseTest,
                        public CpuTestWithFusing {
public:
    static std::string generateBaseMoeTestName(const MoePatternParams& moe_params,
                                               const MoEType& moe_type,
                                               const ov::AnyMap& additional_config) {
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

        result << "config=(";
        for (const auto& configEntry : additional_config) {
            result << configEntry.first << "=" << configEntry.second.as<std::string>() << "_";
        }
        result << ")";

        return result.str();
    }
    static std::string getTestCaseName(const testing::TestParamInfo<MoeTestParams>& obj) {
        const auto& [moe_params, moe_type, additional_config] = obj.param;
        return generateBaseMoeTestName(moe_params, moe_type, additional_config);
    }

    static size_t get_expected_gather_mm_count(MoEType moe_type) {
        switch (moe_type) {
        case MoEType::MoE2GeMM:
            return 2;
        case MoEType::MoE3GeMM:
            return 3;
        default:
            OPENVINO_THROW("Unsupported MoEType");
        }
    }

    static std::set<std::shared_ptr<ov::Node>> get_gather_mm_nodes(const std::shared_ptr<const ov::Model>& model,
                                                                   MoEType moe_type) {
        const std::string expected_gather_mm_type = "GatherMatmul";
        std::set<std::shared_ptr<ov::Node>> gather_mm_nodes;
        for (const auto& node : model->get_ordered_ops()) {
            if (node->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>() == expected_gather_mm_type) {
                gather_mm_nodes.insert(node);
            }
        }
        return gather_mm_nodes;
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto& [shape_params, moe_type, additional_config] = GetParam();

        configuration.insert(additional_config.begin(), additional_config.end());
        init_input_shapes({shape_params.data_shape});
        inType = outType = ov::element::f32;

        auto itr = configuration.find(ov::hint::inference_precision.name());
        if (itr != configuration.end() && itr->second == ov::element::bf16) {
            rel_threshold = 0.1f;
            abs_threshold = 0.1f;
            inType = outType = ov::element::bf16;
        }

        if (moe_type == MoEType::MoE2GeMM) {
            function = initMoE2GeMMSubgraph(shape_params, ov::element::f32, ov::element::f32);
        } else if (moe_type == MoEType::MoE3GeMM) {
            function = initMoE3GeMMSubgraph(shape_params, ov::element::f32, ov::element::f32);
        } else {
            OPENVINO_THROW("Unsupported MoEType");
        }
    }

    void check_results() {
        const auto& moe_type = std::get<1>(GetParam());
        const auto& gather_mm_nodes = get_gather_mm_nodes(compiledModel.get_runtime_model(), moe_type);
        const size_t expected_gather_mm_count = get_expected_gather_mm_count(moe_type);
        EXPECT_EQ(gather_mm_nodes.size(), expected_gather_mm_count);
    }
};


TEST_P(MoESubgraphTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    check_results();
}

namespace {
const std::vector<MoEType> moe_types = {MoEType::MoE2GeMM, MoEType::MoE3GeMM};

const std::vector<MoePatternParams> moe_params_smoke = {
    {
        {{-1, -1, 256}, {{2, 15, 256}, {2, 1, 256}, {3, 8, 256}}},  // data_shape,
                                                                    // seq_len=dynamic, hidden_size=256
        4,                                                          // topk
        8,                                                          // number_of_experts
        512                                                         // intermediate_size
    },
    {
        {{-1, -1, 128}, {{1, 32, 128}, {1, 1, 128}, {1, 16, 128}}},  // Different seq length
        2,                                                           // topk
        4,                                                           // number_of_experts
        256                                                          // intermediate_size
    },
};

std::vector<ov::AnyMap> generate_additional_config() {
    std::vector<ov::AnyMap> additional_config = {{{ov::hint::inference_precision.name(), ov::element::f32}}};
    return additional_config;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_MoESubgraph_basic,
                         MoESubgraphTest,
                         ::testing::Combine(::testing::ValuesIn(moe_params_smoke),
                                            ::testing::ValuesIn(moe_types),
                                            ::testing::ValuesIn(generate_additional_config())),
                         MoESubgraphTest::getTestCaseName);

}  // namespace ov::test
