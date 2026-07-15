// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils/cpu_test_utils.hpp"
#include "common_test_utils/node_builders/moe_builders.hpp"
#include "shared_test_classes/subgraph/weights_decompression_params.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov::test {
enum class MoEType { MoE2GeMM, MoE3GeMM };

struct MoeTestShapeParams {
    InputShape data_shape;
    size_t topk;
    size_t number_of_experts;
    size_t intermediate_size;
};
using MoeTestParams = std::tuple<MoeTestShapeParams,
                                 MoEType,            // MoE builder type
                                 MoEActivationType,  // gate activation type (3GeMM only)
                                 ov::AnyMap>;        // additional config

using MoeCompressedWeightsTestParams = std::tuple<MoeTestShapeParams,
                                                  MoEType,                // MoE builder type
                                                  MoEActivationType,      // gate activation type (3GeMM only)
                                                  ov::test::ElementType,  // weights precision
                                                  ov::test::ElementType,  // decompression precision
                                                  ov::test::ElementType,  // scale precision
                                                  ov::test::utils::DecompressionType,  // decompression multiply type
                                                  ov::test::utils::DecompressionType,  // decompression subtract type
                                                  bool,        // reshape on decompression constants
                                                  int,         // decompression_group_size
                                                  ov::AnyMap,  // additional config
                                                  bool>;       // use_matmul_decompression_impl
class MoESubgraphTest : public testing::WithParamInterface<MoeTestParams>,
                        virtual public SubgraphBaseTest,
                        public CpuTestWithFusing {
public:
    static std::string generateBaseMoeTestName(const MoeTestShapeParams& moe_params,
                                               const MoEType& moe_type,
                                               const ov::AnyMap& additional_config);
    static std::string getTestCaseName(const testing::TestParamInfo<MoeTestParams>& obj);
    static size_t get_expected_gather_mm_count(MoEType moe_type);
    static std::set<std::shared_ptr<ov::Node>> get_gather_mm_nodes(const std::shared_ptr<const ov::Model>& model,
                                                                   MoEType moe_type);
protected:
    void SetUp() override;
    void check_results(); 

};
class MoECompressedWeightsSubgraphTest : public testing::WithParamInterface<MoeCompressedWeightsTestParams>,
                                         virtual public SubgraphBaseTest,
                                         public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoeCompressedWeightsTestParams>& obj);
protected:
    void SetUp() override;  
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    void check_results();
};
} // namespace ov::test