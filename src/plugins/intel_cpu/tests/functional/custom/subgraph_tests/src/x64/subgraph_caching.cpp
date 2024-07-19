// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Motivation:
// In a dynamic scenario, depending on the input shapes for the current node,
//   -  we can either generate a new jit kernel or get an existing one from the cache
//   -  we can either make shape inference or get existing output shapes from the cache
// But the current single layer tests do not allow checking the case when the same kernel can be used for different nodes.
// We check 2 Subgraphs with MatMuls inside to validate Kernel Executor table also

//  -----------              -----------    -----------              -----------
//  |input 0.0|              |input 0.1|    |input 1.0|              |input 1.1|
//  -----------              -----------    -----------              -----------
//       |                        |              |                        |
//  ------------------------------------    ------------------------------------
//  |            MatMul 0              |    |            Matmul 1              |
//  ------------------------------------    ------------------------------------
//                   |                                       |
//  ------------------------------------    ------------------------------------
//  |              Add 0               |    |              Add 1               |
//  ------------------------------------    ------------------------------------
//                   |                                       |
//  ----------------------------------------------------------------------------
//  |                                 concat                                   |
//  ----------------------------------------------------------------------------
//                                       |
//                                   --------
//                                   |output|
//                                   --------

#include "snippets/op/subgraph.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "internal_properties.hpp"


using namespace CPUTestUtils;

namespace ov {
namespace test {
using namespace ov::test::utils;

typedef std::tuple<
        std::vector<InputShape>, // Input Shapes
        std::vector<ElementType> // Input precisions
> SubgraphCacheTestParams;

class SubgraphCacheTest : public testing::WithParamInterface<SubgraphCacheTestParams>,
                          virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SubgraphCacheTestParams> &obj) {
        std::vector<InputShape> inputShapes;
        std::vector<ElementType> inputPrecisions;
        std::tie(inputShapes, inputPrecisions) = obj.param;

        std::ostringstream results;

         for (size_t i = 0; i < inputShapes.size(); i++) {
            results << "IS[" << i << "]=" << inputShapes[i];
        }

        for (size_t i = 0; i < inputPrecisions.size(); i++) {
            results << "InPRC" << std::to_string(i) << "=" << inputPrecisions[i] << "_";
        }

        return results.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        std::vector<InputShape> inputShapes;
        std::vector<ElementType> inputPrecisions;
        std::tie(inputShapes, inputPrecisions) = this->GetParam();

        init_input_shapes(inputShapes);

        // Enable Snippets
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK));

        ov::ParameterVector paramVec;
        for (size_t i = 0; i < inputDynamicShapes.size(); i++) {
            paramVec.push_back(std::make_shared<ov::op::v0::Parameter>(inputPrecisions[i], inputDynamicShapes[i]));
        }

        auto matmul0 = std::make_shared<ov::op::v0::MatMul>(paramVec[0], paramVec[1]);
        auto matmul1 = std::make_shared<ov::op::v0::MatMul>(paramVec[2], paramVec[3]);

        auto const0 = utils::make_constant(matmul0->get_output_element_type(0), ov::Shape{1});
        auto const1 = utils::make_constant(matmul1->get_output_element_type(0), ov::Shape{1});

        auto add0 = std::make_shared<ov::op::v1::Add>(matmul0, const0);
        auto add1 = std::make_shared<ov::op::v1::Add>(matmul1, const1);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{add0, add1}, -1);
        function = std::make_shared<ov::Model>(concat, paramVec, "Subgraph");
    }

    void validate_result() const {
        const std::set<std::string> types = { "Input", "Output", "Subgraph", "Concatenation" };
        const auto function = compiledModel.get_runtime_model();
        for (const auto& op : function->get_ordered_ops()) {
            const auto& rtInfo = op->get_rt_info();
            auto it = rtInfo.find(ov::exec_model_info::LAYER_TYPE);
            OPENVINO_ASSERT(rtInfo.end() != it);
            const auto node_type = it->second.as<std::string>();
            ASSERT_TRUE(types.count(node_type)) << "The execution node is unexpected : " << node_type;
        }
    }
};

TEST_P(SubgraphCacheTest, CompareWithRefs) {
    run();
    validate_result();
}

namespace {

std::vector<ElementType> inputPrecisions {
    ElementType::f32, ElementType::f32, ElementType::f32, ElementType::f32
};

std::vector<InputShape> inputShapes {
    {{1, 2, -1, -1}, {{1, 2, 10, 3}, {1, 2, 10, 3}, {1, 2, 10, 8}, {1, 2, 10, 3}}},
    {{1, 2, -1, -1}, {{1, 2, 3, 12}, {1, 2, 3, 12}, {1, 2, 8,  9}, {1, 2, 3, 12}}},
    {{1, 2, -1, -1}, {{1, 2, 10, 8}, {1, 2, 10, 3}, {1, 2, 10, 3}, {1, 2, 10, 8}}},
    {{1, 2, -1, -1}, {{1, 2, 8,  9}, {1, 2, 3, 12}, {1, 2, 3, 12}, {1, 2, 8,  9}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_SubgraphCache, SubgraphCacheTest,
                        ::testing::Combine(
                                ::testing::Values(inputShapes),
                                ::testing::Values(inputPrecisions)),
                        SubgraphCacheTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov