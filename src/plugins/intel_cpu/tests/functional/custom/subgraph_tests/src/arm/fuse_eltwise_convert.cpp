// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/test_enums.hpp"
#include "internal_properties.hpp"
#include "openvino/op/convert.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/general_utils.h"

using namespace CPUTestUtils;

namespace ov {
namespace test {

// Subgraph:
/*
┌──────────────────┐           ┌──────────────────┐
│      INPUT       │           │      INPUT       │
└─────┬───┬────────┘           └─────────┬────────┘
      |   │      ┌──────────────────┐    │
      |   └──────┤     ELTWISE      ├────┘
      |          └──┬───────────────┘
      |             │
    ┌─┴─────────────┴──┐
    │     ELTWISE      │
    └────────┬─────────┘
             |
             |
    ┌────────┴─────────┐
    │      CONVERT     │
    └──────────────────┘

Verify Eltwise Floor does not fuse Convert node if Convert output is not f16/f32
 */

using eltwiseFusingCPUTestParamsSet = std::tuple<utils::EltwiseTypes,
                                                 ElementType          //convert out type
                                       >;

class MergeEltwiseAndConvertTransformationCPUTest: public testing::WithParamInterface<eltwiseFusingCPUTestParamsSet>,
                                            public CPUTestsBase, virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<eltwiseFusingCPUTestParamsSet> obj) {
        const auto& [eltwiseType, convertOutType] = obj.param;
        std::ostringstream result;
        result << "EltwiseType=" << eltwiseType << "_";
        result << "convertOutType=" << convertOutType;

        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto& [eltwiseType, convertOutType] = this->GetParam();
        const ov::Shape inputShape = {1, 64, 12, 12};
        ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape)};

        const auto eltwise1 = utils::make_eltwise(inputParams[0], inputParams[0], eltwiseType);
        const auto eltwise2 = utils::make_eltwise(eltwise1, inputParams[0], eltwiseType);
        const auto convert = std::make_shared<op::v0::Convert>(eltwise2, convertOutType);

        ResultVector results;
        results.push_back(std::make_shared<ov::op::v0::Result>(convert));

        function = std::make_shared<ov::Model>(results, inputParams, "groupConvolution");
        configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
    }
};

TEST_P(MergeEltwiseAndConvertTransformationCPUTest, CompareWithRefs) {
    run();
}

namespace {
std::vector<utils::EltwiseTypes> eltwiseTypes = {
    utils::EltwiseTypes::FLOOR,
    utils::EltwiseTypes::DIVIDE,
    utils::EltwiseTypes::ADD
};

std::vector<ElementType> convertOutType = {
    ov::element::f32,
    ov::element::f16,
    ov::element::u8
};

const auto mergeEltwiseAndConvertTransformationParams = ::testing::Combine(::testing::ValuesIn(eltwiseTypes),
                                                                           ::testing::ValuesIn(convertOutType));

INSTANTIATE_TEST_SUITE_P(smoke_MergeEltwiseAndConvertTransformationTest, MergeEltwiseAndConvertTransformationCPUTest,
                         mergeEltwiseAndConvertTransformationParams, MergeEltwiseAndConvertTransformationCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
