// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <shared_test_classes/base/ov_subgraph.hpp>
#include <ngraph_functions/builders.hpp>
#include "test_utils/cpu_test_utils.hpp"
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>

using namespace ov::test;
using namespace CPUTestUtils;

namespace CPULayerTestsDefinitions {
using InsertConvertI64I32CPUTestParams = std::tuple<ElementType, InputShape>;

class InsertConvertI64I32CPUTest : public testing::WithParamInterface<InsertConvertI64I32CPUTestParams>,
                                   virtual public SubgraphBaseTest,
                                   public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<InsertConvertI64I32CPUTestParams>& obj) {
        ElementType inType;
        InputShape inputShape;
        std::tie(inType, inputShape) = obj.param;

        std::ostringstream result;
        result << "IS=" << inputShape << "_";
        result << "Prc=" << inType;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        configuration[InferenceEngine::PluginConfigInternalParams::KEY_CPU_NATIVE_I64] = InferenceEngine::PluginConfigParams::YES;

        InputShape inputShape;
        std::tie(inType, inputShape) = this->GetParam();

        init_input_shapes({inputShape});
        auto inputParams = ngraph::builder::makeDynamicParams(inType, inputDynamicShapes);
        auto nonZero = std::make_shared<ov::op::v3::NonZero>(inputParams[0]);

        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(nonZero)};
        function = std::make_shared<ov::Model>(results, inputParams, "insertConvertI64I32");
    }
};

TEST_P(InsertConvertI64I32CPUTest, CompareWithRefs) {
    run();
    CheckNumberOfNodesWithType(compiledModel, "Convert", 2);
}

const InputShape inputShapes = {
        {}, {{1, 3, 32, 32}}
};

INSTANTIATE_TEST_SUITE_P(smoke_CustomOp,
                         InsertConvertI64I32CPUTest,
                         ::testing::Combine(::testing::Values(ElementType::i64), ::testing::Values(inputShapes)),
                         InsertConvertI64I32CPUTest::getTestCaseName);

} // namespace CPULayerTestsDefinitions
