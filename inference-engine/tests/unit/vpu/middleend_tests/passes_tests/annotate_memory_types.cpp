// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"
#include "common_test_utils/common_utils.hpp"

namespace vpu {

namespace ie = InferenceEngine;

namespace {

using TestParam = std::tuple<
    // input MemoryType of first stage is always DDR
    std::tuple<MemoryType, MemoryType, MemoryType>, // outputs MemoryTypes for first stage
    std::tuple<MemoryType, MemoryType>              // outputs MemoryTypes for second stage
    // output MemoryType of third stage is always DDR
>;

}

class AnnotateMemoryTypes : public GraphTransformerTest, public testing::WithParamInterface<TestParam> {
protected:
    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(GraphTransformerTest::SetUp());
        config.compileConfig().enableMemoryTypesAnnotation = true;

        ASSERT_NO_FATAL_FAILURE(InitCompileEnv());
        ASSERT_NO_FATAL_FAILURE(InitPipeline());

        const auto& parameters = GetParam();
        const auto& firstStageOutputs = CommonTestUtils::tuple2Vector(std::get<0>(parameters));
        const auto& secondStageOutputs = CommonTestUtils::tuple2Vector(std::get<1>(parameters));
        ASSERT_NO_FATAL_FAILURE(InitModel(firstStageOutputs, secondStageOutputs));
        ASSERT_NO_FATAL_FAILURE(Compile());
        ASSERT_NO_FATAL_FAILURE(Validate(firstStageOutputs, secondStageOutputs));
    }

    void Compile() {
        m_pipeline.run(m_testModel.getBaseModel());
    }

    void Validate(const std::vector<MemoryType>& firstStageOutputs, const std::vector<MemoryType>& secondStageOutputs) {
        const auto& stages = m_testModel.getStages();
        ASSERT_EQ(stages.size(), 3);

        const auto& stage0 = stages.front();
        const auto& stage1 = stages[1];
        const auto& stage2 = stages.back();

        ASSERT_TRUE(CommonTestUtils::endsWith(stage0->name(), "@[DDR]->" + GenerateSuffix(firstStageOutputs)));
        ASSERT_TRUE(CommonTestUtils::endsWith(stage1->name(), "@" + GenerateSuffix(firstStageOutputs) + "->" + GenerateSuffix(secondStageOutputs)));
        ASSERT_TRUE(CommonTestUtils::endsWith(stage2->name(), "@" + GenerateSuffix(secondStageOutputs) + "->[DDR]"));
    }

protected:
    TestModel m_testModel;

private:
    void InitModel(const std::vector<MemoryType>& firstStageOutputs, const std::vector<MemoryType>& secondStageOutputs) {
        m_testModel = CreateTestModel();

        m_testModel.createInputs();
        m_testModel.createOutputs();

        const auto generateInputs = [](const std::vector<MemoryType>& inputsMemoryTypes, std::size_t prevStageIndex) {
            std::vector<int> indices(inputsMemoryTypes.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::vector<InputInfo> inputs;
            std::transform(indices.cbegin(), indices.cend(), std::back_inserter(inputs),
                [&prevStageIndex](int index) { return InputInfo::fromPrevStage(static_cast<int>(prevStageIndex), index); });
            return inputs;
        };

        const auto generateOutputs = [](const std::vector<MemoryType>& outputsMemoryTypes) {
            std::vector<OutputInfo> outputs;
            std::transform(outputsMemoryTypes.cbegin(), outputsMemoryTypes.cend(), std::back_inserter(outputs),
                [](MemoryType type) { return OutputInfo::intermediate(type); });
            return outputs;
        };

        m_testModel.addStage({InputInfo::fromNetwork()}, generateOutputs(firstStageOutputs));
        m_testModel.addStage(generateInputs(firstStageOutputs, 0), generateOutputs(secondStageOutputs));
        m_testModel.addStage(generateInputs(secondStageOutputs, 1), {OutputInfo::fromNetwork()});
    }

    void InitPipeline() {
        m_pipeline = PassSet();
        m_pipeline.addPass(passManager->annotateMemoryTypes());
    }

    template<class T>
    static std::string GenerateSuffix(const T& outputs) {
        std::stringstream suffix;
        printTo(suffix, outputs);
        return suffix.str();
    }

    PassSet m_pipeline;
};

TEST_P(AnnotateMemoryTypes, SubgraphOf3Stages) {
}

INSTANTIATE_TEST_SUITE_P(unit, AnnotateMemoryTypes, testing::Combine(
    testing::Combine(
        testing::Values(MemoryType::DDR, MemoryType::CMX),
        testing::Values(MemoryType::DDR, MemoryType::CMX),
        testing::Values(MemoryType::DDR, MemoryType::CMX)),
    testing::Combine(
        testing::Values(MemoryType::DDR, MemoryType::CMX),
        testing::Values(MemoryType::DDR, MemoryType::CMX))
));

} // namespace vpu
