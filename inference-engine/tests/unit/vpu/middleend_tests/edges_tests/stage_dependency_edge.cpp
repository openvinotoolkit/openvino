// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

namespace vpu {

namespace ie = InferenceEngine;

class StageDependencyEdgeProcessingTests : public GraphTransformerTest {
protected:
    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(GraphTransformerTest::SetUp());

        ASSERT_NO_FATAL_FAILURE(InitCompileEnv());

        _middleEnd = passManager->buildMiddleEnd();
        _testModel = CreateTestModel();
    }

protected:
    TestModel _testModel;
    PassSet::Ptr _middleEnd = nullptr;
};

TEST_F(StageDependencyEdgeProcessingTests, AddStageDependencyDoesNotAssertOnOutputData) {
    //
    //                    -> [Data] -> (Stage) -> [Output]
    // [Input] -> (Stage)                            |
    //                    -> [Data] ------------> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc)});
    auto dependentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    auto dependencyProducer = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->addStageDependency(dependentStage, dependencyProducer->output(0)));
}

TEST_F(StageDependencyEdgeProcessingTests, DISABLED_NetWithTwoStagesHasCorrectExecOrder) {
    //
    //                    -> [Data] -> (Stage) -> [Data] -> (Stage) -> [Output]
    // [Input] -> (Stage)                            |
    //                    -> [Data] ------------> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc)});
    auto dependentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    auto dependencyProducer = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::intermediate(desc)});
    _testModel.addStage({InputInfo::fromPrevStage(2)}, {OutputInfo::fromNetwork(1)});

    auto model = _testModel.getBaseModel();

    ASSERT_TRUE(checkExecutionOrder(model, {dependentStage->id(), dependencyProducer->id()}));

    ASSERT_NO_THROW(model->addStageDependency(dependentStage, dependencyProducer->output(0)));

    ASSERT_TRUE(checkExecutionOrder(model, {dependencyProducer->id(), dependentStage->id()}));
}

TEST_F(StageDependencyEdgeProcessingTests, DISABLED_NetWithThreeStagesHasCorrectExecOrder) {
    //
    //                    -> [Data] -> (Stage) -> [Data] -> (Stage) -> [Data] -> (Stage) -> [Output]
    // [Input] -> (Stage)                                                 |
    //                    -> [Data] ---------------------------------> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc)});
    auto dependentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::intermediate(desc)});
    auto dependencyProducer = _testModel.addStage({InputInfo::fromPrevStage(2)}, {OutputInfo::intermediate(desc)});
    _testModel.addStage({InputInfo::fromPrevStage(3)}, {OutputInfo::fromNetwork(1)});

    auto model = _testModel.getBaseModel();

    ASSERT_TRUE(checkExecutionOrder(model, {dependentStage->id(), dependencyProducer->id()}));

    ASSERT_NO_THROW(model->addStageDependency(dependentStage, dependencyProducer->output(0)));

    ASSERT_TRUE(checkExecutionOrder(model, {dependencyProducer->id(), dependentStage->id()}));
}

} // namespace vpu