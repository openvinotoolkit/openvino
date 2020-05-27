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

        _testModel = CreateTestModel();
    }

protected:
    TestModel _testModel;
};

TEST_F(StageDependencyEdgeProcessingTests, AddStageDependencyAssertsOnNetworkInput) {
    //
    //                    -> [Output]
    // [Input] -> (Stage)
    //                    -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    auto stage = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork(0),
                                                                  OutputInfo::fromNetwork(1)});

    auto model = _testModel.getBaseModel();

    ASSERT_ANY_THROW(model->addStageDependency(stage, _testModel.getInputs().front()));
}

TEST_F(StageDependencyEdgeProcessingTests, AddStageDependencyAssertsOnStageInput) {
    //
    //                    -> [Output]
    // [Input] -> (Stage)
    //                    -> [Data] -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                     OutputInfo::fromNetwork(1)});
    auto stage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});

    auto model = _testModel.getBaseModel();

    ASSERT_ANY_THROW(model->addStageDependency(stage, stage->input(0)));
}

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

TEST_F(StageDependencyEdgeProcessingTests, AddStageDependencyAssertsIfDependencyExists) {
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
    ASSERT_ANY_THROW(model->addStageDependency(dependentStage, dependencyProducer->output(0)));
}

TEST_F(StageDependencyEdgeProcessingTests, NetWithTwoStagesHasCorrectExecOrder) {
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

TEST_F(StageDependencyEdgeProcessingTests, NetWithThreeStagesHasCorrectExecOrder) {
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

TEST_F(StageDependencyEdgeProcessingTests, ReplaceStageDependencyAssertsOnNetworkInput) {
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

    const auto edge = dependencyProducer->output(0)->dependentStagesEdges().front();

    ASSERT_ANY_THROW(model->replaceStageDependency(edge, _testModel.getInputs().front()));
}

TEST_F(StageDependencyEdgeProcessingTests, ReplaceStageDependencyAssertsOnStageInput) {
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

    const auto edge = dependencyProducer->output(0)->dependentStagesEdges().front();

    ASSERT_ANY_THROW(model->replaceStageDependency(edge, dependentStage->input(0)));
}

TEST_F(StageDependencyEdgeProcessingTests, ReplaceStageDependencyAssertsIfDependencyExists) {
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

    const auto edge = dependencyProducer->output(0)->dependentStagesEdges().front();

    ASSERT_ANY_THROW(model->replaceStageDependency(edge, dependencyProducer->output(0)));
}

TEST_F(StageDependencyEdgeProcessingTests, ReplaceStageDependencyReplacesConnection) {
    //
    //                    -> [Data] -> (Stage) -> [Output]
    //
    // [Input] -> (Stage) -> [Data] -> (Stage) -> [Output]
    //                                               |
    //                    -> [Data] ------------> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc, desc});

    _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc)});
    auto dependentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    auto initialDependencyProducer = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});
    auto resultDependencyProducer = _testModel.addStage({InputInfo::fromPrevStage(0).output(2)}, {OutputInfo::fromNetwork(2)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->addStageDependency(dependentStage, initialDependencyProducer->output(0)));

    ASSERT_TRUE(checkExecutionOrder(model, {initialDependencyProducer->id(), dependentStage->id()}));
    ASSERT_TRUE(checkExecutionOrder(model, {dependentStage->id(), resultDependencyProducer->id()}));

    const auto edge = initialDependencyProducer->output(0)->dependentStagesEdges().front();

    ASSERT_NO_THROW(model->replaceStageDependency(edge, resultDependencyProducer->output(0)));

    ASSERT_EQ(initialDependencyProducer->output(0)->dependentStagesEdges().size(), 0);
    ASSERT_EQ(edge->dependency(), resultDependencyProducer->output(0));

    ASSERT_TRUE(checkExecutionOrder(model, {resultDependencyProducer->id(), dependentStage->id()}));
}

TEST_F(StageDependencyEdgeProcessingTests, ReplaceDependentStageAssertsOnStageInput) {
    //
    //                    -----------> [Data] -> (Stage) -> [Output]
    // [Input] -> (Stage)                 |
    //                    -> [Data] -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    auto dependencyProducer = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc)});
    auto initialDependentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    auto resultDependentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->addStageDependency(initialDependentStage, dependencyProducer->output(1)));

    const auto edge = dependencyProducer->output(1)->dependentStagesEdges().front();

    ASSERT_ANY_THROW(model->replaceDependentStage(edge, resultDependentStage));
}

TEST_F(StageDependencyEdgeProcessingTests, ReplaceDependentStageAssertsIfDependencyExists) {
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

    const auto edge = dependencyProducer->output(0)->dependentStagesEdges().front();

    ASSERT_ANY_THROW(model->replaceDependentStage(edge, dependentStage));
}

TEST_F(StageDependencyEdgeProcessingTests, ReplaceDependentStageReplacesConnection) {
    //
    //                    -> [Data] -> (Stage) -> [Output]
    //
    // [Input] -> (Stage) -> [Data] -> (Stage) -> [Output]
    //                                                |
    //                    -> [Data] ------------> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc, desc});

    _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc)});
    auto initialDependentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    auto resultDependentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});
    auto dependencyProducer = _testModel.addStage({InputInfo::fromPrevStage(0).output(2)}, {OutputInfo::fromNetwork(2)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->addStageDependency(initialDependentStage, dependencyProducer->output(0)));

    ASSERT_TRUE(checkExecutionOrder(model, {dependencyProducer->id(), initialDependentStage->id()}));
    ASSERT_TRUE(checkExecutionOrder(model, {resultDependentStage->id(), dependencyProducer->id()}));

    const auto edge = dependencyProducer->output(0)->dependentStagesEdges().front();

    ASSERT_NO_THROW(model->replaceDependentStage(edge, resultDependentStage));

    ASSERT_EQ(edge->dependentStage(), resultDependentStage);

    ASSERT_TRUE(checkExecutionOrder(model, {dependencyProducer->id(), resultDependentStage->id()}));
}

TEST_F(StageDependencyEdgeProcessingTests, RemoveStageDependencyUpdatesNextPrevStages) {
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

    const auto edge = dependencyProducer->output(0)->dependentStagesEdges().front();

    ASSERT_NO_THROW(model->removeStageDependency(edge));

    const auto prevStages = dependentStage->prevStages();
    const auto nextStages = dependencyProducer->prevStages();

    auto it = std::find(prevStages.begin(), prevStages.end(), dependencyProducer);
    ASSERT_EQ(it, prevStages.end());

    it = std::find(nextStages.begin(), nextStages.end(), dependentStage);
    ASSERT_EQ(it, nextStages.end());
}

} // namespace vpu