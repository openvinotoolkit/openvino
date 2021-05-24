// Copyright (C) 2018-2021 Intel Corporation
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

TEST_F(StageDependencyEdgeProcessingTests, AddStageDependencyDoesNotAssertOnOutputProducer) {
    //
    //                    -> [Data] -> (Stage) -> [Output]
    // [Input] -> (Stage)                 |
    //                    -> [Data] -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc)});
    auto childStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    auto parentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->addStageDependency(parentStage, childStage));
}

TEST_F(StageDependencyEdgeProcessingTests, AddStageDependencyAssertsIfDependencyExists) {
    //
    //                    -> [Data] -> (Stage) -> [Output]
    // [Input] -> (Stage)                 |
    //                    -> [Data] -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc)});
    auto childStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    auto parentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->addStageDependency(parentStage, childStage));
    ASSERT_ANY_THROW(model->addStageDependency(parentStage, childStage));
}

TEST_F(StageDependencyEdgeProcessingTests, NetWithTwoStagesHasCorrectExecOrder) {
    //
    //                    -> [Data] -> (Stage) -> [Data] -> (Stage) -> [Output]
    // [Input] -> (Stage)                 |
    //                    -> [Data] -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc)});
    auto childStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    auto parentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::intermediate(desc)});
    _testModel.addStage({InputInfo::fromPrevStage(2)}, {OutputInfo::fromNetwork(1)});

    auto model = _testModel.getBaseModel();

    ASSERT_TRUE(checkExecutionOrder(model, {childStage->id(), parentStage->id()}));

    ASSERT_NO_THROW(model->addStageDependency(parentStage, childStage));

    ASSERT_TRUE(checkExecutionOrder(model, {parentStage->id(), childStage->id()}));
}

TEST_F(StageDependencyEdgeProcessingTests, NetWithThreeStagesHasCorrectExecOrder) {
    //
    //                    -> [Data] -> (Stage) -> [Data] -> (Stage) -> [Data] -> (Stage) -> [Output]
    // [Input] -> (Stage)                                      |
    //                    -> [Data] ----------------------> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc)});
    auto childStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::intermediate(desc)});
    auto parentStage = _testModel.addStage({InputInfo::fromPrevStage(2)}, {OutputInfo::intermediate(desc)});
    _testModel.addStage({InputInfo::fromPrevStage(3)}, {OutputInfo::fromNetwork(1)});

    auto model = _testModel.getBaseModel();

    ASSERT_TRUE(checkExecutionOrder(model, {childStage->id(), parentStage->id()}));

    ASSERT_NO_THROW(model->addStageDependency(parentStage, childStage));

    ASSERT_TRUE(checkExecutionOrder(model, {parentStage->id(), childStage->id()}));
}

TEST_F(StageDependencyEdgeProcessingTests, ReplaceStageDependencyParentAssertsIfDependencyExists) {
    //
    //                    -> [Data] -> (Stage) -> [Output]
    // [Input] -> (Stage)                 |
    //                    -> [Data] -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc)});
    auto childStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    auto parentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->addStageDependency(parentStage, childStage));

    const auto edge = parentStage->childDependencyEdges().front();

    ASSERT_ANY_THROW(model->replaceStageDependencyParent(edge, parentStage));
}

TEST_F(StageDependencyEdgeProcessingTests, ReplaceStageDependencyParentReplacesConnection) {
    //
    //                    -> [Data] -> (Stage) -> [Output]
    //
    // [Input] -> (Stage) -> [Data] -> (Stage) -> [Output]
    //                                    |
    //                    -> [Data] -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc, desc});

    _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc)});
    auto childStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    auto initialParentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});
    auto resultParentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(2)}, {OutputInfo::fromNetwork(2)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->addStageDependency(initialParentStage, childStage));

    ASSERT_TRUE(checkExecutionOrder(model, {initialParentStage->id(), childStage->id()}));
    ASSERT_TRUE(checkExecutionOrder(model, {childStage->id(), resultParentStage->id()}));

    const auto edge = initialParentStage->childDependencyEdges().front();

    ASSERT_NO_THROW(model->replaceStageDependencyParent(edge, resultParentStage));

    ASSERT_EQ(initialParentStage->childDependencyEdges().size(), 0);
    ASSERT_EQ(edge->parent(), resultParentStage);

    ASSERT_TRUE(checkExecutionOrder(model, {resultParentStage->id(), childStage->id()}));
}

TEST_F(StageDependencyEdgeProcessingTests, ReplaceStageDependencyChildAssertsIfDependencyExists) {
    //
    //                    -> [Data] -> (Stage) -> [Output]
    // [Input] -> (Stage)                 |
    //                    -> [Data] -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc)});
    auto childStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    auto parentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->addStageDependency(parentStage, childStage));

    const auto edge = parentStage->childDependencyEdges().front();

    ASSERT_ANY_THROW(model->replaceStageDependencyChild(edge, childStage));
}

TEST_F(StageDependencyEdgeProcessingTests, ReplaceStageDependencyChildReplacesConnection) {
    //
    //                    -> [Data] -> (Stage) -> [Output]
    //
    // [Input] -> (Stage) -> [Data] -> (Stage) -> [Output]
    //                                    |
    //                    -> [Data] -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc, desc});

    _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc)});
    auto initialChildStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    auto resultChildStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});
    auto parentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(2)}, {OutputInfo::fromNetwork(2)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->addStageDependency(parentStage, initialChildStage));

    ASSERT_TRUE(checkExecutionOrder(model, {parentStage->id(), initialChildStage->id()}));
    ASSERT_TRUE(checkExecutionOrder(model, {resultChildStage->id(), parentStage->id()}));

    const auto edge = parentStage->childDependencyEdges().front();

    ASSERT_NO_THROW(model->replaceStageDependencyChild(edge, resultChildStage));

    ASSERT_EQ(edge->child(), resultChildStage);

    ASSERT_TRUE(checkExecutionOrder(model, {parentStage->id(), resultChildStage->id()}));
}

TEST_F(StageDependencyEdgeProcessingTests, RemoveStageDependencyUpdatesNextPrevStages) {
    //
    //                    -> [Data] -> (Stage) -> [Output]
    // [Input] -> (Stage)                 |
    //                    -> [Data] -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                     OutputInfo::intermediate(desc)});
    auto childStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    auto parentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->addStageDependency(parentStage, childStage));

    const auto edge = parentStage->childDependencyEdges().front();

    ASSERT_NO_THROW(model->removeStageDependency(edge));

    const auto prevStages = childStage->prevStages();
    const auto nextStages = parentStage->prevStages();

    auto it = std::find(prevStages.begin(), prevStages.end(), parentStage);
    ASSERT_EQ(it, prevStages.end());

    it = std::find(nextStages.begin(), nextStages.end(), childStage);
    ASSERT_EQ(it, nextStages.end());
}

TEST_F(StageDependencyEdgeProcessingTests, RemoveStageDependencyViaDataToShapeEdgeUpdatesNextPrevStages) {
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
    auto childStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    auto parentStage = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->addStageDependency(parentStage, childStage));

    ASSERT_NO_THROW(model->removeStageDependency(parentStage, childStage));

    const auto prevStages = childStage->prevStages();
    const auto nextStages = parentStage->prevStages();

    auto it = std::find(prevStages.begin(), prevStages.end(), parentStage);
    ASSERT_EQ(it, prevStages.end());

    it = std::find(nextStages.begin(), nextStages.end(), childStage);
    ASSERT_EQ(it, nextStages.end());
}

} // namespace vpu
