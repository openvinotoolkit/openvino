// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/allocator.hpp>
#include <vpu/hw/mx_stage.hpp>
#include <vpu/hw/utility.hpp>
#include <vpu/utils/numeric.hpp>

#include "graph_transformer_tests.hpp"

using namespace vpu;

class VPU_AdjustDataBatchTest : public GraphTransformerTest {
public:
    int batchSize;

    TestModel testModel;

    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(GraphTransformerTest::SetUp());

        ASSERT_NO_FATAL_FAILURE(InitCompileEnv());

        batchSize = 4;

        DataDesc dataDesc(DataType::FP16, DimsOrder::NCHW, {16, 16, 3, batchSize});
        testModel = CreateTestModel(dataDesc);
    }

    void RunPass() {
        PassSet pipeline;
        pipeline.addPass(passManager->dumpModel("initial"));
        pipeline.addPass(passManager->adjustDataBatch());
        pipeline.addPass(passManager->dumpModel("adjustDataBatch"));
        pipeline.run(testModel.getBaseModel());
    }

    Stage CheckSingleSplit(const Data& data) {
        EXPECT_EQ(data->numConsumers(), 1);

        const auto& splitStage = data->singleConsumer();
        EXPECT_EQ(splitStage->type(), StageType::Split);
        EXPECT_EQ(splitStage->numOutputs(), batchSize);

        return splitStage;
    }

    Data CheckSingleConnection(const Data& data, int testInd) {
        EXPECT_EQ(data->numConsumers(), 1);

        const auto& nextStage = data->singleConsumer();
        EXPECT_EQ(nextStage->type(), StageType::None);
        EXPECT_EQ(nextStage->attrs().get<int>("test_ind"), testInd);
        EXPECT_EQ(nextStage->numOutputs(), 1);

        return nextStage->output(0);
    }
};

//
// [Input] -> (Split) -> (Split) -> (Split) -> [Output]
//

TEST_F(VPU_AdjustDataBatchTest, Case1_LinearSplit) {
    testModel.createInputs(1);
    testModel.createOutputs(1);

    testModel.addStage(
        {InputInfo::fromNetwork()},
        {OutputInfo::intermediate()}
    );
    testModel.setStageBatchInfo(
        0,
        {{0, BatchSupport::Split}}
    );

    testModel.addStage(
        {InputInfo::fromPrevStage(0)},
        {OutputInfo::intermediate()}
    );
    testModel.setStageBatchInfo(
        1,
        {{0, BatchSupport::Split}}
    );

    testModel.addStage(
        {InputInfo::fromPrevStage(1)},
        {OutputInfo::fromNetwork()}
    );
    testModel.setStageBatchInfo(
        2,
        {{0, BatchSupport::Split}}
    );

    RunPass();

    const auto& split = CheckSingleSplit(testModel.getInputs().at(0));

    Stage concat;
    for (const auto& outEdge : split->outputEdges()) {
        auto data0 = CheckSingleConnection(outEdge->output(), 0);
        auto data1 = CheckSingleConnection(data0, 1);
        auto data2 = CheckSingleConnection(data1, 2);

        if (outEdge->portInd() == 0) {
            concat = data2->singleConsumer();
            ASSERT_EQ(concat->type(), StageType::Concat);
        } else {
            ASSERT_EQ(data2->singleConsumer(), concat);
        }
    }

    ASSERT_EQ(concat->numInputs(), batchSize);
    ASSERT_EQ(concat->numOutputs(), 1);

    ASSERT_EQ(concat->output(0), testModel.getOutputs().at(0));
}

//
// [Input] -> (Split) -> (Split) -> (Batched) -> (Split) -> [Output]
//

TEST_F(VPU_AdjustDataBatchTest, Case2_LinearWithBatchedInMiddle) {
    testModel.createInputs(1);
    testModel.createOutputs(1);

    testModel.addStage(
        {InputInfo::fromNetwork()},
        {OutputInfo::intermediate()}
    );
    testModel.setStageBatchInfo(
        0,
        {{0, BatchSupport::Split}}
    );

    testModel.addStage(
        {InputInfo::fromPrevStage(0)},
        {OutputInfo::intermediate()}
    );
    testModel.setStageBatchInfo(
        1,
        {{0, BatchSupport::Split}}
    );

    testModel.addStage(
        {InputInfo::fromPrevStage(1)},
        {OutputInfo::intermediate()}
    );

    testModel.addStage(
        {InputInfo::fromPrevStage(2)},
        {OutputInfo::fromNetwork()}
    );
    testModel.setStageBatchInfo(
        3,
        {{0, BatchSupport::Split}}
    );

    RunPass();

    const auto& split1 = CheckSingleSplit(testModel.getInputs().at(0));

    Stage concat1;
    for (const auto& outEdge : split1->outputEdges()) {
        auto data0 = CheckSingleConnection(outEdge->output(), 0);
        auto data1 = CheckSingleConnection(data0, 1);

        if (outEdge->portInd() == 0) {
            concat1 = data1->singleConsumer();
            ASSERT_EQ(concat1->type(), StageType::Concat);
        } else {
            ASSERT_EQ(data1->singleConsumer(), concat1);
        }
    }

    ASSERT_EQ(concat1->numInputs(), batchSize);
    ASSERT_EQ(concat1->numOutputs(), 1);

    auto data2 = CheckSingleConnection(concat1->output(0), 2);

    const auto& split2 = CheckSingleSplit(data2);

    Stage concat2;
    for (const auto& outEdge : split2->outputEdges()) {
        auto data3 = CheckSingleConnection(outEdge->output(), 3);

        if (outEdge->portInd() == 0) {
            concat2 = data3->singleConsumer();
            ASSERT_EQ(concat1->type(), StageType::Concat);
        } else {
            ASSERT_EQ(data3->singleConsumer(), concat2);
        }
    }

    ASSERT_EQ(concat2->output(0), testModel.getOutputs().at(0));
}

//
//         -> (Batched) -> (Batched) -> [Output]
// [Input]
//         -> (Split) -> (Split) -> [Output]
//

TEST_F(VPU_AdjustDataBatchTest, Case3_SubGraphsWithSinglePlainInput) {
    testModel.createInputs(1);
    testModel.createOutputs(2);

    testModel.addStage(
        {InputInfo::fromNetwork()},
        {OutputInfo::intermediate()}
    );

    testModel.addStage(
        {InputInfo::fromPrevStage(0)},
        {OutputInfo::fromNetwork(0)}
    );

    testModel.addStage(
        {InputInfo::fromNetwork()},
        {OutputInfo::intermediate()}
    );
    testModel.setStageBatchInfo(
        2,
        {{0, BatchSupport::Split}}
    );

    testModel.addStage(
        {InputInfo::fromPrevStage(2)},
        {OutputInfo::fromNetwork(1)}
    );
    testModel.setStageBatchInfo(
        3,
        {{0, BatchSupport::Split}}
    );

    RunPass();

    auto input = testModel.getInputs().at(0);
    auto inputConsumers = input->consumers() | asVector();
    ASSERT_EQ(2, inputConsumers.size());

    Stage split, batched0;
    for (const auto& stage : inputConsumers) {
        if (stage->type() == StageType::Split) {
            ASSERT_EQ(stage->numOutputs(), batchSize);
            split = stage;
        } else {
            EXPECT_EQ(stage->type(), StageType::None);
            EXPECT_EQ(stage->attrs().get<int>("test_ind"), 0);
            batched0 = stage;
        }
    }
    ASSERT_NE(nullptr, split);
    ASSERT_NE(nullptr, batched0);

    ASSERT_EQ(batchSize, batched0->output(0)->desc().dim(Dim::N));
    ASSERT_EQ(testModel.getOutputs().at(0), CheckSingleConnection(batched0->output(0), 1));

    Stage concat;
    for (const auto& outEdge : split->outputEdges()) {
        auto data2 = CheckSingleConnection(outEdge->output(), 2);
        auto data3 = CheckSingleConnection(data2, 3);

        if (outEdge->portInd() == 0) {
            concat = data3->singleConsumer();
            ASSERT_EQ(concat->type(), StageType::Concat);
        } else {
            ASSERT_EQ(data3->singleConsumer(), concat);
        }
    }

    ASSERT_EQ(concat->numInputs(), batchSize);
    ASSERT_EQ(concat->numOutputs(), 1);

    ASSERT_EQ(concat->output(0), testModel.getOutputs().at(1));
}

//
//                               -> (Batched) -> (Batched) -> [Output]
// [Input] -> (Split) -> (Split)
//                               -> (Split) -> (Split) -> [Output]
//

TEST_F(VPU_AdjustDataBatchTest, Case4_SubGraphsWithSingleSplitInput) {
    testModel.createInputs(1);
    testModel.createOutputs(2);

    testModel.addStage(
        {InputInfo::fromNetwork()},
        {OutputInfo::intermediate()}
    );
    testModel.setStageBatchInfo(
        0,
        {{0, BatchSupport::Split}}
    );

    testModel.addStage(
        {InputInfo::fromPrevStage(0)},
        {OutputInfo::intermediate()}
    );
    testModel.setStageBatchInfo(
        1,
        {{0, BatchSupport::Split}}
    );

    testModel.addStage(
        {InputInfo::fromPrevStage(1)},
        {OutputInfo::intermediate()}
    );

    testModel.addStage(
        {InputInfo::fromPrevStage(2)},
        {OutputInfo::fromNetwork(0)}
    );

    testModel.addStage(
        {InputInfo::fromPrevStage(1)},
        {OutputInfo::intermediate()}
    );
    testModel.setStageBatchInfo(
        4,
        {{0, BatchSupport::Split}}
    );

    testModel.addStage(
        {InputInfo::fromPrevStage(4)},
        {OutputInfo::fromNetwork(1)}
    );
    testModel.setStageBatchInfo(
        5,
        {{0, BatchSupport::Split}}
    );

    RunPass();

    const auto& split1 = CheckSingleSplit(testModel.getInputs().at(0));

    Stage concat1;
    for (const auto& outEdge : split1->outputEdges()) {
        auto data0 = CheckSingleConnection(outEdge->output(), 0);
        auto data1 = CheckSingleConnection(data0, 1);

        if (outEdge->portInd() == 0) {
            concat1 = data1->singleConsumer();
            ASSERT_EQ(concat1->type(), StageType::Concat);
        } else {
            ASSERT_EQ(data1->singleConsumer(), concat1);
        }
    }

    ASSERT_EQ(concat1->numInputs(), batchSize);
    ASSERT_EQ(concat1->numOutputs(), 1);

    auto concat1Consumers = concat1->nextStages() | asVector();
    ASSERT_EQ(2, concat1Consumers.size());

    Stage split2, batched2;
    for (const auto& stage : concat1Consumers) {
        if (stage->type() == StageType::Split) {
            ASSERT_EQ(stage->numOutputs(), batchSize);
            split2 = stage;
        } else {
            EXPECT_EQ(stage->type(), StageType::None);
            EXPECT_EQ(stage->attrs().get<int>("test_ind"), 2);
            batched2 = stage;
        }
    }
    ASSERT_NE(nullptr, split2);
    ASSERT_NE(nullptr, batched2);

    ASSERT_EQ(batchSize, batched2->output(0)->desc().dim(Dim::N));
    ASSERT_EQ(testModel.getOutputs().at(0), CheckSingleConnection(batched2->output(0), 3));

    Stage concat2;
    for (const auto& outEdge : split2->outputEdges()) {
        auto data4 = CheckSingleConnection(outEdge->output(), 4);
        auto data5 = CheckSingleConnection(data4, 5);

        if (outEdge->portInd() == 0) {
            concat2 = data5->singleConsumer();
            ASSERT_EQ(concat2->type(), StageType::Concat);
        } else {
            ASSERT_EQ(data5->singleConsumer(), concat2);
        }
    }

    ASSERT_EQ(concat2->numInputs(), batchSize);
    ASSERT_EQ(concat2->numOutputs(), 1);

    ASSERT_EQ(concat2->output(0), testModel.getOutputs().at(1));
}

//
// [Input] -> (Batched) -> (Batched) ->
//                                      (Split) -> (Split) -> [Output]
// [Input] -> (Split) -> (Split) ->
//

TEST_F(VPU_AdjustDataBatchTest, Case5_MergeSubGraphs) {
    testModel.createInputs(2);
    testModel.createOutputs(1);

    testModel.addStage(
        {InputInfo::fromNetwork(0)},
        {OutputInfo::intermediate()}
    );

    testModel.addStage(
        {InputInfo::fromPrevStage(0)},
        {OutputInfo::intermediate()}
    );

    testModel.addStage(
        {InputInfo::fromNetwork(1)},
        {OutputInfo::intermediate()}
    );
    testModel.setStageBatchInfo(
        2,
        {{0, BatchSupport::Split}}
    );

    testModel.addStage(
        {InputInfo::fromPrevStage(2)},
        {OutputInfo::intermediate()}
    );
    testModel.setStageBatchInfo(
        3,
        {{0, BatchSupport::Split}}
    );

    testModel.addStage(
        {InputInfo::fromPrevStage(1), InputInfo::fromPrevStage(3)},
        {OutputInfo::intermediate()}
    );
    testModel.setStageBatchInfo(
        4,
        {{0, BatchSupport::Split}, {1, BatchSupport::Split}}
    );

    testModel.addStage(
        {InputInfo::fromPrevStage(4)},
        {OutputInfo::fromNetwork()}
    );
    testModel.setStageBatchInfo(
        5,
        {{0, BatchSupport::Split}}
    );

    RunPass();

    auto data0 = CheckSingleConnection(testModel.getInputs().at(0), 0);
    auto data1 = CheckSingleConnection(data0, 1);

    const auto& split1 = CheckSingleSplit(testModel.getInputs().at(1));

    Stage concat1;
    for (const auto& outEdge : split1->outputEdges()) {
        auto data2 = CheckSingleConnection(outEdge->output(), 2);
        auto data3 = CheckSingleConnection(data2, 3);

        if (outEdge->portInd() == 0) {
            concat1 = data3->singleConsumer();
            ASSERT_EQ(concat1->type(), StageType::Concat);
        } else {
            ASSERT_EQ(data3->singleConsumer(), concat1);
        }
    }

    ASSERT_EQ(concat1->numInputs(), batchSize);
    ASSERT_EQ(concat1->numOutputs(), 1);

    const auto& split2 = CheckSingleSplit(data1);
    const auto& split3 = CheckSingleSplit(concat1->output(0));

    Stage concat2;
    for (const auto& outEdge : split2->outputEdges()) {
        auto data4 = CheckSingleConnection(outEdge->output(), 4);
        auto data5 = CheckSingleConnection(data4, 5);

        if (outEdge->portInd() == 0) {
            concat2 = data5->singleConsumer();
            ASSERT_EQ(concat2->type(), StageType::Concat);
        } else {
            ASSERT_EQ(data5->singleConsumer(), concat2);
        }
    }
    for (const auto& outEdge : split3->outputEdges()) {
        auto data4 = CheckSingleConnection(outEdge->output(), 4);
        auto data5 = CheckSingleConnection(data4, 5);

        ASSERT_EQ(data5->singleConsumer(), concat2);
    }

    ASSERT_EQ(concat2->numInputs(), batchSize);
    ASSERT_EQ(concat2->numOutputs(), 1);

    ASSERT_EQ(concat2->output(0), testModel.getOutputs().at(0));
}
