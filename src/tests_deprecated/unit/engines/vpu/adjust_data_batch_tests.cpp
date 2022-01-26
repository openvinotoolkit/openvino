// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/middleend/allocator/allocator.hpp>
#include <vpu/stages/mx_stage.hpp>
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

    DataVector checkSingleLoopStart(const Data& data) {
        EXPECT_EQ(data->desc().dim(Dim::N), batchSize);
        EXPECT_EQ(data->numConsumers(), 2);

        DataVector outputs;
        for (const auto& consumer : data->consumers()) {
            EXPECT_TRUE(consumer->type() == StageType::LoopStart || consumer->type() == StageType::LoopEnd);
            if (consumer->type() == StageType::LoopStart) {
                for (const auto& output : consumer->outputs()) {
                    EXPECT_EQ(output->desc().dim(Dim::N), 1);
                    outputs.push_back(output);
                }
            }
        }

        return outputs;
    }

    DataVector checkBranches(const Data& root, const std::vector<StageType>& consumersTypes) {
        auto successors = DataVector{};

        const auto& consumers = root->consumers() | asVector();
        EXPECT_EQ(consumers.size(), consumersTypes.size());
        for (std::size_t i = 0; i < consumers.size(); ++i) {
            const auto& consumer = consumers[i];
            const auto& expected = consumersTypes[i];
            EXPECT_EQ(consumer->type(), expected);

            EXPECT_EQ(consumer->numOutputs(), 1);
            const auto& output = consumer->output(0);
            successors.push_back(output);

            if (expected == StageType::LoopStart) {
                EXPECT_EQ(consumer->numOutputs(), 1);
                EXPECT_EQ(output->desc().dim(Dim::N), 1);
            } else if (expected == StageType::LoopEnd) {
                EXPECT_EQ(output->desc().dim(Dim::N), batchSize);
            }
        }

        return successors;
    }

    DataVector checkSingleLoopEnd(const Data& data) {
        EXPECT_EQ(data->numConsumers(), 1);

        const auto& consumer = data->singleConsumer();
        EXPECT_EQ(consumer->type(), StageType::LoopEnd);
        DataVector outputs;
        for (const auto& output : consumer->outputs()) {
            EXPECT_EQ(output->desc().dim(Dim::N), batchSize);
            outputs.push_back(output);
        }

        return outputs;
    }

    static Data CheckSingleConnection(const Data& data, int testInd, int batch = 1) {
        EXPECT_EQ(data->numConsumers(), 1);

        const auto& consumer = data->singleConsumer();
        EXPECT_EQ(consumer->type(), StageType::None);
        EXPECT_EQ(consumer->attrs().get<int>("test_ind"), testInd);
        EXPECT_TRUE(consumer->numOutputs() == 1);
        const auto& output = consumer->output(0);
        EXPECT_EQ(output->desc().dim(Dim::N), batch);
        return output;
    }

    static Data singleElement(const DataVector& dataObjects) {
        EXPECT_EQ(dataObjects.size(), 1);
        return dataObjects.front();
    }
};


TEST_F(VPU_AdjustDataBatchTest, Case1_LinearSplit) {
    //
    // [Input] -> (Split) -> (Split) -> (Split) -> [Output]
    //

    testModel.createInputs(1);
    testModel.createOutputs(1);

    testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(0, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(0)}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(1, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(1)}, {OutputInfo::fromNetwork()});
    testModel.setStageBatchInfo(2, {{0, BatchSupport::Split}});

    RunPass();

    const auto& data0 = singleElement(checkSingleLoopStart(testModel.getInputs().at(0)));
    const auto& data1 = CheckSingleConnection(data0, 0);
    const auto& data2 = CheckSingleConnection(data1, 1);
    const auto& data3 = CheckSingleConnection(data2, 2);
    const auto& data4 = checkSingleLoopEnd(data3);
    ASSERT_EQ(data4, testModel.getOutputs());
}

TEST_F(VPU_AdjustDataBatchTest, Case2_LinearWithBatchedInMiddle) {
    //
    // [Input] -> (Split) -> (Split) -> (Split) -> (Batched) -> (Split) -> (Split) -> (Split) -> [Output]
    //

    testModel.createInputs(1);
    testModel.createOutputs(1);

    testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(0, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(0)}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(1, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(1)}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(2, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(2)}, {OutputInfo::intermediate()});

    testModel.addStage({InputInfo::fromPrevStage(3)}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(4, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(4)}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(5, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(5)}, {OutputInfo::fromNetwork()});
    testModel.setStageBatchInfo(6, {{0, BatchSupport::Split}});

    RunPass();

    const auto& data0 = singleElement(checkSingleLoopStart(testModel.getInputs().at(0)));
    const auto& data1 = CheckSingleConnection(data0, 0);
    const auto& data2 = CheckSingleConnection(data1, 1);
    const auto& data3 = CheckSingleConnection(data2, 2);
    const auto& data4 = singleElement(checkSingleLoopEnd(data3));

    const auto& data5 = CheckSingleConnection(data4, 3, batchSize);

    const auto& data6 = singleElement(checkSingleLoopStart(data5));
    const auto& data7 = CheckSingleConnection(data6, 4);
    const auto& data8 = CheckSingleConnection(data7, 5);
    const auto& data9 = CheckSingleConnection(data8, 6);
    const auto& data10 = checkSingleLoopEnd(data9);

    ASSERT_EQ(data10, testModel.getOutputs());
}

TEST_F(VPU_AdjustDataBatchTest, Case3_SubGraphsWithSinglePlainInput) {
    //
    //         -> (Batched) -> (Batched) -> (Batched) -> [Output]
    // [Input]
    //         -> (Split) - > (Split) -> (Split) -> [Output]
    //

    testModel.createInputs(1);
    testModel.createOutputs(2);

    testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate()});
    testModel.addStage({InputInfo::fromPrevStage(0)}, {OutputInfo::intermediate()});
    testModel.addStage({InputInfo::fromPrevStage(1)}, {OutputInfo::fromNetwork(0)});

    testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(3, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(3)}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(4, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(4)}, {OutputInfo::fromNetwork(1)});
    testModel.setStageBatchInfo(5, {{0, BatchSupport::Split}});

    RunPass();

    const auto& branches = checkBranches(testModel.getInputs().at(0), {StageType::None, StageType::LoopStart, StageType::LoopEnd});
    const auto& withBatch = branches[0];
    const auto& withLoop = branches[1];
    ASSERT_EQ(branches[2], testModel.getOutputs().at(1));

    ASSERT_EQ(withBatch->producer()->attrs().get<int>("test_ind"), 0);
    ASSERT_EQ(withBatch->desc().dim(Dim::N), batchSize);
    const auto& withBatch0 = CheckSingleConnection(withBatch,  1, batchSize);
    const auto& withBatch1 = CheckSingleConnection(withBatch0, 2, batchSize);
    ASSERT_EQ(withBatch1, testModel.getOutputs().at(0));

    const auto& withLoop0 = CheckSingleConnection(withLoop,  3);
    const auto& withLoop1 = CheckSingleConnection(withLoop0, 4);
    const auto& withLoop2 = CheckSingleConnection(withLoop1, 5);
    const auto& withLoop3 = singleElement(checkSingleLoopEnd(withLoop2));
    ASSERT_EQ(withLoop3, testModel.getOutputs().at(1));
}

TEST_F(VPU_AdjustDataBatchTest, Case4_SubGraphsWithSingleSplitInput) {
    //
    //                                          -> (Batched) -> (Batched) -> (Batched) -> [Output]
    // [Input] -> (Split) -> (Split) -> (Split)
    //                                          -> (Split) -> (Split) -> (Split) -> [Output]
    //

    testModel.createInputs(1);
    testModel.createOutputs(2);

    testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(0, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(0)}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(1, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(1)}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(2, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(2)}, {OutputInfo::intermediate()});
    testModel.addStage({InputInfo::fromPrevStage(3)}, {OutputInfo::intermediate()});
    testModel.addStage({InputInfo::fromPrevStage(4)}, {OutputInfo::fromNetwork(0)});

    testModel.addStage({InputInfo::fromPrevStage(2)}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(6, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(6)}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(7, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(7)}, {OutputInfo::fromNetwork(1)});
    testModel.setStageBatchInfo(8, {{0, BatchSupport::Split}});

    RunPass();

    const auto& data0 = singleElement(checkSingleLoopStart(testModel.getInputs().at(0)));
    const auto& data1 = CheckSingleConnection(data0, 0);
    const auto& data2 = CheckSingleConnection(data1, 1);
    const auto& data3 = CheckSingleConnection(data2, 2);
    const auto& data4 = singleElement(checkSingleLoopEnd(data3));

    const auto& branches = checkBranches(data4, {StageType::None, StageType::LoopStart, StageType::LoopEnd});
    const auto& withBatch = branches[0];
    const auto& withLoop = branches[1];
    ASSERT_EQ(branches[2], testModel.getOutputs().at(1));

    ASSERT_EQ(withBatch->producer()->attrs().get<int>("test_ind"), 3);
    ASSERT_EQ(withBatch->desc().dim(Dim::N), batchSize);
    const auto& withBatch0 = CheckSingleConnection(withBatch,  4, batchSize);
    const auto& withBatch1 = CheckSingleConnection(withBatch0, 5, batchSize);
    ASSERT_EQ(withBatch1, testModel.getOutputs().at(0));

    const auto& withLoop0 = CheckSingleConnection(withLoop,  6);
    const auto& withLoop1 = CheckSingleConnection(withLoop0, 7);
    const auto& withLoop2 = CheckSingleConnection(withLoop1, 8);
    const auto& withLoop3 = singleElement(checkSingleLoopEnd(withLoop2));
    ASSERT_EQ(withLoop3, testModel.getOutputs().at(1));
}

TEST_F(VPU_AdjustDataBatchTest, Case5_MergeSubGraphs) {
    //
    // [Input] -> (Batched) -> (Batched) -> (Batched) ->
    //                                                   (Split) -> (Split) -> (Split) -> [Output]
    // [Input] -> (Split) -> (Split) -> (Split)       ->
    //

    testModel.createInputs(2);
    testModel.createOutputs(1);

    testModel.addStage({InputInfo::fromNetwork(0)}, {OutputInfo::intermediate()});
    testModel.addStage({InputInfo::fromPrevStage(0)}, {OutputInfo::intermediate()});
    testModel.addStage({InputInfo::fromPrevStage(1)}, {OutputInfo::intermediate()});

    testModel.addStage({InputInfo::fromNetwork(1)}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(3, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(3)}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(4, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(4)}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(5, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(2), InputInfo::fromPrevStage(5)}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(6, {{0, BatchSupport::Split}, {1, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(6)}, {OutputInfo::intermediate()});
    testModel.setStageBatchInfo(7, {{0, BatchSupport::Split}});

    testModel.addStage({InputInfo::fromPrevStage(7)}, {OutputInfo::fromNetwork()});
    testModel.setStageBatchInfo(8, {{0, BatchSupport::Split}});

    RunPass();

    const auto& data0 = CheckSingleConnection(testModel.getInputs().at(0), 0, batchSize);
    const auto& data1 = CheckSingleConnection(data0, 1, batchSize);
    const auto& data2 = CheckSingleConnection(data1, 2, batchSize);
    const auto& loopStartOutputs0 = checkSingleLoopStart(data2);

    const auto& data3 = singleElement(checkSingleLoopStart(testModel.getInputs().at(1)));
    const auto& data4 = CheckSingleConnection(data3, 3);
    const auto& data5 = CheckSingleConnection(data4, 4);
    const auto& data6 = CheckSingleConnection(data5, 5);
    const auto& data7 = singleElement(checkSingleLoopEnd(data6));
    const auto& loopStartOutputs1 = checkSingleLoopStart(data7);

    ASSERT_EQ(loopStartOutputs0, loopStartOutputs1);
    ASSERT_EQ(loopStartOutputs0.size(), 2);
    const auto& data8 = loopStartOutputs0.front();
    const auto& data9 = loopStartOutputs0.back();

    const auto& data10 = CheckSingleConnection(data8, 6);
    const auto& data11 = CheckSingleConnection(data9, 6);
    ASSERT_EQ(data10, data11);

    const auto& data12 = CheckSingleConnection(data10, 7);
    const auto& data13 = CheckSingleConnection(data12, 8);
    const auto& data14 = singleElement(checkSingleLoopEnd(data13));
    ASSERT_EQ(data14, testModel.getOutputs().at(0));
}
