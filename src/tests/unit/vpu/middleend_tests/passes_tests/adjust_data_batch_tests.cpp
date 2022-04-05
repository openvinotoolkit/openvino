// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/stages/mx_stage.hpp>
#include <vpu/utils/numeric.hpp>

#include "graph_transformer_tests.hpp"

using namespace vpu;

class VPU_AdjustDataBatchTest : public GraphTransformerTest {
protected:
    const int batchSize = 4;
    TestModel testModel;

public:
    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(GraphTransformerTest::SetUp());

        ASSERT_NO_FATAL_FAILURE(InitCompileEnv());

        testModel = CreateTestModel();
    }

    void RunPass() {
        PassSet pipeline;
        pipeline.addPass(passManager->dumpModel("initial"));
        pipeline.addPass(passManager->adjustDataBatch());
        pipeline.addPass(passManager->dumpModel("adjustDataBatch"));
        pipeline.run(testModel.getBaseModel());
    }

    DataVector checkSingleLoopStart(const Data& data) {
        EXPECT_EQ(data->desc().dim(Dim::N), 4);
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
                EXPECT_EQ(output->desc().dim(Dim::N), 4);
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
            EXPECT_EQ(output->desc().dim(Dim::N), 4);
            outputs.push_back(output);
        }

        return outputs;
    }

    static Data CheckSingleConnection(const Data& data, int testInd, int batch = 1) {
        EXPECT_EQ(data->numConsumers(), 1);

        const auto& consumer = data->singleConsumer();
        EXPECT_EQ(consumer->type(), StageType::None);
        EXPECT_EQ(consumer->attrs().get<int>("test_ind"), testInd);
        EXPECT_EQ(consumer->numOutputs(), 1);
        const auto& output = consumer->output(0);
        EXPECT_EQ(output->desc().dim(Dim::N), batch);
        return output;
    }

    static Data singleElement(const DataVector& dataObjects) {
        EXPECT_EQ(dataObjects.size(), 1);
        return dataObjects.front();
    }
};

TEST_F(VPU_AdjustDataBatchTest, LinearWithBatchedInTheEnd) {
    //
    // [Input] -> (Split) -> (Split) -> (Split) -> (Split) -> (Split) -> (Split) -> (Batched) -> [Output]
    //
    const DataDesc desc{16, 16, 3, batchSize};

    testModel.createInputs({desc});
    testModel.createOutputs({desc});

    for (int i = 0; i < 6; i++) {
        if (i > 0)
            testModel.addStage({InputInfo::fromPrevStage(i - 1)}, {OutputInfo::intermediate(desc)});
        else
            testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc)});
        testModel.setStageBatchInfo(i, {{0, BatchSupport::Split}});
    }
    testModel.addStage({InputInfo::fromPrevStage(5)}, {OutputInfo::fromNetwork(0)});

    RunPass();

    const auto& data0 = singleElement(checkSingleLoopStart(testModel.getInputs().at(0)));
    const auto& data1 = CheckSingleConnection(data0, 0);
    const auto& data2 = CheckSingleConnection(data1, 1);
    const auto& data3 = CheckSingleConnection(data2, 2);
    const auto& data4 = CheckSingleConnection(data3, 3);
    const auto& data5 = CheckSingleConnection(data4, 4);
    const auto& data6 = CheckSingleConnection(data5, 5);
    const auto& data7 = singleElement(checkSingleLoopEnd(data6));

    const auto& data8 = CheckSingleConnection(data7, 6, batchSize);

    ASSERT_EQ(data8, testModel.getOutputs().at(0));
}

TEST_F(VPU_AdjustDataBatchTest, BranchedWithBatchSplitItems) {
    //                                                                                      -> (Batched) -> [Output]
    // [Input] -> (Split) -> (Split) -> (Split) -> (Split) -> (Split) -> (Split) -> (Split)
    //                                                                                      -> (Batched) -> [Output]
    const DataDesc desc{16, 16, 3, batchSize};

    testModel.createInputs({desc});
    testModel.createOutputs({desc, desc});

    for (int i = 0; i < 7; i++) {
        if (i > 0)
            testModel.addStage({InputInfo::fromPrevStage(i - 1)}, {OutputInfo::intermediate(desc)});
        else
            testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc)});
        testModel.setStageBatchInfo(i, {{0, BatchSupport::Split}});
    }

    testModel.addStage({InputInfo::fromPrevStage(6)}, {OutputInfo::fromNetwork(0)});
    testModel.addStage({InputInfo::fromPrevStage(6)}, {OutputInfo::fromNetwork(1)});

    RunPass();

    const auto& data0 = singleElement(checkSingleLoopStart(testModel.getInputs().at(0)));
    const auto& data1 = CheckSingleConnection(data0, 0);
    const auto& data2 = CheckSingleConnection(data1, 1);
    const auto& data3 = CheckSingleConnection(data2, 2);
    const auto& data4 = CheckSingleConnection(data3, 3);
    const auto& data5 = CheckSingleConnection(data4, 4);
    const auto& data6 = CheckSingleConnection(data5, 5);
    const auto& data7 = CheckSingleConnection(data6, 6);
    const auto& data8 = singleElement(checkSingleLoopEnd(data7));

    const auto& branches = checkBranches(data8, {StageType::None, StageType::None});
    const auto& withBatch = branches[0];
    const auto& withBatch_1 = branches[1];

    ASSERT_EQ(withBatch->producer()->attrs().get<int>("test_ind"), 7);
    ASSERT_EQ(withBatch->desc().dim(Dim::N), batchSize);
    ASSERT_EQ(withBatch, testModel.getOutputs().at(0));

    ASSERT_EQ(withBatch_1->producer()->attrs().get<int>("test_ind"), 8);
    ASSERT_EQ(withBatch_1->desc().dim(Dim::N), batchSize);
    ASSERT_EQ(withBatch_1, testModel.getOutputs().at(1));
}

TEST_F(VPU_AdjustDataBatchTest, LinearWithBatchedInTheBeginning) {
    //
    // [Input] -> (Batched) -> (Split) -> (Split) -> (Split) -> (Split) -> (Split) -> (Split) -> [Output]
    //
    const DataDesc desc{16, 16, 3, batchSize};

    testModel.createInputs({desc});
    testModel.createOutputs({desc});

    for (int i = 0; i < 6; i++) {
        if (i > 0)
            testModel.addStage({InputInfo::fromPrevStage(i - 1)}, {OutputInfo::intermediate(desc)});
        else
            testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc)});
        if (i > 0)
            testModel.setStageBatchInfo(i, {{0, BatchSupport::Split}});
    }

    testModel.addStage({InputInfo::fromPrevStage(5)}, {OutputInfo::fromNetwork()});
    testModel.setStageBatchInfo(6, {{0, BatchSupport::Split}});

    RunPass();

    const auto& data0 = CheckSingleConnection(testModel.getInputs().at(0), 0, batchSize);
    const auto& data7 = singleElement(checkSingleLoopStart(data0));
    const auto& data3 = CheckSingleConnection(data7, 1);
    const auto& data4 = CheckSingleConnection(data3, 2);
    const auto& data5 = CheckSingleConnection(data4, 3);
    const auto& data6 = CheckSingleConnection(data5, 4);
    const auto& data8 = CheckSingleConnection(data6, 5);
    const auto& data10 = CheckSingleConnection(data8, 6);
    const auto& data11 = checkSingleLoopEnd(data10);

    ASSERT_EQ(data11, testModel.getOutputs());
}

TEST_F(VPU_AdjustDataBatchTest, BranchedWithBatchItemsInTheEnd) {
    //                                                                                      -> (Batched) -> [Output]
    // [Input] -> (Split) -> (Split) -> (Split) -> (Split) -> (Split) -> (Split) -> (Batch)
    //                                                                                      -> (Batched) -> [Output]
    const DataDesc desc{16, 16, 3, batchSize};

    testModel.createInputs({desc});
    testModel.createOutputs({desc, desc});

    for (int i = 0; i < 6; i++) {
        if (i > 0)
            testModel.addStage({InputInfo::fromPrevStage(i - 1)}, {OutputInfo::intermediate(desc)});
        else
            testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc)});
        testModel.setStageBatchInfo(i, {{0, BatchSupport::Split}});
    }

    testModel.addStage({InputInfo::fromPrevStage(5)}, {OutputInfo::intermediate(desc)});
    testModel.addStage({InputInfo::fromPrevStage(6)}, {OutputInfo::fromNetwork(0)});
    testModel.addStage({InputInfo::fromPrevStage(6)}, {OutputInfo::fromNetwork(1)});

    RunPass();

    const auto& data0 = singleElement(checkSingleLoopStart(testModel.getInputs().at(0)));
    const auto& data1 = CheckSingleConnection(data0, 0);
    const auto& data2 = CheckSingleConnection(data1, 1);
    const auto& data3 = CheckSingleConnection(data2, 2);
    const auto& data4 = CheckSingleConnection(data3, 3);
    const auto& data5 = CheckSingleConnection(data4, 4);
    const auto& data6 = CheckSingleConnection(data5, 5);
    const auto& data7 = singleElement(checkSingleLoopEnd(data6));

    const auto& data8 = CheckSingleConnection(data7, 6, batchSize);

    const auto& branches = checkBranches(data8, {StageType::None, StageType::None});
    const auto& withBatch = branches[0];
    const auto& withBatch_1 = branches[1];

    ASSERT_EQ(withBatch->producer()->attrs().get<int>("test_ind"), 7);
    ASSERT_EQ(withBatch->desc().dim(Dim::N), batchSize);
    ASSERT_EQ(withBatch, testModel.getOutputs().at(0));

    ASSERT_EQ(withBatch_1->producer()->attrs().get<int>("test_ind"), 8);
    ASSERT_EQ(withBatch_1->desc().dim(Dim::N), batchSize);
    ASSERT_EQ(withBatch_1, testModel.getOutputs().at(1));
}

TEST_F(VPU_AdjustDataBatchTest, DISABLED_BranchedWithSplitAndBatchItemsInTheEnd) {
    //
    //                                         -> (Split) -> (Batched) -> [Output]
    // [Input] -> (Split) -> (Split) -> (Split)
    //                                         -> (Split) -> [Output]
    //
    const DataDesc desc{16, 16, 3, batchSize};

    testModel.createInputs({desc});
    testModel.createOutputs({desc, desc});

    for (int i = 0; i < 5; i++) {
        if (i > 0)
            testModel.addStage({InputInfo::fromPrevStage(i - 1)}, {OutputInfo::intermediate(desc)});
        else
            testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc)});
        if (i != 3)
            testModel.setStageBatchInfo(i, {{0, BatchSupport::Split}});
    }

    testModel.addStage({InputInfo::fromPrevStage(2)}, {OutputInfo::fromNetwork(1)});
    testModel.setStageBatchInfo(5, {{0, BatchSupport::Split}});

    RunPass();

    const auto& data0 = singleElement(checkSingleLoopStart(testModel.getInputs().at(0)));
    const auto& data1 = CheckSingleConnection(data0, 0);
    const auto& data2 = CheckSingleConnection(data1, 1);
    const auto& data3 = CheckSingleConnection(data2, 2);

    const auto& branches = checkBranches(data3, {StageType::None, StageType::LoopEnd});
    const auto& branch1 = branches[0];
    const auto& branch2 = branches[1];

    const auto& data4 = CheckSingleConnection(branch1, 3);
    const auto& data7 = singleElement(checkSingleLoopEnd(data4));
    const auto& data5 = CheckSingleConnection(data7, 4, batchSize);
    ASSERT_EQ(data5, testModel.getOutputs().at(0));
    const auto& data6 = CheckSingleConnection(branch2, 5);
    ASSERT_EQ(data6, testModel.getOutputs().at(1));
}

TEST_F(VPU_AdjustDataBatchTest, DISABLED_BranchedWithBatchAndSplitItemsInTheEnd) {
    //
    //                                         -> (Split) -> [Output]
    // [Input] -> (Split) -> (Split) -> (Split)
    //                                         -> (Split) -> [Output]
    //
    const DataDesc desc{16, 16, 3, batchSize};

    testModel.createInputs({desc});
    testModel.createOutputs({desc, desc});

    for (int i = 0; i < 3; i++) {
        if (i > 0)
            testModel.addStage({InputInfo::fromPrevStage(i - 1)}, {OutputInfo::intermediate(desc)});
        else
            testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc)});
        testModel.setStageBatchInfo(i, {{0, BatchSupport::Split}});
    }
    for (int i = 0; i < 2; i++) {
        testModel.addStage({InputInfo::fromNetwork(2)}, {OutputInfo::intermediate(desc)});
        testModel.setStageBatchInfo(3 + i, {{0, BatchSupport::Split}});
    }

    testModel.addStage({InputInfo::fromPrevStage(2)}, {OutputInfo::fromNetwork(0)});
    testModel.setStageBatchInfo(3, {{0, BatchSupport::Split}});
    testModel.addStage({InputInfo::fromPrevStage(2)}, {OutputInfo::fromNetwork(1)});
    testModel.setStageBatchInfo(4, {{0, BatchSupport::Split}});

    RunPass();

    const auto& data1 = singleElement(checkSingleLoopStart(testModel.getInputs().at(0)));
    const auto& data2 = CheckSingleConnection(data1, 1);
    const auto& data3 = CheckSingleConnection(data2, 2);
    const auto& branches = checkBranches(data3, {StageType::None, StageType::LoopEnd});
    const auto& branch1 = branches[0];
    const auto& branch2 = branches[1];
    const auto& data4 = CheckSingleConnection(branch1, 3);
    (void)data4;
    const auto& data5 = CheckSingleConnection(branch2, 4);
    const auto& data6 = checkSingleLoopEnd(data5);
    (void)data6;
}
