// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

#include "ie_memcpy.h"

namespace vpu {

namespace ie = InferenceEngine;

class AdjustDynamicBatchTests : public GraphTransformerTest {
protected:
    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(GraphTransformerTest::SetUp());
        ASSERT_NO_FATAL_FAILURE(InitCompileEnv());
        ASSERT_NO_FATAL_FAILURE(InitPipeline());

        _testModel = CreateTestModel();
    }

    void Compile() {
        _pipeline.run(_testModel.getBaseModel());
    }

    void InitPipeline() {
        _pipeline = PassSet();
        _pipeline.addPass(passManager->dumpModel("before-adjust-batch"));
        _pipeline.addPass(passManager->initialCheck());
        _pipeline.addPass(passManager->adjustDataBatch());
        _pipeline.addPass(passManager->dumpModel("after-adjust-batch"));
    }

    static Data singleElement(const DataVector& dataObjects) {
        EXPECT_EQ(dataObjects.size(), 1);
        return dataObjects.front();
    }

    DataVector checkSingleLoopStart(const Data& data) {
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

    static Data checkSingleConnection(const Data& data, int testInd, int batch = 1) {
        EXPECT_EQ(data->numConsumers(), 1);

        const auto& consumer = data->singleConsumer();
        EXPECT_EQ(consumer->type(), StageType::None);
        EXPECT_EQ(consumer->attrs().get<int>("test_ind"), testInd);
        EXPECT_EQ(consumer->numOutputs(), 1);
        const auto& output = consumer->output(0);
        EXPECT_EQ(output->desc().dim(Dim::N), batch);
        return output;
    }

    DataVector checkSingleLoopEnd(const Data& data) {
        EXPECT_EQ(data->numConsumers(), 1);

        const auto& consumer = data->singleConsumer();
        EXPECT_EQ(consumer->type(), StageType::LoopEnd);
        DataVector outputs;
        for (const auto& output : consumer->outputs()) {
            outputs.push_back(output);
        }

        return outputs;
    }

protected:
    PassSet _pipeline;
    TestModel _testModel;
};

TEST_F(AdjustDynamicBatchTests, AdjustBatch_oneStage) {
    //
    // [Input] -> (Split) -> [Output]
    //

    const DataDesc desc_input{3, 3, 3, 3};
    const DataDesc desc_input_shape{4};

    const DataDesc desc_output{3, 3, 3, 3};

    _testModel.createInputs({desc_input, desc_input_shape});
    _testModel.createOutputs({desc_output});

    auto stage = _testModel.addStage({InputInfo::fromNetwork(0)}, {OutputInfo::fromNetwork(0)});
    _testModel.setStageBatchInfo(0, {{0, BatchSupport::Split}});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->connectDataWithShape(_testModel.getInputs()[1], _testModel.getInputs()[0]));
    ASSERT_NO_THROW(model->connectDataWithShape(_testModel.getInputs()[1], _testModel.getOutputs()[0]));

    ASSERT_NO_THROW(Compile());

    const auto& data0 = singleElement(checkSingleLoopStart(_testModel.getInputs()[0]));
    const auto& data1 = checkSingleConnection(data0, 0);
    const auto& data2 = checkSingleLoopEnd(data1);
    ASSERT_EQ(data2, _testModel.getOutputs());
}

TEST_F(AdjustDynamicBatchTests, AdjustBatch) {
    //
    // [Input] -> (Split) -> (Split) -> (Split) -> [Output]
    //

    const DataDesc desc_input{3, 3, 3, 3};
    const DataDesc desc_input_shape{4};

    const DataDesc desc_output{3, 3, 3, 3};

    _testModel.createInputs({desc_input, desc_input_shape});
    _testModel.createOutputs({desc_output});

    auto stage1 = _testModel.addStage({InputInfo::fromNetwork(0)}, {OutputInfo::intermediate(desc_input)});
    _testModel.setStageBatchInfo(0, {{0, BatchSupport::Split}});
    auto stage2 = _testModel.addStage({InputInfo::fromPrevStage(0)}, {OutputInfo::fromNetwork(0)});
    _testModel.setStageBatchInfo(1, {{0, BatchSupport::Split}});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->connectDataWithShape(_testModel.getInputs()[1], _testModel.getInputs()[0]));
    ASSERT_NO_THROW(model->connectDataWithShape(_testModel.getInputs()[1], _testModel.getOutputs()[0]));

    ASSERT_NO_THROW(Compile());

    const auto& data0 = singleElement(checkSingleLoopStart(_testModel.getInputs()[0]));
    const auto& data1 = checkSingleConnection(data0, 0);
    const auto& data2 = checkSingleConnection(data1, 1);
    const auto& data3 = checkSingleLoopEnd(data2);
    ASSERT_EQ(data3, _testModel.getOutputs());
}

} // namespace vpu
