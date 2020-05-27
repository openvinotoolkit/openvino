// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

namespace vpu {

namespace ie = InferenceEngine;

class DataToShapeEdgeProcessingTests : public GraphTransformerTest {
protected:
    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(GraphTransformerTest::SetUp());

        ASSERT_NO_FATAL_FAILURE(InitCompileEnv());

        _middleEnd = passManager->buildMiddleEnd();
        _testModel = CreateTestModel();
    }

    void setupNetWithNonProcessingShape() {
        //
        //                       -> [Shape]
        // [Input] -> (ShapeProd)      |
        //                       -> [Data] -> (Stage) -> [Output]
        //

        const auto& dataDesc = DataDesc({800});
        const auto& shapeDesc = DataDesc({1});

        _testModel.createInputs({dataDesc});
        _testModel.createOutputs({dataDesc});

        auto shapeParent = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(dataDesc),
                                                                            OutputInfo::intermediate(shapeDesc)});
        _testModel.addStage({InputInfo::fromPrevStage(0)}, {OutputInfo::fromNetwork()});

        auto model = _testModel.getBaseModel();
        model->connectDataWithShape(shapeParent->output(1), shapeParent->output(0));
    }

    void setupNetWithShapeBeingProcessedOnce() {
        //
        //                       -> [Shape] -> (ShapeProc) -> [Shape]
        // [Input] -> (ShapeProd)      |                         |
        //                       -> [Data]  -> (DataProc)  -> [Data] -> (Stage) -> [Output]
        //

        const auto& dataDesc = DataDesc({800});
        const auto& shapeDesc = DataDesc({1});

        _testModel.createInputs({dataDesc});
        _testModel.createOutputs({dataDesc});

        auto model = _testModel.getBaseModel();

        auto dataAndShapeParent = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(dataDesc),
                                                                            OutputInfo::intermediate(shapeDesc)});
        model->connectDataWithShape(dataAndShapeParent->output(1), dataAndShapeParent->output(0));

        auto dataChild = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::intermediate(dataDesc)});
        auto shapeChild = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::intermediate(shapeDesc)});
        _testModel.addStage({InputInfo::fromPrevStage(1)}, {OutputInfo::fromNetwork()});

        model->connectDataWithShape(shapeChild->output(0), dataChild->output(0));
    }

    void setupNetWithShapeBeingProcessedTwice() {
        //
        //                       -> [Shape] -> (ShapeProc) -> [Shape] -> (ShapeProc) -> [Shape]
        // [Input] -> (ShapeProd)      |                         |                         |
        //                       -> [Data]  -> (DataProc)  -> [Data]  -> (DataProc)  -> [Data] -> (Stage) -> [Output]
        //

        const auto& dataDesc = DataDesc({800});
        const auto& shapeDesc = DataDesc({1});

        _testModel.createInputs({dataDesc});
        _testModel.createOutputs({dataDesc});

        auto model = _testModel.getBaseModel();

        auto dataAndShapeParent = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(dataDesc),
                                                                                   OutputInfo::intermediate(shapeDesc)});
        model->connectDataWithShape(dataAndShapeParent->output(1), dataAndShapeParent->output(0));

        auto dataChild = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::intermediate(dataDesc)});
        auto shapeChild = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::intermediate(shapeDesc)});
        model->connectDataWithShape(shapeChild->output(0), dataChild->output(0));

        dataChild = _testModel.addStage({InputInfo::fromPrevStage(1).output(0)}, {OutputInfo::intermediate(dataDesc)});
        shapeChild = _testModel.addStage({InputInfo::fromPrevStage(2).output(0)}, {OutputInfo::intermediate(shapeDesc)});
        model->connectDataWithShape(shapeChild->output(0), dataChild->output(0));

        _testModel.addStage({InputInfo::fromPrevStage(3)}, {OutputInfo::fromNetwork()});
    }

protected:
    TestModel _testModel;
    PassSet::Ptr _middleEnd = nullptr;
};

TEST_F(DataToShapeEdgeProcessingTests, ShapeDataWithoutConsumerDoesntThrow) {
    setupNetWithNonProcessingShape();

    ASSERT_NO_THROW(_middleEnd->run(_testModel.getBaseModel()));
}

TEST_F(DataToShapeEdgeProcessingTests, DataToShapeEdgeSharesMemory) {
    setupNetWithNonProcessingShape();

    const auto& model = _testModel.getBaseModel();

    ASSERT_NO_THROW(_middleEnd->run(model));

    Stage shapeProducer = nullptr;
    for (const auto& stage : model->getStages()) {
        // Find shape produced stage
        if (stage->numOutputs() == 2) {
            shapeProducer = stage;
        }
    }

    ASSERT_NE(shapeProducer, nullptr);

    const auto& data = shapeProducer->output(0);
    const auto& shape = shapeProducer->output(1);

    const auto& shapeDataLocation = shape->dataLocation();
    const auto& dataShapeLocation = data->shapeLocation();

    ASSERT_EQ(shapeDataLocation.location, dataShapeLocation.dimsLocation);
    ASSERT_EQ(shapeDataLocation.offset, dataShapeLocation.dimsOffset);
}

TEST_F(DataToShapeEdgeProcessingTests, ShapeProcessingOnceDoesntThrow) {
    setupNetWithShapeBeingProcessedOnce();

    ASSERT_NO_THROW(_middleEnd->run(_testModel.getBaseModel()));
}

TEST_F(DataToShapeEdgeProcessingTests, ShapeProcessingOnceSharesMemory) {
    setupNetWithShapeBeingProcessedOnce();

    const auto& model = _testModel.getBaseModel();

    ASSERT_NO_THROW(_middleEnd->run(model));

    Stage shapeProducer = nullptr;
    for (const auto& stage : model->getStages()) {
        // Find shape produced stage
        if (stage->numOutputs() == 2) {
            shapeProducer = stage;
        }
    }

    ASSERT_NE(shapeProducer, nullptr);

    const auto& data = shapeProducer->output(0);
    const auto& shape = shapeProducer->output(1);

    const auto& shapeDataLocation = shape->dataLocation();
    const auto& dataShapeLocation = data->shapeLocation();

    ASSERT_EQ(shapeDataLocation.location, dataShapeLocation.dimsLocation);
    ASSERT_EQ(shapeDataLocation.offset, dataShapeLocation.dimsOffset);

    const auto& processedData = data->singleConsumer()->output(0);
    const auto& processedShape = shape->singleConsumer()->output(0);

    const auto& processedShapeDataLocation = processedShape->dataLocation();
    const auto& processedDataShapeLocation = processedData->shapeLocation();

    ASSERT_EQ(processedShapeDataLocation.location, processedDataShapeLocation.dimsLocation);
    ASSERT_EQ(processedShapeDataLocation.offset, processedDataShapeLocation.dimsOffset);
}

TEST_F(DataToShapeEdgeProcessingTests, DISABLED_ShapeProcessingOnceHasCorrectExecutionOrder) {
    setupNetWithShapeBeingProcessedOnce();

    const auto& model = _testModel.getBaseModel();

    ASSERT_NO_THROW(_middleEnd->run(model));

    Stage shapeProducer = nullptr;
    for (const auto& stage : model->getStages()) {
        // Find shape produced stage
        if (stage->numOutputs() == 2) {
            shapeProducer = stage;
        }
    }

    ASSERT_NE(shapeProducer, nullptr);

    const auto dataProcessor = shapeProducer->output(0)->singleConsumer();
    const auto shapeProcessor = shapeProducer->output(1)->singleConsumer();

    ASSERT_TRUE(checkExecutionOrder(model, {shapeProcessor->id(), dataProcessor->id()}));
}

TEST_F(DataToShapeEdgeProcessingTests, ShapeProcessingTwiceDoesntThrow) {
    setupNetWithShapeBeingProcessedTwice();

    ASSERT_NO_THROW(_middleEnd->run(_testModel.getBaseModel()));
}

TEST_F(DataToShapeEdgeProcessingTests, ShapeProcessingTwiceSharesMemory) {
    setupNetWithShapeBeingProcessedTwice();

    const auto& model = _testModel.getBaseModel();

    ASSERT_NO_THROW(_middleEnd->run(model));

    Stage shapeProducer = nullptr;
    for (const auto& stage : model->getStages()) {
        // Find shape produced stage
        if (stage->numOutputs() == 2) {
            shapeProducer = stage;
        }
    }

    ASSERT_NE(shapeProducer, nullptr);

    const auto& data = shapeProducer->output(0);
    const auto& shape = shapeProducer->output(1);

    const auto& shapeDataLocation = shape->dataLocation();
    const auto& dataShapeLocation = data->shapeLocation();

    ASSERT_EQ(shapeDataLocation.location, dataShapeLocation.dimsLocation);
    ASSERT_EQ(shapeDataLocation.offset, dataShapeLocation.dimsOffset);

    const auto& dataProcessedOnce = data->singleConsumer()->output(0);
    const auto& shapeProcessedOnce = shape->singleConsumer()->output(0);

    auto processedShapeDataLocation = shapeProcessedOnce->dataLocation();
    auto processedDataShapeLocation = dataProcessedOnce->shapeLocation();

    ASSERT_EQ(processedShapeDataLocation.location, processedDataShapeLocation.dimsLocation);
    ASSERT_EQ(processedShapeDataLocation.offset, processedDataShapeLocation.dimsOffset);

    const auto dataProcessedTwice = dataProcessedOnce->singleConsumer()->output(0);
    const auto shapeProcessedTwice = shapeProcessedOnce->singleConsumer()->output(0);

    processedShapeDataLocation = shapeProcessedTwice->dataLocation();
    processedDataShapeLocation = dataProcessedTwice->shapeLocation();

    ASSERT_EQ(processedShapeDataLocation.location, processedDataShapeLocation.dimsLocation);
    ASSERT_EQ(processedShapeDataLocation.offset, processedDataShapeLocation.dimsOffset);
}

TEST_F(DataToShapeEdgeProcessingTests, DISABLED_ShapeProcessingTwiceHasCorrectExecutionOrder) {
    setupNetWithShapeBeingProcessedTwice();

    const auto& model = _testModel.getBaseModel();

    ASSERT_NO_THROW(_middleEnd->run(model));

    Stage shapeProducer = nullptr;
    for (const auto& stage : model->getStages()) {
        // Find shape produced stage
        if (stage->numOutputs() == 2) {
            shapeProducer = stage;
        }
    }

    ASSERT_NE(shapeProducer, nullptr);

    const auto dataFirstProcessor = shapeProducer->output(0)->singleConsumer();
    const auto shapeFirstProcessor = shapeProducer->output(1)->singleConsumer();

    ASSERT_TRUE(checkExecutionOrder(model, {shapeFirstProcessor->id(), dataFirstProcessor->id()}));

    const auto dataSecondProcessor = dataFirstProcessor->output(0)->singleConsumer();
    const auto shapeSecondProcessor = shapeFirstProcessor->output(0)->singleConsumer();

    ASSERT_TRUE(checkExecutionOrder(model, {shapeSecondProcessor->id(), dataSecondProcessor->id()}));
}

} // namespace vpu
