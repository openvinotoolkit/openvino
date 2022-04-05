// Copyright (C) 2018-2022 Intel Corporation
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

    // Find shape producer
    auto it = std::find_if(model->getStages().begin(), model->getStages().end(), [](const Stage& stage) {
        return stage->numOutputs() == 2;
    });

    ASSERT_NE(it, model->getStages().end());

    auto shapeProducer = *it;

    const auto& data = shapeProducer->output(0);
    ASSERT_NE(data->parentDataToShapeEdge(), nullptr);
    const auto& edge = data->parentDataToShapeEdge();
    ASSERT_NE(edge->parent(), nullptr);

    const auto& shapeDataLocation = edge->parent()->dataLocation();

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

    // Find shape producer
    auto it = std::find_if(model->getStages().begin(), model->getStages().end(), [](const Stage& stage) {
        return stage->numOutputs() == 2;
    });

    ASSERT_NE(it, model->getStages().end());

    auto shapeProducer = *it;

    const auto& data = shapeProducer->output(0);
    ASSERT_NE(data->parentDataToShapeEdge(), nullptr);
    auto edge = data->parentDataToShapeEdge();
    ASSERT_NE(edge->parent(), nullptr);

    const auto& shapeDataLocation = edge->parent()->dataLocation();
    const auto& dataShapeLocation = data->shapeLocation();

    ASSERT_EQ(shapeDataLocation.location, dataShapeLocation.dimsLocation);
    ASSERT_EQ(shapeDataLocation.offset, dataShapeLocation.dimsOffset);

    const auto& processedData = data->singleConsumer()->output(0);
    ASSERT_NE(processedData->parentDataToShapeEdge(), nullptr);
    edge = processedData->parentDataToShapeEdge();
    ASSERT_NE(edge->parent(), nullptr);

    const auto& processedShapeDataLocation = edge->parent()->dataLocation();
    const auto& processedDataShapeLocation = processedData->shapeLocation();

    ASSERT_EQ(processedShapeDataLocation.location, processedDataShapeLocation.dimsLocation);
    ASSERT_EQ(processedShapeDataLocation.offset, processedDataShapeLocation.dimsOffset);
}

TEST_F(DataToShapeEdgeProcessingTests, ShapeProcessingOnceHasCorrectExecutionOrder) {
    setupNetWithShapeBeingProcessedOnce();

    const auto& model = _testModel.getBaseModel();

    ASSERT_NO_THROW(_middleEnd->run(model));

    // Find shape producer
    auto it = std::find_if(model->getStages().begin(), model->getStages().end(), [](const Stage& stage) {
        return stage->numOutputs() == 2;
    });

    ASSERT_NE(it, model->getStages().end());

    auto shapeProducer = *it;

    const auto dataProcessor = shapeProducer->output(0)->singleConsumer();
    ASSERT_NE(shapeProducer->output(0)->parentDataToShapeEdge(), nullptr);
    const auto& edge = shapeProducer->output(0)->parentDataToShapeEdge();
    ASSERT_NE(edge->parent(), nullptr);

    ASSERT_TRUE(checkExecutionOrder(model, {edge->parent()->producer()->id(), dataProcessor->id()}));
}

TEST_F(DataToShapeEdgeProcessingTests, ShapeProcessingTwiceDoesntThrow) {
    setupNetWithShapeBeingProcessedTwice();

    ASSERT_NO_THROW(_middleEnd->run(_testModel.getBaseModel()));
}

TEST_F(DataToShapeEdgeProcessingTests, ShapeProcessingTwiceSharesMemory) {
    setupNetWithShapeBeingProcessedTwice();

    const auto& model = _testModel.getBaseModel();

    ASSERT_NO_THROW(_middleEnd->run(model));

    // Find shape producer
    auto it = std::find_if(model->getStages().begin(), model->getStages().end(), [](const Stage& stage) {
        return stage->numOutputs() == 2;
    });

    ASSERT_NE(it, model->getStages().end());

    auto shapeProducer = *it;

    const auto& data = shapeProducer->output(0);
    ASSERT_NE(data->parentDataToShapeEdge(), nullptr);
    auto edge = data->parentDataToShapeEdge();
    ASSERT_NE(edge->parent(), nullptr);

    const auto& shapeDataLocation = edge->parent()->dataLocation();
    const auto& dataShapeLocation = data->shapeLocation();

    ASSERT_EQ(shapeDataLocation.location, dataShapeLocation.dimsLocation);
    ASSERT_EQ(shapeDataLocation.offset, dataShapeLocation.dimsOffset);

    const auto& dataProcessedOnce = data->singleConsumer()->output(0);
    ASSERT_NE(dataProcessedOnce->parentDataToShapeEdge(), nullptr);
    edge = dataProcessedOnce->parentDataToShapeEdge();
    ASSERT_NE(edge->parent(), nullptr);

    auto processedShapeDataLocation = edge->parent()->dataLocation();
    auto processedDataShapeLocation = dataProcessedOnce->shapeLocation();

    ASSERT_EQ(processedShapeDataLocation.location, processedDataShapeLocation.dimsLocation);
    ASSERT_EQ(processedShapeDataLocation.offset, processedDataShapeLocation.dimsOffset);

    const auto dataProcessedTwice = dataProcessedOnce->singleConsumer()->output(0);
    ASSERT_NE(dataProcessedTwice->parentDataToShapeEdge(), nullptr);
    edge = dataProcessedTwice->parentDataToShapeEdge();
    ASSERT_NE(edge->parent(), nullptr);

    processedShapeDataLocation = edge->parent()->dataLocation();
    processedDataShapeLocation = dataProcessedTwice->shapeLocation();

    ASSERT_EQ(processedShapeDataLocation.location, processedDataShapeLocation.dimsLocation);
    ASSERT_EQ(processedShapeDataLocation.offset, processedDataShapeLocation.dimsOffset);
}

TEST_F(DataToShapeEdgeProcessingTests, ShapeProcessingTwiceHasCorrectExecutionOrder) {
    setupNetWithShapeBeingProcessedTwice();

    const auto& model = _testModel.getBaseModel();

    ASSERT_NO_THROW(_middleEnd->run(model));

    // Find shape producer
    auto it = std::find_if(model->getStages().begin(), model->getStages().end(), [](const Stage& stage) {
        return stage->numOutputs() == 2;
    });

    ASSERT_NE(it, model->getStages().end());

    auto shapeProducer = *it;

    const auto dataFirstProcessor = shapeProducer->output(0)->singleConsumer();
    ASSERT_NE(shapeProducer->output(0)->parentDataToShapeEdge(), nullptr);
    auto edge = shapeProducer->output(0)->parentDataToShapeEdge();
    ASSERT_NE(edge->parent(), nullptr);

    ASSERT_TRUE(checkExecutionOrder(model, {edge->parent()->producer()->id(), dataFirstProcessor->id()}));

    const auto dataSecondProcessor = dataFirstProcessor->output(0)->singleConsumer();
    ASSERT_NE(dataSecondProcessor->input(0)->parentDataToShapeEdge(), nullptr);
    edge = dataSecondProcessor->input(0)->parentDataToShapeEdge();
    ASSERT_NE(edge->parent(), nullptr);

    ASSERT_TRUE(checkExecutionOrder(model, {edge->parent()->producer()->id(), dataSecondProcessor->id()}));
}

TEST_F(DataToShapeEdgeProcessingTests, ReplaceDataToShapeParentReplacesConnections) {
    //
    //                    -> [Shape] -> (ShapeProc) -> [Shape]
    //                                                    |
    // [Input] -> (Stage) -> [Data]  -> (DataProc)  -> [Data]
    //                                                    |
    //                    -> [Shape] -> (ShapeProc) -> [Shape]
    //

    const auto& desc = DataDesc({1});

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc, desc});

    const auto dataAndShapeParent = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                                               OutputInfo::intermediate(desc),
                                                                               OutputInfo::intermediate(desc)});
    const auto firstShapeProcessor = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    const auto dataProcessor = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});
    const auto secondShapeProcessor = _testModel.addStage({InputInfo::fromPrevStage(0).output(2)}, {OutputInfo::fromNetwork(2)});

    const auto& model = _testModel.getBaseModel();
    const auto& processedData = dataProcessor->output(0);
    const auto& initialShape = firstShapeProcessor->output(0);
    const auto& finalShape = secondShapeProcessor->output(0);
    ASSERT_NO_THROW(model->connectDataWithShape(initialShape, processedData));

    const auto& dataToShapeEdge = processedData->parentDataToShapeEdge();
    ASSERT_NE(dataToShapeEdge, nullptr);

    ASSERT_NO_THROW(model->replaceDataToShapeParent(dataToShapeEdge, finalShape));

    ASSERT_TRUE(initialShape->childDataToShapeEdges().empty());
    ASSERT_EQ(finalShape->childDataToShapeEdges().front(), dataToShapeEdge);
    ASSERT_EQ(dataToShapeEdge->parent(), finalShape);

    ASSERT_TRUE(firstShapeProcessor->childDependencyEdges().empty());
    ASSERT_FALSE(secondShapeProcessor->childDependencyEdges().empty());

    const auto& stageDependencyEdge = secondShapeProcessor->childDependencyEdges().front();
    ASSERT_EQ(stageDependencyEdge->child(), dataProcessor);
    ASSERT_EQ(stageDependencyEdge->parent(), secondShapeProcessor);
}

TEST_F(DataToShapeEdgeProcessingTests, ReplaceDataToShapeChildReplacesConnections) {
    //
    //                    -> [Data]  -> (DataProc)  -> [Data]
    //                                                    |
    // [Input] -> (Stage) -> [Shape] -> (ShapeProc) -> [Shape]
    //                                                    |
    //                    -> [Data]  -> (DataProc)  -> [Data]
    //

    const auto& desc = DataDesc({1});

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc, desc});

    const auto dataAndShapeParent = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                                                     OutputInfo::intermediate(desc),
                                                                                     OutputInfo::intermediate(desc)});
    const auto firstDataProcessor = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    const auto shapeProcessor = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});
    const auto secondDataProcessor = _testModel.addStage({InputInfo::fromPrevStage(0).output(2)}, {OutputInfo::fromNetwork(2)});

    const auto& model = _testModel.getBaseModel();
    const auto& processedShape = shapeProcessor->output(0);
    const auto& initialData = firstDataProcessor->output(0);
    const auto& finalData = secondDataProcessor->output(0);
    ASSERT_NO_THROW(model->connectDataWithShape(processedShape, initialData));

    const auto& dataToShapeEdge = processedShape->childDataToShapeEdges().front();
    ASSERT_NE(dataToShapeEdge, nullptr);

    ASSERT_NO_THROW(model->replaceDataToShapeChild(dataToShapeEdge, finalData));

    ASSERT_EQ(initialData->parentDataToShapeEdge(), nullptr);
    ASSERT_EQ(finalData->parentDataToShapeEdge(), dataToShapeEdge);
    ASSERT_EQ(dataToShapeEdge->child(), finalData);

    ASSERT_FALSE(shapeProcessor->childDependencyEdges().empty());
    const auto& stageDependencyEdge = shapeProcessor->childDependencyEdges().front();

    ASSERT_EQ(stageDependencyEdge->child(), secondDataProcessor);
    ASSERT_EQ(stageDependencyEdge->parent(), shapeProcessor);
}

TEST_F(DataToShapeEdgeProcessingTests, DisconnectDatasRemovesConnections) {
    //
    //                    -> [Shape] -> (ShapeProc) -> [Shape]
    // [Input] -> (Stage)                                 |
    //                    -> [Data]  -> (DataProc)  -> [Data]
    //

    const auto& desc = DataDesc({1});

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    const auto dataAndShapeParent = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                                                     OutputInfo::intermediate(desc)});
    const auto dataProcessor = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    const auto shapeProcessor = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});

    const auto& model = _testModel.getBaseModel();
    const auto& processedData = dataProcessor->output(0);
    const auto& processedShape = shapeProcessor->output(0);
    ASSERT_NO_THROW(model->connectDataWithShape(processedShape, processedData));

    const auto& dataToShapeEdge = processedShape->childDataToShapeEdges().front();
    ASSERT_NE(dataToShapeEdge, nullptr);

    ASSERT_NO_THROW(model->disconnectDatas(dataToShapeEdge));

    ASSERT_EQ(processedData->parentDataToShapeEdge(), nullptr);
    ASSERT_TRUE(processedShape->childDataToShapeEdges().empty());
    ASSERT_TRUE(shapeProcessor->childDependencyEdges().empty());
}

} // namespace vpu
