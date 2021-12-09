// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

#include "ie_memcpy.h"

namespace vpu {

namespace ie = InferenceEngine;

class ConvertShapeNotationTests : public GraphTransformerTest {
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
        _pipeline.addPass(passManager->dumpModel("before-convert-shape-notation"));
        _pipeline.addPass(passManager->initialCheck());
        _pipeline.addPass(passManager->convertShapeNotation());
        _pipeline.addPass(passManager->dumpModel("after-convert-shape-notation"));
    }

    int getGathersCount() {
        int gathersCount = 0;

        for (const auto& stage : _testModel.getBaseModel()->getStages()) {
            if (stage->type() == StageType::Gather) {
                gathersCount++;
            }
        }

        return gathersCount;
    }

    void checkGatherParams() {
        for (const auto& gather : _testModel.getBaseModel()->getStages()) {
            if (gather->type() != StageType::Gather) {
                continue;
            }

            const auto gatherIndices = gather->input(1);
            ASSERT_NE(gatherIndices, nullptr);

            const auto indicesDesc = gatherIndices->desc();
            ASSERT_EQ(indicesDesc.type(), DataType::S32);
            ASSERT_EQ(indicesDesc.dims().size(), 1);

            const auto indicesCount = static_cast<size_t>(indicesDesc.totalDimSize());
            const auto indicesBuffer = gatherIndices->content()->get<int32_t*>();

            std::vector<int32_t> expectedIndices(indicesCount);
            std::iota(expectedIndices.rbegin(), expectedIndices.rend(), 0);

            std::vector<int32_t> actualIndices(indicesCount);
            ie_memcpy(actualIndices.data(), indicesCount, indicesBuffer, indicesCount);

            ASSERT_TRUE(std::equal(actualIndices.begin(), actualIndices.end(), expectedIndices.begin()));
        }
    }

    Data findConvertedShape(const Data& shape) {
        for (const auto& consumer : shape->consumers()) {
            if (consumer->type() == StageType::Gather) {
                return consumer->output(0);
            }
        }
        return nullptr;
    }

    void checkDataToShapeDependency(const Data& shape, const Data& data) {
        ASSERT_FALSE(shape->childDataToShapeEdges().empty());
        ASSERT_NE(data->parentDataToShapeEdge(), nullptr);

        const auto& edge = data->parentDataToShapeEdge();

        ASSERT_EQ(edge->child(), data);
        ASSERT_EQ(edge->parent(), shape);
    }

    void checkStageDependency(const Stage& parent, const Stage& child) {
        ASSERT_FALSE(parent->childDependencyEdges().empty());
        const auto& childDependencyEdges = parent->childDependencyEdges();

        auto it = std::find_if(childDependencyEdges.begin(), childDependencyEdges.end(),
                               [&child](const StageDependency& edge) {
                                   return edge->child() == child;
                               });

        ASSERT_NE(it, childDependencyEdges.end());
    }

    void checkNoDataToShapeDependency(const Data& shape, const Data& data) {
        ASSERT_TRUE(!data->parentDataToShapeEdge() || data->parentDataToShapeEdge()->parent() != shape);

        const auto& childDataToShapeEdges = shape->childDataToShapeEdges();

        auto it = std::find_if(childDataToShapeEdges.begin(), childDataToShapeEdges.end(),
                               [&data](const DataToShapeAllocation& edge) {
                                   return edge->child() == data;
                               });

        ASSERT_EQ(it, childDataToShapeEdges.end());
    }

    void checkNoStageDependency(const Stage& parent, const Stage& child) {
        const auto& childDependencyEdges = parent->childDependencyEdges();

        auto it = std::find_if(childDependencyEdges.begin(), childDependencyEdges.end(),
                               [&child](const StageDependency& edge) {
                                   return edge->child() == child;
                               });

        ASSERT_EQ(it, parent->childDependencyEdges().end());
    }

    void checkGathers(int gathersCount) {
        ASSERT_EQ(getGathersCount(), gathersCount);
        checkGatherParams();
    }

    void checkShapeWithConsumers(const Data& shape, const DataVector& datasConsumingShape) {
        const auto convertedShape = findConvertedShape(shape);

        for (const auto& dataConsumingShape : datasConsumingShape) {
            checkDataToShapeDependency(convertedShape, dataConsumingShape);
            checkNoDataToShapeDependency(shape, dataConsumingShape);

            Stage dependentStage;
            if (shape->producer() && shape->producer() == dataConsumingShape->producer()) {
                dependentStage = dataConsumingShape->singleConsumer();
            } else {
                ASSERT_NE(dataConsumingShape->producer(), nullptr);
                dependentStage = dataConsumingShape->producer();
            }

            checkStageDependency(convertedShape->producer(), dependentStage);
            checkNoStageDependency(shape->producer(), dependentStage);
        }
    }

protected:
    PassSet _pipeline;
    TestModel _testModel;
};

TEST_F(ConvertShapeNotationTests, BothDataAndShapeIntermediate) {
    //
    //                    -> [Shape] -> (Stage) -> [Output]
    // [Input] -> (Stage)       |
    //                    -> [Data]  -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    auto shapeProducer = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                                          OutputInfo::intermediate(desc)});
    _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->connectDataWithShape(shapeProducer->output(1), shapeProducer->output(0)));

    ASSERT_NO_THROW(Compile());

    checkGathers(1);
    checkShapeWithConsumers(shapeProducer->output(1), {shapeProducer->output(0)});
}

TEST_F(ConvertShapeNotationTests, DataOutputShapeIntermediate) {
    //
    //                    -> [Shape] -> (Stage) -> [Output]
    // [Input] -> (Stage)       |
    //                    -> [Data]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    auto shapeProducer = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork(0),
                                                                          OutputInfo::intermediate(desc)});
    _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->connectDataWithShape(shapeProducer->output(1), shapeProducer->output(0)));

    ASSERT_NO_THROW(Compile());

    checkGathers(1);
    checkShapeWithConsumers(shapeProducer->output(1), {});
}

TEST_F(ConvertShapeNotationTests, DataIntermediateShapeOutput) {
    //
    //                    -> [Shape]
    // [Input] -> (Stage)       |
    //                    -> [Data]  -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    auto shapeProducer = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                                          OutputInfo::fromNetwork(1)});
    _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->connectDataWithShape(shapeProducer->output(1), shapeProducer->output(0)));

    ASSERT_NO_THROW(Compile());

    checkGathers(1);
    checkShapeWithConsumers(shapeProducer->output(1), {shapeProducer->output(0)});
}

TEST_F(ConvertShapeNotationTests, BothDataAndShapeOutput) {
    //
    //                    -> [Shape]
    // [Input] -> (Stage)       |
    //                    -> [Data]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    auto shapeProducer = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork(0),
                                                                          OutputInfo::fromNetwork(1)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->connectDataWithShape(shapeProducer->output(1), shapeProducer->output(0)));

    ASSERT_NO_THROW(Compile());

    checkGathers(1);
    checkShapeWithConsumers(shapeProducer->output(1), {});
}

TEST_F(ConvertShapeNotationTests, TwoShapeConsumersAllInterm) {
    //
    //                    -> [Data]  -> (Stage) -> [Output]
    //                          |
    // [Input] -> (Stage) -> [Shape] -> (Stage) -> [Output]
    //                          |
    //                    -> [Data]  -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc, desc});

    auto shapeProducer = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                                          OutputInfo::intermediate(desc),
                                                                          OutputInfo::intermediate(desc)});
    _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});
    _testModel.addStage({InputInfo::fromPrevStage(0).output(2)}, {OutputInfo::fromNetwork(2)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->connectDataWithShape(shapeProducer->output(1), shapeProducer->output(0)));
    ASSERT_NO_THROW(model->connectDataWithShape(shapeProducer->output(1), shapeProducer->output(2)));

    ASSERT_NO_THROW(Compile());

    checkGathers(1);
    checkShapeWithConsumers(shapeProducer->output(1), {shapeProducer->output(0), shapeProducer->output(2)});
}

TEST_F(ConvertShapeNotationTests, TwoShapeConsumersShapeOutput) {
    //
    //                    -> [Data]  -> (Stage) -> [Output]
    //                          |
    // [Input] -> (Stage) -> [Shape]
    //                          |
    //                    -> [Data]  -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc, desc});

    auto shapeProducer = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::intermediate(desc),
                                                                          OutputInfo::fromNetwork(1),
                                                                          OutputInfo::intermediate(desc)});
    _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});
    _testModel.addStage({InputInfo::fromPrevStage(0).output(2)}, {OutputInfo::fromNetwork(2)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->connectDataWithShape(shapeProducer->output(1), shapeProducer->output(0)));
    ASSERT_NO_THROW(model->connectDataWithShape(shapeProducer->output(1), shapeProducer->output(2)));

    ASSERT_NO_THROW(Compile());

    checkGathers(1);
    checkShapeWithConsumers(shapeProducer->output(1), {shapeProducer->output(0), shapeProducer->output(2)});
}

TEST_F(ConvertShapeNotationTests, DataAndShapeHaveDifferentProducers) {
    //
    //                       [Shape] -> (Stage) -> [Shape]
    // [Input] -> (Stage) ->                          |
    //                       [Data]  -> (Stage) -> [Data]  -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    _testModel.addStage({InputInfo::fromNetwork(0)}, {OutputInfo::intermediate(desc), OutputInfo::intermediate(desc)});
    auto dataProducer = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::intermediate(desc)});
    auto shapeProducer = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});
    _testModel.addStage({InputInfo::fromPrevStage(1)}, {OutputInfo::fromNetwork(0)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->connectDataWithShape(shapeProducer->output(0), dataProducer->output(0)));

    ASSERT_NO_THROW(Compile());

    checkGathers(1);
    checkShapeWithConsumers(shapeProducer->output(0), {dataProducer->output(0)});
}

TEST_F(ConvertShapeNotationTests, TwoConsumersWithSameShape) {
    //
    //                       [Shape]-------------------
    // [Input] -> (Stage) ->    |                     |
    //                       [Data]  -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    auto shapeProducer = _testModel.addStage({InputInfo::fromNetwork(0)}, {OutputInfo::intermediate(desc), OutputInfo::fromNetwork(1)});
    auto dataProcessor = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::fromNetwork(0)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->connectDataWithShape(shapeProducer->output(1), shapeProducer->output(0)));
    ASSERT_NO_THROW(model->connectDataWithShape(shapeProducer->output(1), dataProcessor->output(0)));

    ASSERT_NO_THROW(Compile());

    checkGathers(1);
    checkShapeWithConsumers(shapeProducer->output(1), {shapeProducer->output(0), dataProcessor->output(0)});
}

TEST_F(ConvertShapeNotationTests, ThreeConsumersWithSameShape) {
    //
    //                       [Shape]-----------------------------------------
    // [Input] -> (Stage) ->    |                     |                     |
    //                       [Data]  -> (Stage) -> [Data] -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    auto shapeProducer = _testModel.addStage({InputInfo::fromNetwork(0)}, {OutputInfo::intermediate(desc), OutputInfo::fromNetwork(1)});
    auto firstDataProcessor = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::intermediate(desc)});
    auto secondDataProcessor = _testModel.addStage({InputInfo::fromPrevStage(1).output(0)}, {OutputInfo::fromNetwork(0)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->connectDataWithShape(shapeProducer->output(1), shapeProducer->output(0)));
    ASSERT_NO_THROW(model->connectDataWithShape(shapeProducer->output(1), firstDataProcessor->output(0)));
    ASSERT_NO_THROW(model->connectDataWithShape(shapeProducer->output(1), secondDataProcessor->output(0)));

    ASSERT_NO_THROW(Compile());

    checkGathers(1);
    checkShapeWithConsumers(shapeProducer->output(1), {shapeProducer->output(0), firstDataProcessor->output(0), secondDataProcessor->output(0)});
}

TEST_F(ConvertShapeNotationTests, TwoShapesTwoConsumers) {
    //
    //                       [Shape] -> (Stage) -> [Shape]
    // [Input] -> (Stage) ->    |                     |
    //                       [Data]  -> (Stage) -> [Data]  -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc});

    auto firstShapeProducer = _testModel.addStage({InputInfo::fromNetwork(0)}, {OutputInfo::intermediate(desc), OutputInfo::intermediate(desc)});
    auto firstDataProducer = _testModel.addStage({InputInfo::fromPrevStage(0).output(0)}, {OutputInfo::intermediate(desc)});
    auto secondShapeProducer = _testModel.addStage({InputInfo::fromPrevStage(0).output(1)}, {OutputInfo::fromNetwork(1)});
    auto secondDataProducer = _testModel.addStage({InputInfo::fromPrevStage(1)}, {OutputInfo::fromNetwork(0)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->connectDataWithShape(firstShapeProducer->output(1), firstShapeProducer->output(0)));
    ASSERT_NO_THROW(model->connectDataWithShape(secondShapeProducer->output(0), firstDataProducer->output(0)));

    ASSERT_NO_THROW(Compile());

    checkGathers(2);
    checkShapeWithConsumers(firstShapeProducer->output(1), {firstShapeProducer->output(0)});
    checkShapeWithConsumers(secondShapeProducer->output(0), {firstDataProducer->output(0)});
}

} // namespace vpu
