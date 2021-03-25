// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer_tests.hpp"

namespace vpu {

namespace ie = InferenceEngine;

class InjectStageTests : public GraphTransformerTest {
protected:
    void SetUp() override {
        ASSERT_NO_FATAL_FAILURE(GraphTransformerTest::SetUp());

        ASSERT_NO_FATAL_FAILURE(InitCompileEnv());

        _testModel = CreateTestModel();
    }

protected:
    TestModel _testModel;
};

TEST_F(InjectStageTests, InjectionRedirectsChildStageDependency) {
    //
    //         -> (Stage) -> [Output]
    // [Input] -> (Stage) -> [Output]
    //         -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc, desc});

    const auto hwStage = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork(0)}, StageType::MyriadXHwOp);
    const auto swStage = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork(1)}, StageType::Copy);
    const auto childStage = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork(2)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->addStageDependency(swStage, childStage));
    ASSERT_TRUE(checkExecutionOrder(model, {swStage->id(), childStage->id()}));

    ASSERT_NO_THROW(model->injectStage()
                            .parentHW(hwStage)
                            .childSW(swStage)
                            .done());
    ASSERT_TRUE(checkExecutionOrder(model, {hwStage->id(), childStage->id()}));
}

TEST_F(InjectStageTests, InjectionRedirectsParentStageDependency) {
    //
    //         -> (Stage) -> [Output]
    // [Input] -> (Stage) -> [Output]
    //         -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc, desc});

    const auto hwStage = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork(0)}, StageType::MyriadXHwOp);
    const auto swStage = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork(1)}, StageType::Copy);
    const auto parentStage = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork(2)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->addStageDependency(parentStage, swStage));
    ASSERT_TRUE(checkExecutionOrder(model, {parentStage->id(), swStage->id()}));

    ASSERT_NO_THROW(model->injectStage()
                            .parentHW(hwStage)
                            .childSW(swStage)
                            .done());
    ASSERT_TRUE(checkExecutionOrder(model, {parentStage->id(), hwStage->id()}));
}

TEST_F(InjectStageTests, RevertInjectionRedirectsChildStageDependency) {
    //
    //         -> (Stage) -> [Output]
    // [Input] -> (Stage) -> [Output]
    //         -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc, desc});

    const auto hwStage = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork(0)}, StageType::MyriadXHwOp);
    const auto swStage = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork(1)}, StageType::Copy);
    const auto parentStage = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork(2)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->addStageDependency(parentStage, swStage));
    ASSERT_TRUE(checkExecutionOrder(model, {parentStage->id(), swStage->id()}));

    Injection edge;
    ASSERT_NO_THROW(edge = model->injectStage()
                                .parentHW(hwStage)
                                .childSW(swStage)
                                .done());
    ASSERT_NO_THROW(model->revertInjection(edge));
    ASSERT_TRUE(checkExecutionOrder(model, {parentStage->id(), swStage->id()}));
}

TEST_F(InjectStageTests, RevertInjectionRedirectsParentStageDependency) {
    //
    //         -> (Stage) -> [Output]
    // [Input] -> (Stage) -> [Output]
    //         -> (Stage) -> [Output]
    //

    const DataDesc desc{1};

    _testModel.createInputs({desc});
    _testModel.createOutputs({desc, desc, desc});

    const auto hwStage = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork(0)}, StageType::MyriadXHwOp);
    const auto swStage = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork(1)}, StageType::Copy);
    const auto parentStage = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork(2)});

    auto model = _testModel.getBaseModel();

    ASSERT_NO_THROW(model->addStageDependency(parentStage, swStage));
    ASSERT_TRUE(checkExecutionOrder(model, {parentStage->id(), swStage->id()}));

    Injection edge;
    ASSERT_NO_THROW(edge = model->injectStage()
                            .parentHW(hwStage)
                            .childSW(swStage)
                            .done());
    ASSERT_NO_THROW(model->revertInjection(edge));
    ASSERT_TRUE(checkExecutionOrder(model, {parentStage->id(), swStage->id()}));
}

} // namespace vpu
