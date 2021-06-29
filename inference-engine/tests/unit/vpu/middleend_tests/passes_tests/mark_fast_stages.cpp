// // Copyright (C) 2018-2021 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

// #include "graph_transformer_tests.hpp"

// namespace vpu {

// namespace ie = InferenceEngine;

// class MarkFastStagesTests : public GraphTransformerTest {
// protected:
//     void SetUp() override {
//         ASSERT_NO_FATAL_FAILURE(GraphTransformerTest::SetUp());
//         ASSERT_NO_FATAL_FAILURE(InitCompileEnv());
//         ASSERT_NO_FATAL_FAILURE(InitPipeline());

//         _testModel = CreateTestModel();
//     }

//     void Compile() {
//         _pipeline.run(_testModel.getBaseModel());
//     }

//     void InitPipeline() {
//         _pipeline = PassSet();
//         _pipeline.addPass(passManager->dumpModel("before-mark-fast-stages"));
//         _pipeline.addPass(passManager->initialCheck());
//         _pipeline.addPass(passManager->markFastStages());
//         _pipeline.addPass(passManager->dumpModel("after-mark-fast-stages"));
//     }

// protected:
//     PassSet _pipeline;
//     TestModel _testModel;
// };

// TEST_F(MarkFastStagesTests, FastStageIsMarked) {
//     const DataDesc desc{1};

//     _testModel.createInputs({desc});
//     _testModel.createOutputs({desc});

//     auto stage = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork()});
//     ASSERT_NO_THROW(Compile());

//     ASSERT_NE(stage->name().find("@fast-stage"), std::string::npos);
// }

// TEST_F(MarkFastStagesTests, SlowStageIsNotMarked) {
//     const DataDesc desc{1000};

//     _testModel.createInputs({desc});
//     _testModel.createOutputs({desc});

//     auto stage = _testModel.addStage({InputInfo::fromNetwork()}, {OutputInfo::fromNetwork()});
//     ASSERT_NO_THROW(Compile());

//     ASSERT_EQ(stage->name().find("@fast-stage"), std::string::npos);
// }

// } // namespace vpu
