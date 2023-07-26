// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/single_layer/loop.hpp"

namespace LayerTestsDefinitions {


TEST_P(LoopTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
}

TEST_P(StaticShapeLoopTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    Run();
}

TEST_P(StaticShapeLoopTest, CompareWithPredefinedRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    LoadNetwork();
    GenerateInputs();
    Infer();
    auto expectedOutputs = PredefinedRefs(); // use predefined refs instead of CalculateRefs function
    const auto& actualOutputs = GetOutputs();

    if (expectedOutputs.empty()) {
        return;
    }

    IE_ASSERT(actualOutputs.size() == expectedOutputs.size())
    << "nGraph interpreter has " << expectedOutputs.size() << " outputs, while IE " << actualOutputs.size();

    Compare(expectedOutputs, actualOutputs);
}

TEST_P(TrivialLoopTest, PassThroughBody) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Precision iePrc;
    InferenceEngine::SizeVector ieShape;
    std::tie(iePrc, ieShape, targetDevice) = GetParam();

    const auto prc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(iePrc);
    const auto shape = ngraph::Shape{ieShape};
    const auto scalarShape = ngraph::Shape{};

    auto start = std::make_shared<ngraph::opset5::Parameter>(prc, shape);
    auto count = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, scalarShape, 5);
    auto icond = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, scalarShape, true);

    // Loop body
    auto b_data = std::make_shared<ngraph::opset5::Parameter>(prc, shape);
    auto b_cond = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::boolean, scalarShape);

    auto body = std::make_shared<ngraph::Function>(
            ngraph::OutputVector    {b_cond, b_data},   // | passthrough body, no data changes
            ngraph::ParameterVector {b_cond, b_data});  // | input -> output

    auto loop = std::make_shared<ngraph::opset5::Loop>(count, icond);
    loop->set_function(body);
    loop->set_special_body_ports({-1, 0});
    loop->set_invariant_input(b_cond, icond);
    loop->set_invariant_input(b_data, start);
    loop->get_iter_value(b_data, -1);

    function = std::make_shared<ngraph::Function>(
            ngraph::OutputVector    {loop},
            ngraph::ParameterVector {start});

    // Precalculated ref blobs
    auto blob = make_blob_with_precision({iePrc, ieShape, InferenceEngine::TensorDesc::getLayoutByDims(ieShape)});
    blob->allocate();
    ov::test::utils::fill_data_with_broadcast(blob, 0, {10});

    inputGens[""] = [&] (InferenceEngine::TensorDesc tdesc) { return blob; };
    outputGens[""] = [&] (InferenceEngine::TensorDesc tdesc) { return blob; };

    Run();
}

TEST_P(TrivialLoopTest, UnusedInputBody) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Precision iePrc;
    InferenceEngine::SizeVector ieShape;
    std::tie(iePrc, ieShape, targetDevice) = GetParam();

    const auto prc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(iePrc);
    const auto shape = ngraph::Shape{ieShape};
    const auto scalarShape = ngraph::Shape{};

    auto start = std::make_shared<ngraph::opset5::Parameter>(prc, shape);
    auto count = std::make_shared<ngraph::opset5::Constant>(ngraph::element::i64, scalarShape, 5);
    auto icond = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, scalarShape, true);

    // Loop body
    auto b_data = std::make_shared<ngraph::opset5::Parameter>(prc, shape);
    auto b_cond = std::make_shared<ngraph::opset5::Constant>(ngraph::element::boolean, scalarShape, true);
    auto b_iter = std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, scalarShape);

    auto body = std::make_shared<ngraph::Function>(
            ngraph::OutputVector    {b_cond, b_data},
            ngraph::ParameterVector {b_data, b_iter});

    auto loop = std::make_shared<ngraph::opset5::Loop>(count, icond);
    loop->set_function(body);
    loop->set_special_body_ports({1, 0});
    loop->set_invariant_input(b_data, start);
    loop->get_iter_value(b_data, -1);

    function = std::make_shared<ngraph::Function>(
            ngraph::OutputVector    {loop},
            ngraph::ParameterVector {start});

    // Precalculated ref blobs
    auto blob = make_blob_with_precision({iePrc, ieShape, InferenceEngine::TensorDesc::getLayoutByDims(ieShape)});
    blob->allocate();
    ov::test::utils::fill_data_with_broadcast(blob, 0, {10});

    inputGens[""] = [&] (InferenceEngine::TensorDesc tdesc) { return blob; };
    outputGens[""] = [&] (InferenceEngine::TensorDesc tdesc) { return blob; };

    Run();
}



TEST_P(TrivialLoopTest, AutoSlicingInput_CheckPredefinedValues) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Precision iePrc;
    InferenceEngine::SizeVector ieShape;
    std::tie(iePrc, ieShape, targetDevice) = GetParam();
    const size_t batch_size = 5;
    const size_t num_iteration = 3;
    ieShape[0] = 1;
    auto ieShape_to_slice = ieShape;
    ieShape_to_slice[0] = batch_size;
    CreateSlicedLoop(batch_size, num_iteration, iePrc, ieShape);
    Run();
    // Precalculated ref blobs
    auto blob = make_blob_with_precision({iePrc, ieShape_to_slice, InferenceEngine::TensorDesc::getLayoutByDims(ieShape_to_slice)});
    blob->allocate();
    std::vector<float> seq_raw_data(batch_size);
    std::iota(seq_raw_data.begin(), seq_raw_data.end(), 1);
    ov::test::utils::fill_data_with_broadcast(blob, 0, seq_raw_data);

    auto blob_ref = make_blob_with_precision({iePrc, ieShape, InferenceEngine::TensorDesc::getLayoutByDims(ieShape)});
    blob_ref->allocate();
    ov::test::utils::fill_data_with_broadcast(blob_ref, 0, { num_iteration * (num_iteration + 1) / 2});

    inputGens[""] = [&] (InferenceEngine::TensorDesc tdesc) { return blob; };
    outputGens[""] = [&] (InferenceEngine::TensorDesc tdesc) { return blob_ref; };
}

TEST_P(TrivialLoopTest, AutoSlicingInputWithDynCondition_CheckPredefinedValues) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Precision iePrc;
    InferenceEngine::SizeVector ieShape;
    std::tie(iePrc, ieShape, targetDevice) = GetParam();

    // auto slicing size : 5
    // trip count limit  : 4
    // dyn exit after iter  : 3
    // ---------------------
    //   should exit after 4 iterations
    const size_t batch_size = 5;
    const size_t trip_count = 5;
    const size_t num_iteration = 3;

    ieShape[0] = 1;
    auto ieShape_to_slice = ieShape;
    ieShape_to_slice[0] = batch_size;

    CreateSlicedLoopDynCondition(batch_size, num_iteration, iePrc, ieShape, trip_count);
    // Precalculated ref blobs
    auto blob = make_blob_with_precision({iePrc, ieShape_to_slice, InferenceEngine::TensorDesc::getLayoutByDims(ieShape_to_slice)});
    blob->allocate();
    std::vector<float> seq_raw_data(batch_size);
    std::iota(seq_raw_data.begin(), seq_raw_data.end(), 1);
    ov::test::utils::fill_data_with_broadcast(blob, 0, seq_raw_data);

    auto blob_ref = make_blob_with_precision({iePrc, ieShape, InferenceEngine::TensorDesc::getLayoutByDims(ieShape)});
    blob_ref->allocate();
    const size_t real_iter = num_iteration + 1;
    ov::test::utils::fill_data_with_broadcast(blob_ref, 0, { real_iter * (real_iter + 1) / 2});

    inputGens[""] = [&] (InferenceEngine::TensorDesc tdesc) { return blob; };
    outputGens[""] = [&] (InferenceEngine::TensorDesc tdesc) { return blob_ref; };

    Run();
}

TEST_P(TrivialLoopTest, AutoSlicingInput_CheckReference) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Precision iePrc;
    InferenceEngine::SizeVector ieShape;
    std::tie(iePrc, ieShape, targetDevice) = GetParam();
    const size_t batch_size = 5;
    const size_t num_iteration = 3;
    ieShape[0] = 1;
    auto ieShape_to_slice = ieShape;
    ieShape_to_slice[0] = batch_size;
    CreateSlicedLoop(batch_size, num_iteration, iePrc, ieShape);
    Run();
}

TEST_P(TrivialLoopTest, AutoSlicingInputWithDynCondition_CheckReference) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    InferenceEngine::Precision iePrc;
    InferenceEngine::SizeVector ieShape;
    std::tie(iePrc, ieShape, targetDevice) = GetParam();

    // auto slicing size : 5
    // trip count limit  : 4
    // dyn exit after iter  : 3
    // ---------------------
    //   should exit after 4 iterations
    const size_t batch_size = 5;
    const size_t trip_count = 5;
    const size_t num_iteration = 3;

    ieShape[0] = 1;
    auto ieShape_to_slice = ieShape;
    ieShape_to_slice[0] = batch_size;

    CreateSlicedLoopDynCondition(batch_size, num_iteration, iePrc, ieShape, trip_count);
    Run();
}

}  // namespace LayerTestsDefinitions
