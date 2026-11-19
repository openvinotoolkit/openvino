// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <vector>

#include "common_test_utils/test_assertions.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "openvino/core/layout.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "transformations.hpp"

using namespace intel_npu;

namespace {

// The tensors of the models used here intentionally have no names: the plugin batching must not depend on them.

std::shared_ptr<ov::Model> make_batched_relu_model(const ov::PartialShape& shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    param->set_layout("N...");
    auto relu = std::make_shared<ov::op::v0::Relu>(param);
    auto result = std::make_shared<ov::op::v0::Result>(relu);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "batched_relu");
}

// The output is the shape of the NonZero result, so its shape is always [2] no matter what the batch is. When the batch
// is 2 as well, the output looks like a batched one, although it does not depend on the batch.
std::shared_ptr<ov::Model> make_not_batched_output_model(const ov::Shape& shape) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    param->set_layout("N...");
    auto nonZero = std::make_shared<ov::op::v3::NonZero>(param);
    auto shapeOf = std::make_shared<ov::op::v3::ShapeOf>(nonZero);
    auto result = std::make_shared<ov::op::v0::Result>(shapeOf);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "not_batched_output");
}

// The input [4] with the layout "N" is reshaped to [1] when debatching, so the output of Squeeze, which removes all the
// dimensions equal to 1 when no axes are given, becomes a scalar, while it was [4] before.
std::shared_ptr<ov::Model> make_scalar_after_debatching_model(size_t batch) {
    auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{batch});
    param->set_layout("N");
    auto squeeze = std::make_shared<ov::op::v0::Squeeze>(param);
    auto result = std::make_shared<ov::op::v0::Result>(squeeze);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{param}, "scalar_after_debatching");
}

// Both inputs have the batch in their leading dimension, but only the first one has the batch layout, so only this one
// is reshaped when debatching. The second input keeps the batch of 4, while the output becomes [1, 4].
std::shared_ptr<ov::Model> make_mixed_layout_model() {
    auto paramA = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 3});
    paramA->set_layout("N...");
    auto paramB = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 3});
    auto matMul = std::make_shared<ov::op::v0::MatMul>(paramA, paramB, false, true);
    auto result = std::make_shared<ov::op::v0::Result>(matMul);
    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{paramA, paramB}, "mixed_layout");
}

// Both inputs are reshaped when debatching, but the output of the MatMul with the transposed second input depends on
// the batch in both of its dimensions: it is [4, 4] in the given model and [1, 1] after debatching.
std::shared_ptr<ov::Model> make_two_layouts_matmul_model() {
    auto paramA = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 3});
    paramA->set_layout("N...");
    auto paramB = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{4, 3});
    paramB->set_layout("N...");
    auto matMul = std::make_shared<ov::op::v0::MatMul>(paramA, paramB, false, true);
    auto result = std::make_shared<ov::op::v0::Result>(matMul);
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::ParameterVector{paramA, paramB},
                                       "two_layouts_matmul");
}

// Same as above, but with a bounded dynamic batch: the second input and the output keep the bounds of the batch
// ([1..10, 3]) after debatching, as only the first input is reshaped.
std::shared_ptr<ov::Model> make_dynamic_mixed_layout_model() {
    const ov::PartialShape shape{ov::Dimension(1, 10), 3};
    auto paramA = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    paramA->set_layout("N...");
    auto paramB = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    auto add = std::make_shared<ov::op::v1::Add>(paramA, paramB);
    auto result = std::make_shared<ov::op::v0::Result>(add);
    return std::make_shared<ov::Model>(ov::ResultVector{result},
                                       ov::ParameterVector{paramA, paramB},
                                       "dynamic_mixed_layout");
}

class HandlePluginBatchingTest : public ::testing::Test {
protected:
    std::tuple<std::shared_ptr<ov::Model>, bool> handle(const std::shared_ptr<const ov::Model>& model,
                                                        std::optional<ov::intel_npu::BatchMode> batchMode) {
        return batch_helpers::handlePluginBatching(
            model,
            [this](ov::intel_npu::BatchMode mode) {
                updatedBatchModes.push_back(mode);
            },
            batchMode,
            originalBatch,
            Logger::global());
    }

    std::vector<ov::intel_npu::BatchMode> updatedBatchModes;
    std::optional<ov::Dimension> originalBatch;
};

}  // namespace

TEST_F(HandlePluginBatchingTest, StaticBatchIsReshapedToOne) {
    const auto model = make_batched_relu_model({4, 3, 2, 2});

    auto [resultModel, successfullyDebatched] = handle(model, ov::intel_npu::BatchMode::AUTO);

    ASSERT_TRUE(successfullyDebatched);
    ASSERT_TRUE(originalBatch.has_value());
    ASSERT_EQ(originalBatch.value(), ov::Dimension(4));
    ASSERT_EQ(resultModel->input(0).get_partial_shape(), ov::PartialShape({1, 3, 2, 2}));
    ASSERT_EQ(resultModel->output(0).get_partial_shape(), ov::PartialShape({1, 3, 2, 2}));
    // The given model must not be modified
    ASSERT_EQ(model->input(0).get_partial_shape(), ov::PartialShape({4, 3, 2, 2}));
    ASSERT_EQ(updatedBatchModes, std::vector<ov::intel_npu::BatchMode>{ov::intel_npu::BatchMode::COMPILER});
}

TEST_F(HandlePluginBatchingTest, DynamicBoundedBatchIsReshapedToOne) {
    const auto model = make_batched_relu_model({ov::Dimension(1, 10), 3, 2, 2});

    auto [resultModel, successfullyDebatched] = handle(model, ov::intel_npu::BatchMode::AUTO);

    ASSERT_TRUE(successfullyDebatched);
    ASSERT_TRUE(originalBatch.has_value());
    ASSERT_TRUE(originalBatch->is_dynamic());
    ASSERT_EQ(resultModel->input(0).get_partial_shape(), ov::PartialShape({1, 3, 2, 2}));
    ASSERT_EQ(resultModel->output(0).get_partial_shape(), ov::PartialShape({1, 3, 2, 2}));
}

/**
 * @brief The output of this model is [2] before and after reshaping the input batch, which is 2 as well. Therefore, it
 * passes the validation of the batch, but it is not batched, so the model cannot be handled by the plugin.
 */
TEST_F(HandlePluginBatchingTest, NotBatchedOutputFallsBackToCompilerInAutoMode) {
    const auto model = make_not_batched_output_model({2, 16, 16, 16});

    auto [resultModel, successfullyDebatched] = handle(model, ov::intel_npu::BatchMode::AUTO);

    ASSERT_FALSE(successfullyDebatched);
    ASSERT_FALSE(originalBatch.has_value());
    // The original model is returned, with the original batch
    ASSERT_EQ(resultModel, model);
    ASSERT_EQ(resultModel->input(0).get_partial_shape(), ov::PartialShape({2, 16, 16, 16}));
    ASSERT_EQ(updatedBatchModes, std::vector<ov::intel_npu::BatchMode>{ov::intel_npu::BatchMode::COMPILER});
}

TEST_F(HandlePluginBatchingTest, NotBatchedOutputThrowsInPluginMode) {
    const auto model = make_not_batched_output_model({2, 16, 16, 16});

    OV_EXPECT_THROW_HAS_SUBSTRING(handle(model, ov::intel_npu::BatchMode::PLUGIN), ov::Exception, "PLUGIN batch mode");
    // The given model must not be modified
    ASSERT_EQ(model->input(0).get_partial_shape(), ov::PartialShape({2, 16, 16, 16}));
}

/**
 * @brief The output has the shape [4] in the given model, but it becomes a scalar once the input is reshaped to [1].
 * A scalar has no batch dimension, so debatching must be reported as failed.
 */
TEST_F(HandlePluginBatchingTest, OutputBecomingScalarFallsBackToCompilerInAutoMode) {
    const auto model = make_scalar_after_debatching_model(4);

    auto [resultModel, successfullyDebatched] = handle(model, ov::intel_npu::BatchMode::AUTO);

    ASSERT_FALSE(successfullyDebatched);
    ASSERT_FALSE(originalBatch.has_value());
    ASSERT_EQ(resultModel, model);
    ASSERT_EQ(resultModel->input(0).get_partial_shape(), ov::PartialShape({4}));
    ASSERT_EQ(resultModel->output(0).get_partial_shape(), ov::PartialShape({4}));
    ASSERT_EQ(updatedBatchModes, std::vector<ov::intel_npu::BatchMode>{ov::intel_npu::BatchMode::COMPILER});
}

TEST_F(HandlePluginBatchingTest, OutputBecomingScalarThrowsInPluginMode) {
    const auto model = make_scalar_after_debatching_model(4);

    OV_EXPECT_THROW_HAS_SUBSTRING(handle(model, ov::intel_npu::BatchMode::PLUGIN), ov::Exception, "PLUGIN batch mode");
    // The given model must not be modified
    ASSERT_EQ(model->input(0).get_partial_shape(), ov::PartialShape({4}));
    ASSERT_EQ(model->output(0).get_partial_shape(), ov::PartialShape({4}));
}

/**
 * @brief Only the inputs having the batch layout are reshaped when debatching, although the model validation accepts
 * the inputs by their leading dimension. The output looks debatched ([1, 4]), but one input keeps the batch of 4.
 */
TEST_F(HandlePluginBatchingTest, InputWithoutBatchLayoutFallsBackToCompilerInAutoMode) {
    const auto model = make_mixed_layout_model();

    auto [resultModel, successfullyDebatched] = handle(model, ov::intel_npu::BatchMode::AUTO);

    ASSERT_FALSE(successfullyDebatched);
    ASSERT_FALSE(originalBatch.has_value());
    ASSERT_EQ(resultModel, model);
    ASSERT_EQ(resultModel->input(0).get_partial_shape(), ov::PartialShape({4, 3}));
    ASSERT_EQ(resultModel->input(1).get_partial_shape(), ov::PartialShape({4, 3}));
    ASSERT_EQ(updatedBatchModes, std::vector<ov::intel_npu::BatchMode>{ov::intel_npu::BatchMode::COMPILER});
}

TEST_F(HandlePluginBatchingTest, InputWithoutBatchLayoutThrowsInPluginMode) {
    const auto model = make_mixed_layout_model();

    OV_EXPECT_THROW_HAS_SUBSTRING(handle(model, ov::intel_npu::BatchMode::PLUGIN), ov::Exception, "PLUGIN batch mode");
    // The given model must not be modified
    ASSERT_EQ(model->input(0).get_partial_shape(), ov::PartialShape({4, 3}));
    ASSERT_EQ(model->input(1).get_partial_shape(), ov::PartialShape({4, 3}));
}

/**
 * @brief The output of the model is [1, 1] after debatching, so its leading dimension is 1, but its other dimension
 * changed as well: the results of the slices cannot be stacked to obtain the original [4, 4] output.
 */
TEST_F(HandlePluginBatchingTest, OutputWithChangedNonBatchDimensionFallsBackToCompilerInAutoMode) {
    const auto model = make_two_layouts_matmul_model();

    auto [resultModel, successfullyDebatched] = handle(model, ov::intel_npu::BatchMode::AUTO);

    ASSERT_FALSE(successfullyDebatched);
    ASSERT_FALSE(originalBatch.has_value());
    ASSERT_EQ(resultModel, model);
    ASSERT_EQ(resultModel->input(0).get_partial_shape(), ov::PartialShape({4, 3}));
    ASSERT_EQ(resultModel->input(1).get_partial_shape(), ov::PartialShape({4, 3}));
    ASSERT_EQ(resultModel->output(0).get_partial_shape(), ov::PartialShape({4, 4}));
    ASSERT_EQ(updatedBatchModes, std::vector<ov::intel_npu::BatchMode>{ov::intel_npu::BatchMode::COMPILER});
}

TEST_F(HandlePluginBatchingTest, OutputWithChangedNonBatchDimensionThrowsInPluginMode) {
    const auto model = make_two_layouts_matmul_model();

    OV_EXPECT_THROW_HAS_SUBSTRING(handle(model, ov::intel_npu::BatchMode::PLUGIN), ov::Exception, "PLUGIN batch mode");
    // The given model must not be modified
    ASSERT_EQ(model->input(0).get_partial_shape(), ov::PartialShape({4, 3}));
    ASSERT_EQ(model->input(1).get_partial_shape(), ov::PartialShape({4, 3}));
    ASSERT_EQ(model->output(0).get_partial_shape(), ov::PartialShape({4, 4}));
}

/**
 * @brief The batch dimension of the input without the batch layout, and of the output, remains dynamic after reshaping.
 * It must not be mistaken for a batch of 1.
 */
TEST_F(HandlePluginBatchingTest, DynamicInputWithoutBatchLayoutFallsBackToCompilerInAutoMode) {
    const auto model = make_dynamic_mixed_layout_model();

    auto [resultModel, successfullyDebatched] = handle(model, ov::intel_npu::BatchMode::AUTO);

    ASSERT_FALSE(successfullyDebatched);
    ASSERT_FALSE(originalBatch.has_value());
    ASSERT_EQ(resultModel, model);
    const ov::PartialShape expectedShape{ov::Dimension(1, 10), 3};
    ASSERT_EQ(resultModel->input(0).get_partial_shape(), expectedShape);
    ASSERT_EQ(resultModel->input(1).get_partial_shape(), expectedShape);
    ASSERT_EQ(updatedBatchModes, std::vector<ov::intel_npu::BatchMode>{ov::intel_npu::BatchMode::COMPILER});
}

TEST_F(HandlePluginBatchingTest, DynamicInputWithoutBatchLayoutThrowsInPluginMode) {
    const auto model = make_dynamic_mixed_layout_model();

    OV_EXPECT_THROW_HAS_SUBSTRING(handle(model, ov::intel_npu::BatchMode::PLUGIN), ov::Exception, "PLUGIN batch mode");
    // The given model must not be modified
    const ov::PartialShape expectedShape{ov::Dimension(1, 10), 3};
    ASSERT_EQ(model->input(0).get_partial_shape(), expectedShape);
    ASSERT_EQ(model->input(1).get_partial_shape(), expectedShape);
}

/**
 * @brief The batch mode is not available when the compiler is too old to support it, thus the batch mode cannot be
 * updated either
 */
TEST_F(HandlePluginBatchingTest, NotBatchedOutputWithoutBatchModeKeepsOriginalModel) {
    const auto model = make_not_batched_output_model({2, 16, 16, 16});

    auto [resultModel, successfullyDebatched] = handle(model, std::nullopt);

    ASSERT_FALSE(successfullyDebatched);
    ASSERT_FALSE(originalBatch.has_value());
    ASSERT_EQ(resultModel, model);
    ASSERT_TRUE(updatedBatchModes.empty());
}

TEST_F(HandlePluginBatchingTest, CompilerBatchModeDoesNotChangeModel) {
    const auto model = make_batched_relu_model({4, 3, 2, 2});

    auto [resultModel, successfullyDebatched] = handle(model, ov::intel_npu::BatchMode::COMPILER);

    ASSERT_FALSE(successfullyDebatched);
    ASSERT_FALSE(originalBatch.has_value());
    ASSERT_EQ(resultModel, model);
    ASSERT_TRUE(updatedBatchModes.empty());
}
