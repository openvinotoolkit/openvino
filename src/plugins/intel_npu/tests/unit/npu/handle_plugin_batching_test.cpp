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
#include "openvino/op/non_zero.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
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
