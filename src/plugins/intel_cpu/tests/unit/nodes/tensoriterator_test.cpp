// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>

#include "graph.h"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"

using namespace ov::intel_cpu;

TEST(TensorIteratorMemoryTest, PreservesOutputHandleWhenDynamicLoopShapeIsUnchanged) {
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 32});
    const auto target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto trip_count = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    const auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(2));
    const auto body_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto reshaped = std::make_shared<ov::op::v1::Reshape>(state, body_shape, false);
    const auto updated =
        std::make_shared<ov::op::v1::Add>(reshaped, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.f}));
    const auto condition = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {true});
    const auto body =
        std::make_shared<ov::Model>(ov::OutputVector{condition, updated}, ov::ParameterVector{state, body_shape});
    const auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, condition);
    loop->set_function(body);
    loop->set_special_body_ports({-1, 0});
    loop->set_merged_input(state, data, updated);
    loop->set_invariant_input(body_shape, target_shape);
    const auto output = loop->get_iter_value(updated, -1);
    const std::shared_ptr<const ov::Model> model =
        std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{data, target_shape, trip_count});

    Config config;
    config.inferencePrecision = ov::element::f32;
    const auto context = std::make_shared<GraphContext>(config, std::make_shared<WeightsSharing>(), false);
    Graph graph;
    graph.CreateGraph(model, context);
    const auto input = graph.getInputNodeByIndex(0)->getChildEdgeAt(0)->getMemoryPtr();
    const auto shape_input = graph.getInputNodeByIndex(1)->getChildEdgeAt(0)->getMemoryPtr();
    const auto count_input = graph.getInputNodeByIndex(2)->getChildEdgeAt(0)->getMemoryPtr();
    const auto result = graph.getOutputNodeByIndex(0)->getParentEdgeAt(0)->getMemoryPtr();

    dnnl::memory previous_handle;
    ov::Shape previous_shape;
    const std::array<ov::Shape, 5> shapes{{{1, 32}, {1, 32}, {2, 16}, {2, 16}, {1, 32}}};
    for (size_t run = 0; run < shapes.size(); ++run) {
        SCOPED_TRACE(run);
        const auto iterations = static_cast<int32_t>(run + 1);
        for (size_t index = 0; index < 32; ++index) {
            input->getDataAs<float>()[index] = static_cast<float>(index + run);
        }
        std::copy(shapes[run].begin(), shapes[run].end(), shape_input->getDataAs<int32_t>());
        count_input->getDataAs<int32_t>()[0] = iterations;
        graph.Infer();

        ASSERT_EQ(result->getStaticDims(), shapes[run]);
        for (size_t index = 0; index < 32; ++index) {
            EXPECT_EQ(result->getDataAs<float>()[index], static_cast<float>(index + run + iterations));
        }
        const auto handle = result->getPrimitive();
        if (previous_handle && previous_shape == shapes[run]) {
            EXPECT_EQ(handle.get(), previous_handle.get());
        }
        EXPECT_EQ(handle.get_data_handle(), result->getData());
        previous_handle = handle;
        previous_shape = shapes[run];
    }
}

namespace {

class TensorIteratorMemoryReleaseTest : public testing::TestWithParam<std::tuple<bool, int32_t>> {};

TEST_P(TensorIteratorMemoryReleaseTest, RestoresDynamicIntermediateOutputAfterRelease) {
    const auto [concatenate, iterations] = GetParam();
    const auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 32});
    const auto target_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto trip_count = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1});
    const auto state = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(2));
    const auto body_shape = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{2});
    const auto reshaped = std::make_shared<ov::op::v1::Reshape>(state, body_shape, false);
    const auto updated =
        std::make_shared<ov::op::v1::Add>(reshaped, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.f}));
    const auto condition = ov::op::v0::Constant::create(ov::element::boolean, ov::Shape{1}, {true});
    const auto body =
        std::make_shared<ov::Model>(ov::OutputVector{condition, updated}, ov::ParameterVector{state, body_shape});
    const auto loop = std::make_shared<ov::op::v5::Loop>(trip_count, condition);
    loop->set_friendly_name("loop");
    loop->set_function(body);
    loop->set_special_body_ports({-1, 0});
    loop->set_merged_input(state, data, updated);
    loop->set_invariant_input(body_shape, target_shape);
    const auto output =
        concatenate ? loop->get_concatenated_slices(updated, 0, 1, 1, -1, 0) : loop->get_iter_value(updated, -1);
    // Model I/O buffers are retained on release; a consumer makes the Loop output an intermediate buffer.
    const auto reduced =
        std::make_shared<ov::op::v1::ReduceSum>(output,
                                                ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
                                                false);
    const std::shared_ptr<const ov::Model> model =
        std::make_shared<ov::Model>(ov::OutputVector{reduced}, ov::ParameterVector{data, target_shape, trip_count});

    Config config;
    config.inferencePrecision = ov::element::f32;
    const auto context = std::make_shared<GraphContext>(config, std::make_shared<WeightsSharing>(), false);
    Graph graph;
    graph.CreateGraph(model, context);
    const auto input = graph.getInputNodeByIndex(0)->getChildEdgeAt(0)->getMemoryPtr();
    const auto shape_input = graph.getInputNodeByIndex(1)->getChildEdgeAt(0)->getMemoryPtr();
    const auto count_input = graph.getInputNodeByIndex(2)->getChildEdgeAt(0)->getMemoryPtr();
    const auto result = graph.getOutputNodeByIndex(0)->getParentEdgeAt(0)->getMemoryPtr();
    MemoryPtr loop_output;
    for (const auto& node : graph.GetNodes()) {
        if (node->getName() == "loop") {
            loop_output = node->getChildEdgeAt(0)->getMemoryPtr();
            break;
        }
    }
    ASSERT_NE(loop_output, nullptr);

    dnnl::memory previous_handle;
    ov::Shape previous_shape;
    const std::array<ov::Shape, 5> shapes{{{1, 32}, {1, 32}, {2, 16}, {2, 16}, {1, 32}}};
    for (size_t run = 0; run < shapes.size(); ++run) {
        SCOPED_TRACE(run);
        const auto& shape = iterations == 0 || concatenate ? shapes.front() : shapes[run];
        for (size_t index = 0; index < 32; ++index) {
            input->getDataAs<float>()[index] = static_cast<float>(index + 100 * run);
        }
        std::copy(shape.begin(), shape.end(), shape_input->getDataAs<int32_t>());
        count_input->getDataAs<int32_t>()[0] = iterations;
        graph.Infer();

        const auto rows = shape[0];
        const auto columns = shape[1];
        const auto slices = concatenate ? static_cast<size_t>(iterations) : 1;
        ASSERT_EQ(result->getStaticDims(), ov::Shape{rows * slices});
        for (size_t slice = 0; slice < slices; ++slice) {
            const auto increments = concatenate ? slice + 1 : static_cast<size_t>(iterations);
            for (size_t row = 0; row < rows; ++row) {
                float expected = 0.f;
                for (size_t column = 0; column < columns; ++column) {
                    expected += static_cast<float>(row * columns + column + 100 * run + increments);
                }
                EXPECT_EQ(result->getDataAs<float>()[slice * rows + row], expected);
            }
        }
        const auto handle = loop_output->getPrimitive();
        const auto output_shape = loop_output->getStaticDims();
        if (previous_handle && previous_shape == output_shape) {
            EXPECT_EQ(handle.get(), previous_handle.get());
        }
        EXPECT_EQ(handle.get_data_handle(), loop_output->getData());
        previous_handle = handle;
        previous_shape = output_shape;

        // CompiledModel::release_memory() delegates to this same graph-context operation.
        context->releaseMemory();
        ASSERT_EQ(loop_output->getMemoryBlock()->getRawPtr(), nullptr);
    }
}

INSTANTIATE_TEST_SUITE_P(LoopOutputs,
                         TensorIteratorMemoryReleaseTest,
                         testing::Values(std::make_tuple(false, 0),
                                         std::make_tuple(false, 3),
                                         std::make_tuple(true, 3)));

}  // namespace
