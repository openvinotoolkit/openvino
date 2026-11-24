// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <common_test_utils/test_assertions.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "common/utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "shared_test_classes/base/ov_behavior_test_utils.hpp"

namespace ov {
namespace test {
namespace behavior {

namespace {

enum class HighRankOp { Relu, Transpose };

std::string toString(HighRankOp op) {
    return op == HighRankOp::Relu ? "Relu" : "Transpose";
}

// Moves the first dimension to the end: the shape stays easy to tell apart from its input
std::vector<int64_t> rotatePermutation(size_t rank) {
    std::vector<int64_t> permutation(rank);
    for (size_t i = 0; i < rank; ++i) {
        permutation[i] = static_cast<int64_t>((i + 1) % rank);
    }
    return permutation;
}

std::shared_ptr<ov::Model> createModel(HighRankOp op, const ov::Shape& shape) {
    auto parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, shape);
    parameter->set_friendly_name("input");
    parameter->get_output_tensor(0).set_names({"input"});

    std::shared_ptr<ov::Node> node;
    if (op == HighRankOp::Relu) {
        node = std::make_shared<ov::op::v0::Relu>(parameter);
    } else {
        const auto permutation = rotatePermutation(shape.size());
        node = std::make_shared<ov::op::v1::Transpose>(
            parameter,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{permutation.size()}, permutation));
    }

    auto result = std::make_shared<ov::op::v0::Result>(node);
    result->set_friendly_name("output");
    result->get_output_tensor(0).set_names({"output"});

    return std::make_shared<ov::Model>(ov::ResultVector{result}, ov::ParameterVector{parameter}, "high_rank");
}

ov::Shape expectedOutputShape(HighRankOp op, const ov::Shape& inputShape) {
    if (op == HighRankOp::Relu) {
        return inputShape;
    }

    const auto permutation = rotatePermutation(inputShape.size());
    ov::Shape outputShape(inputShape.size());
    for (size_t i = 0; i < outputShape.size(); ++i) {
        outputShape[i] = inputShape[permutation[i]];
    }
    return outputShape;
}

ov::Tensor referenceRelu(const ov::Tensor& input) {
    ov::Tensor output(input.get_element_type(), input.get_shape());
    std::transform(input.data<const float>(),
                   input.data<const float>() + input.get_size(),
                   output.data<float>(),
                   [](float value) {
                       return std::max(value, 0.f);
                   });
    return output;
}

ov::Tensor referenceTranspose(const ov::Tensor& input) {
    const auto& inputShape = input.get_shape();
    const size_t rank = inputShape.size();
    const auto permutation = rotatePermutation(rank);

    ov::Tensor output(input.get_element_type(), expectedOutputShape(HighRankOp::Transpose, inputShape));
    const auto& outputShape = output.get_shape();

    std::vector<size_t> inputStrides(rank, 1);
    for (size_t i = rank - 1; i > 0; --i) {
        inputStrides[i - 1] = inputStrides[i] * inputShape[i];
    }

    const auto* source = input.data<const float>();
    auto* destination = output.data<float>();
    std::vector<size_t> coordinate(rank, 0);
    for (size_t index = 0; index < output.get_size(); ++index) {
        size_t sourceIndex = 0;
        for (size_t i = 0; i < rank; ++i) {
            sourceIndex += coordinate[i] * inputStrides[permutation[i]];
        }
        destination[index] = source[sourceIndex];

        for (size_t i = rank; i-- > 0;) {
            if (++coordinate[i] < outputShape[i]) {
                break;
            }
            coordinate[i] = 0;
        }
    }
    return output;
}

ov::Tensor reference(HighRankOp op, const ov::Tensor& input) {
    return op == HighRankOp::Relu ? referenceRelu(input) : referenceTranspose(input);
}

// The values of the input are negative as well, so that Relu has something to cut
ov::Tensor createInput(const ov::Shape& shape, int seed) {
    constexpr uint32_t range = 2;
    constexpr double startFrom = -1;
    constexpr int32_t resolution = 100;
    return ov::test::utils::create_and_fill_tensor(ov::element::f32, shape, range, startFrom, resolution, seed);
}

// All the dimensions differ: a mistaken order of the dimensions or a mistaken rank cannot go unnoticed
const ov::Shape kInputShape{1, 2, 3, 4, 5, 6};

const ov::AnyMap kConfig = {ov::intel_npu::compiler_type(ov::intel_npu::CompilerType::PLUGIN),
                            {"NPU_COMPILATION_MODE", "DefaultHW"}};

// The NPU computes in lower precision than the f32 reference
constexpr double kTolerance = 0.01;

}  // namespace

using HighRankParams = std::tuple<std::string,  // Device name
                                  HighRankOp    // Operation
                                  >;

// Compile and infer six-dimensional models through the plugin compiler's DefaultHW path.
class OVInferRequestHighRankTests : public testing::WithParamInterface<HighRankParams>, public OVInferRequestTestBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<HighRankParams>& obj) {
        std::string targetDevice;
        HighRankOp op;
        std::tie(targetDevice, op) = obj.param;
        std::replace(targetDevice.begin(), targetDevice.end(), ':', '.');

        return "targetDevice=" + targetDevice + "_op=" + toString(op);
    }

    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();

        std::tie(target_device, op) = this->GetParam();
        APIBaseTest::SetUp();
    }

protected:
    std::shared_ptr<ov::Core> core = utils::PluginCache::get().core();
    HighRankOp op = HighRankOp::Relu;
};

TEST_P(OVInferRequestHighRankTests, CompiledModelKeepsTheShapes) {
    ov::CompiledModel compiledModel;
    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(createModel(op, kInputShape), target_device, kConfig));

    EXPECT_EQ(compiledModel.input().get_shape(), kInputShape);
    EXPECT_EQ(compiledModel.output().get_shape(), expectedOutputShape(op, kInputShape));
}

// Binding another tensor of the same rank goes through the update of the arguments of the recorded command lists
TEST_P(OVInferRequestHighRankTests, InferMatchesReferenceForSuccessiveInputTensors) {
    ov::CompiledModel compiledModel;
    OV_ASSERT_NO_THROW(compiledModel = core->compile_model(createModel(op, kInputShape), target_device, kConfig));
    ov::InferRequest request;
    OV_ASSERT_NO_THROW(request = compiledModel.create_infer_request());

    for (const int seed : {1, 2}) {
        const auto input = createInput(kInputShape, seed);
        OV_ASSERT_NO_THROW(request.set_tensor(compiledModel.input(), input));
        OV_ASSERT_NO_THROW(request.infer());

        const auto output = request.get_tensor(compiledModel.output());
        EXPECT_EQ(output.get_shape(), expectedOutputShape(op, kInputShape));
        ov::test::utils::compare(reference(op, input), output, kTolerance, kTolerance);
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVInferRequestHighRankTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_NPU),
                                            ::testing::Values(HighRankOp::Relu, HighRankOp::Transpose)),
                         ov::test::utils::appendPlatformTypeTestName<OVInferRequestHighRankTests>);

}  // namespace behavior
}  // namespace test
}  // namespace ov
