// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/test_enums.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/runtime/properties.hpp"

namespace {
using ov::test::InputShape;

struct GatherNDShapeParams {
    InputShape inputShapes;
    InputShape targetShapes;
    int batch_dims;
};

typedef std::tuple<
        GatherNDShapeParams,
        ov::element::Type,     // Model type
        bool                   // Is const Indices
> GatherNDGPUTestParams;


class GatherNDGPUTest : public testing::WithParamInterface<GatherNDGPUTestParams>,
                        virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherNDGPUTestParams>& obj) {
        const auto& [Shapes, model_type, isIndicesConstant] = obj.param;

        std::ostringstream result;
        result << "IS=(";
        result << ov::test::utils::partialShape2str({Shapes.inputShapes.first}) << "_";
        for (size_t i = 0lu; i < Shapes.inputShapes.second.size(); i++) {
            result << "{";
            result << ov::test::utils::vec2str(Shapes.inputShapes.second[i]) << "_";
            result << "}_";
        }
        result << "TS=(";
        result << ov::test::utils::partialShape2str({Shapes.targetShapes.first}) << "_";
        for (size_t i = 0lu; i < Shapes.targetShapes.second.size(); i++) {
            result << "{";
            result << ov::test::utils::vec2str(Shapes.targetShapes.second[i]) << "_";
            result << "}_";
        }
        result << "batchDims=" << Shapes.batch_dims << "_";
        result << "netPrc=" << model_type << "_";
        result << "constIdx=" << (isIndicesConstant ? "True" : "False") << "_";

        return result.str();
    }

protected:
    void SetUp() override {
        const auto intInputsPrecision = ov::element::i32;

        const auto& [Shapes, model_type, isIndicesConstant] = this->GetParam();
        const int batchDims = Shapes.batch_dims;
        targetDevice = ov::test::utils::DEVICE_GPU;
        std::shared_ptr<ov::Node> indicesNode;
        std::shared_ptr<ov::Node> gather_ndNode;

        if (isIndicesConstant) {
            init_input_shapes({Shapes.inputShapes});
        } else { // Not being tested because currently parameter targetshape is not supported
            init_input_shapes({Shapes.inputShapes, Shapes.targetShapes});
        }

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0])};
        params.back()->set_friendly_name("data");

        if (isIndicesConstant) {
            int64_t idx_range = INT64_MAX;
            for (size_t i = 0; i < Shapes.inputShapes.second.size(); ++i) {
                for (size_t j = 0; j < Shapes.inputShapes.second[i].size(); ++j) {
                    idx_range = std::min(static_cast<int64_t>(Shapes.inputShapes.second[i][j]), idx_range);
                }
            }
            auto indices_tensor = ov::test::utils::create_and_fill_tensor(ov::element::i64, Shapes.targetShapes.second[0], idx_range - 1, 0);
            indicesNode = std::make_shared<ov::op::v0::Constant>(indices_tensor);
        } else {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(intInputsPrecision, inputDynamicShapes[1]));
            params.back()->set_friendly_name("indices");
        }

        gather_ndNode = std::make_shared<ov::op::v8::GatherND>(params[0],
                                                          isIndicesConstant ? indicesNode : params[1],
                                                          batchDims);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(gather_ndNode)};
        function = std::make_shared<ov::Model>(results, params, "GatherND");
    }
};

TEST_P(GatherNDGPUTest, Inference) {
    run();
}

const std::vector<ov::element::Type> model_types = {
    ov::element::f32,
    ov::element::f16,
    ov::element::i32
};

const std::vector<GatherNDShapeParams> dynamicInputShapeConstTargetShape = {
    {
        ov::test::InputShape(ov::PartialShape({-1, -1}), {{2, 2}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 1}}),
        1
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1}), {{10, 14}}),
        ov::test::InputShape(ov::PartialShape({}), {{3, 2}}),
        0
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1}), {{2, 3, 4}, {3, 4, 5}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 1}}),
        0
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1}), {{2, 7, 8, 9}, {2, 7, 4, 8}}),
        ov::test::InputShape(ov::PartialShape({}), {{2, 1}}),
        1
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1}), {{15, 12, 20, 15, 2}, {15, 12, 18, 7, 17}}),
        ov::test::InputShape(ov::PartialShape({}), {{15, 12, 5, 9, 1, 3}}),
        2
    },
    {
        ov::test::InputShape(ov::PartialShape({-1, -1, -1, -1, -1, -1}), {{4, 3, 2, 5, 5, 2}, {4, 3, 2, 5, 7, 2}}),
        ov::test::InputShape(ov::PartialShape({}), {{4, 3, 2, 5, 6, 2}}),
        4
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_dynamic_input_shapes_const_target_shapes, GatherNDGPUTest,
                ::testing::Combine(
                    ::testing::ValuesIn(dynamicInputShapeConstTargetShape),    // input shapes
                    ::testing::ValuesIn(model_types),          // network precision
                    ::testing::Values(true)),                     // is const indices
                GatherNDGPUTest::getTestCaseName);

using GatherNDRuntimePrecisionParams = std::tuple<ov::element::Type, bool, bool>;

class GatherNDVisionRuntimePrecisionTest : public testing::WithParamInterface<GatherNDRuntimePrecisionParams>,
                                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherNDRuntimePrecisionParams>& info) {
        const auto& [precision, model_like, dynamic_shapes] = info.param;
        return std::string(precision == ov::element::bf16 ? "BF16Runtime" : "F16Runtime") +
               (model_like ? "_GatherNDMultiplyModelLike" : "_GatherNDAlone") +
               (dynamic_shapes ? "_Dynamic" : "_Static");
    }

protected:
    void SetUp() override {
        const auto& [precision, model_like, dynamic_shapes] = GetParam();
        targetDevice = ov::test::utils::DEVICE_GPU;
        configuration = {ov::hint::inference_precision(precision)};
        abs_threshold = 0.02f;
        rel_threshold = 0.02f;
        if (dynamic_shapes) {
            init_input_shapes({InputShape{{-1, -1, 1152}, {{2, 3, 1152}, {3, 5, 1152}}},
                               InputShape{{-1, 2}, {{4, 2}, {7, 2}}}});
        } else {
            init_input_shapes({InputShape{{2, 3, 1152}, {{2, 3, 1152}}},
                               InputShape{{4, 2}, {{4, 2}}}});
        }
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[1]);
        data->set_friendly_name("data");
        indices->set_friendly_name("indices");
        std::shared_ptr<ov::Node> gather_data = data;
        if (model_like) {
            auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1}, {0.75f});
            gather_data = std::make_shared<ov::op::v1::Multiply>(data, scale);
        }
        std::shared_ptr<ov::Node> output = std::make_shared<ov::op::v8::GatherND>(gather_data, indices, 0);
        output->set_friendly_name("vision_tower/GatherND");
        if (model_like) {
            std::vector<float> weights(1152);
            for (size_t channel = 0; channel < weights.size(); ++channel) {
                weights[channel] = 0.5f + static_cast<float>(channel % 13) / 16.0f;
            }
            auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1152}, weights);
            output = std::make_shared<ov::op::v1::Multiply>(output, scale);
        }
        function = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{data, indices},
                                               "GatherNDVisionRuntimePrecision");
    }

    void generate_inputs(const std::vector<ov::Shape>& shapes) override {
        inputs.clear();
        ov::Tensor data(ov::element::f32, shapes[0]);
        for (size_t index = 0; index < data.get_size(); ++index) {
            data.data<float>()[index] = static_cast<float>(index % 37) / 32.0f - 0.5f;
        }
        ov::Tensor indices(ov::element::i64, shapes[1]);
        for (size_t row = 0; row < shapes[1][0]; ++row) {
            indices.data<int64_t>()[2 * row] = static_cast<int64_t>(row % shapes[0][0]);
            indices.data<int64_t>()[2 * row + 1] = static_cast<int64_t>((2 * row + 1) % shapes[0][1]);
        }
        inputs.emplace(function->get_parameters()[0], data);
        inputs.emplace(function->get_parameters()[1], indices);
    }
};

TEST_P(GatherNDVisionRuntimePrecisionTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_Gemma4Vision, GatherNDVisionRuntimePrecisionTest,
                        ::testing::Combine(::testing::Values(ov::element::f16, ov::element::bf16),
                                           ::testing::Values(false, true),
                                           ::testing::Values(false, true)),
                        GatherNDVisionRuntimePrecisionTest::getTestCaseName);

using GatherNDVisionPostOpParams = std::tuple<bool, bool>;

class GatherNDVisionBF16PostOpTest : public testing::WithParamInterface<GatherNDVisionPostOpParams>,
                                    virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<GatherNDVisionPostOpParams>& info) {
        const auto& [quantize, dynamic_shapes] = info.param;
        return std::string("BF16Runtime_") + (quantize ? "GatherNDQuantize" : "GatherNDActivation") +
               (dynamic_shapes ? "_Dynamic" : "_Static");
    }

protected:
    void SetUp() override {
        const auto& [quantize, dynamic_shapes] = GetParam();
        targetDevice = ov::test::utils::DEVICE_GPU;
        configuration = {ov::hint::inference_precision(ov::element::bf16)};
        abs_threshold = 0.02f;
        rel_threshold = 0.02f;
        if (dynamic_shapes) {
            init_input_shapes({InputShape{{-1, -1, 1152}, {{2, 3, 1152}, {3, 5, 1152}}},
                               InputShape{{-1, 2}, {{4, 2}, {7, 2}}}});
        } else {
            init_input_shapes({InputShape{{2, 3, 1152}, {{2, 3, 1152}}},
                               InputShape{{4, 2}, {{4, 2}}}});
        }
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[1]);
        auto gathered = std::make_shared<ov::op::v8::GatherND>(data, indices, 0);
        std::shared_ptr<ov::Node> output;
        if (quantize) {
            output = std::make_shared<ov::op::v0::FakeQuantize>(
                gathered,
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f}),
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f}),
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f}),
                ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f}),
                256);
        } else {
            output = std::make_shared<ov::op::v0::Relu>(gathered);
        }
        output->set_friendly_name("vision_tower/GatherND_post_op");
        function = std::make_shared<ov::Model>(ov::OutputVector{output}, ov::ParameterVector{data, indices},
                                               "GatherNDVisionBF16PostOp");
    }

    void generate_inputs(const std::vector<ov::Shape>& shapes) override {
        inputs.clear();
        ov::Tensor data(ov::element::f32, shapes[0]);
        for (size_t index = 0; index < data.get_size(); ++index) {
            data.data<float>()[index] = static_cast<float>(index % 37) / 32.0f - 0.5f;
        }
        ov::Tensor indices(ov::element::i64, shapes[1]);
        for (size_t row = 0; row < shapes[1][0]; ++row) {
            indices.data<int64_t>()[2 * row] = static_cast<int64_t>(row % shapes[0][0]);
            indices.data<int64_t>()[2 * row + 1] = static_cast<int64_t>((2 * row + 1) % shapes[0][1]);
        }
        inputs.emplace(function->get_parameters()[0], data);
        inputs.emplace(function->get_parameters()[1], indices);
    }
};

TEST_P(GatherNDVisionBF16PostOpTest, Inference) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_Gemma4VisionPostOp, GatherNDVisionBF16PostOpTest,
                        ::testing::Combine(::testing::Values(false, true),
                                           ::testing::Values(false, true)),
                        GatherNDVisionBF16PostOpTest::getTestCaseName);
} // namespace
