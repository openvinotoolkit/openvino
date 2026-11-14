// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <sstream>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/precision_support.h"

namespace ov::test {

// Rotary embedding whose cos/sin tables are built at run time from integer positions, as in LLM and encoder
// exports. Under f16 inference the angle chain must stay f32 (positions beyond 2048 are not exact in f16),
// while the rotation of q keeps f16.
enum class RopeAngleOp { MatMulTranspose, Multiply };

using RopeAnglesF16Params = std::tuple<RopeAngleOp,  // angle operation
                                       InputShape>;  // position_ids shape

class RopeAnglesF16CPUTest : public testing::WithParamInterface<RopeAnglesF16Params>, public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RopeAnglesF16Params>& obj) {
        const auto& [angle_op, shape] = obj.param;
        std::ostringstream result;
        result << (angle_op == RopeAngleOp::MatMulTranspose ? "MatMulTranspose" : "Multiply") << "_";
        result << "IS=" << ov::test::utils::partialShape2str({shape.first}) << "_TS=";
        for (const auto& item : shape.second) {
            result << ov::test::utils::vec2str(item) << "_";
        }
        return result.str();
    }

protected:
    static constexpr size_t HEADS = 2;
    static constexpr size_t HALF_DIM = 32;
    static constexpr int64_t FIRST_POSITION = 8192;

    void SetUp() override {
        const auto& [angle_op, positions_shape] = GetParam();
        targetDevice = utils::DEVICE_CPU;
        configuration.insert({ov::hint::inference_precision.name(), ov::element::f16});
        // f16 angles give O(1) errors at these positions; f32 angles with an f16 rotation stay well under this
        abs_threshold = 2e-2;

        const auto& length = positions_shape.first[1];
        InputShape q_shape{ov::PartialShape{1, static_cast<int64_t>(HEADS), length, 2 * HALF_DIM}, {}};
        for (const auto& shape : positions_shape.second) {
            q_shape.second.push_back(ov::Shape{1, HEADS, shape[1], 2 * HALF_DIM});
        }
        init_input_shapes({positions_shape, q_shape});

        auto position_ids = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[0]);
        auto q = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[1]);

        std::vector<float> inv_freq(HALF_DIM);
        for (size_t i = 0; i < HALF_DIM; i++) {
            inv_freq[i] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * i) / static_cast<float>(2 * HALF_DIM));
        }
        std::shared_ptr<ov::Node> angles;
        if (angle_op == RopeAngleOp::MatMulTranspose) {
            auto positions = std::make_shared<ov::op::v0::Unsqueeze>(
                position_ids,
                utils::make_constant(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
            auto converted = std::make_shared<ov::op::v0::Convert>(positions, ov::element::f32);
            auto freq = utils::make_constant(ov::element::f32, ov::Shape{1, HALF_DIM, 1}, inv_freq);
            auto matmul = std::make_shared<ov::op::v0::MatMul>(freq, converted);
            angles = std::make_shared<ov::op::v1::Transpose>(
                matmul,
                utils::make_constant(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{0, 2, 1}));
        } else {
            auto positions = std::make_shared<ov::op::v0::Unsqueeze>(
                position_ids,
                utils::make_constant(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1}));
            auto converted = std::make_shared<ov::op::v0::Convert>(positions, ov::element::f32);
            auto freq = utils::make_constant(ov::element::f32, ov::Shape{1, 1, HALF_DIM}, inv_freq);
            angles = std::make_shared<ov::op::v1::Multiply>(converted, freq);
        }
        auto table = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{angles, angles}, -1);
        auto head_axis = utils::make_constant(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        auto cos = std::make_shared<ov::op::v0::Unsqueeze>(std::make_shared<ov::op::v0::Cos>(table), head_axis);
        auto sin = std::make_shared<ov::op::v0::Unsqueeze>(std::make_shared<ov::op::v0::Sin>(table), head_axis);

        // q * cos + rotate_half(q) * sin
        auto halves = std::make_shared<ov::op::v1::Split>(
            q,
            utils::make_constant(ov::element::i64, ov::Shape{}, std::vector<int64_t>{-1}),
            2);
        auto rotated_half = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{std::make_shared<ov::op::v0::Negative>(halves->output(1)), halves->output(0)},
            -1);
        rotated_half->set_friendly_name("rope_rotate_half");
        auto rotated = std::make_shared<ov::op::v1::Add>(std::make_shared<ov::op::v1::Multiply>(q, cos),
                                                         std::make_shared<ov::op::v1::Multiply>(rotated_half, sin));
        function = std::make_shared<ov::Model>(ov::OutputVector{rotated},
                                               ov::ParameterVector{position_ids, q},
                                               "RopeAnglesF16");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& model_inputs = function->inputs();
        ov::Tensor positions(ov::element::i64, targetInputStaticShapes[0]);
        auto* data = positions.data<int64_t>();
        for (size_t i = 0; i < positions.get_size(); i++) {
            data[i] = FIRST_POSITION + static_cast<int64_t>(i);
        }
        inputs.insert({model_inputs[0].get_node_shared_ptr(), positions});
        utils::InputGenerateData q_data(-1, 2, 1000);
        inputs.insert({model_inputs[1].get_node_shared_ptr(),
                       utils::create_and_fill_tensor(ov::element::f32, targetInputStaticShapes[1], q_data)});
    }

    // the angles and Sin/Cos run in f32, the rotation of q keeps f16
    // the accuracy check covers the angles; check that their f32 does not spread into the rotation of q
    void check_rotation_precision() {
        bool found = false;
        for (const auto& node : compiledModel.get_runtime_model()->get_ordered_ops()) {
            const auto& rt_info = node->get_rt_info();
            std::stringstream names(rt_info.at(ov::exec_model_info::ORIGINAL_NAMES).as<std::string>());
            for (std::string name; std::getline(names, name, ',');) {
                if (name == "rope_rotate_half") {
                    found = true;
                    EXPECT_EQ(rt_info.at(ov::exec_model_info::OUTPUT_PRECISIONS).as<std::string>(), "f16");
                }
            }
        }
        EXPECT_TRUE(found);
    }
};

TEST_P(RopeAnglesF16CPUTest, CompareWithRefs) {
    if (!ov::intel_cpu::hasHardwareSupport(ov::element::f16)) {
        GTEST_SKIP() << "f16 inference precision is not supported on this platform";
    }
    run();
    check_rotation_precision();
}

namespace {

const std::vector<InputShape> positions_shapes = {
    {{1, 64}, {{1, 64}}},
    {{1, -1}, {{1, 64}, {1, 17}}},
};

INSTANTIATE_TEST_SUITE_P(smoke_RopeAnglesF16_CPU,
                         RopeAnglesF16CPUTest,
                         ::testing::Combine(::testing::Values(RopeAngleOp::MatMulTranspose, RopeAngleOp::Multiply),
                                            ::testing::ValuesIn(positions_shapes)),
                         RopeAnglesF16CPUTest::getTestCaseName);

}  // namespace

}  // namespace ov::test
