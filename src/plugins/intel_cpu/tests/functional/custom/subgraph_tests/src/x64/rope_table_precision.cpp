// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/precision_support.h"

namespace ov::test {

// Decomposed rope table applied to a projection: the angle chain must stay f32 under bf16/f16
// inference, MatMuls stay in low precision. The static case also exercises Snippets tokenization.
enum class AngleChain {
    SHIFTED_GRID,   // grid * freqs + shift -> Transpose -> Reshape (LTX-Video)
    CENTERED_GRID,  // (grid * 2 - 1) * freqs -> Reshape (LTX-2, single position axis)
};

inline std::ostream& operator<<(std::ostream& os, AngleChain chain) {
    return os << (chain == AngleChain::SHIFTED_GRID ? "ShiftedGrid" : "CenteredGrid");
}

using RopeTablePrecisionParams = std::tuple<ov::element::Type,  // inference precision
                                            AngleChain,         // angle chain topology
                                            InputShape>;        // hidden states shape

class RopeTablePrecisionCPUTest : public testing::WithParamInterface<RopeTablePrecisionParams>,
                                  public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RopeTablePrecisionParams>& obj) {
        const auto& [infer_prc, chain, shape] = obj.param;
        std::ostringstream result;
        result << "inferPRC=" << infer_prc << "_" << chain << "_" << (shape.first.is_dynamic() ? "dynamic" : "static");
        result << "_IS=" << ov::test::utils::partialShape2str({shape.first});
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [infer_prc, chain, hidden_shape] = GetParam();
        targetDevice = utils::DEVICE_CPU;
        configuration.insert({ov::hint::inference_precision.name(), infer_prc});
        // low precision angles give O(1) errors; honest rounding stays well under this
        rel_threshold = 0.05;
        abs_threshold = 0.5;

        // the rope table size is fixed; only the hidden states shape varies
        init_input_shapes({InputShape{ov::PartialShape{1, GRID, 1}, {ov::Shape{1, GRID, 1}}}, hidden_shape});

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[1])};

        // ~1.6e4 rad max after the freq multiply, as in LTX models: low precision steps exceed 2*pi
        // (f16: 8 rad, bf16: 64 rad), while f32 sin/cos stay exact
        auto freqs = utils::make_constant(ov::element::f32, ov::Shape{1, 1, BANDS}, std::vector<float>{1, 16, 64, 250});
        auto minus_one = utils::make_constant(ov::element::f32, ov::Shape{}, std::vector<float>{-1.0F});
        auto target_shape =
            utils::make_constant(ov::element::i32, ov::Shape{2}, std::vector<int>{1, static_cast<int>(HIDDEN_SIZE)});
        std::shared_ptr<ov::Node> reshape;
        if (chain == AngleChain::SHIFTED_GRID) {
            auto angles = std::make_shared<ov::op::v1::Multiply>(params[0], freqs);
            auto shifted = std::make_shared<ov::op::v1::Add>(angles, minus_one);
            auto order = utils::make_constant(ov::element::i32, ov::Shape{3}, std::vector<int>{0, 2, 1});
            auto transpose = std::make_shared<ov::op::v1::Transpose>(shifted, order);
            reshape = std::make_shared<ov::op::v1::Reshape>(transpose, target_shape, false);
        } else {
            auto two = utils::make_constant(ov::element::f32, ov::Shape{}, std::vector<float>{2.0F});
            auto scaled = std::make_shared<ov::op::v1::Multiply>(params[0], two);
            auto centered = std::make_shared<ov::op::v1::Add>(scaled, minus_one);
            auto angles = std::make_shared<ov::op::v1::Multiply>(centered, freqs);
            reshape = std::make_shared<ov::op::v1::Reshape>(angles, target_shape, false);
        }
        auto cos = std::make_shared<ov::op::v0::Cos>(reshape);
        auto sin = std::make_shared<ov::op::v0::Sin>(reshape);

        ov::test::utils::InputGenerateData weights_data(-1, 2, 256);
        auto proj_weights = utils::make_constant(ov::element::f32, ov::Shape{HIDDEN_SIZE, HIDDEN_SIZE}, weights_data);
        auto q = std::make_shared<ov::op::v0::MatMul>(params[1], proj_weights);

        auto q_cos = std::make_shared<ov::op::v1::Multiply>(q, cos);
        auto q_sin = std::make_shared<ov::op::v1::Multiply>(q, sin);
        auto rotated = std::make_shared<ov::op::v1::Add>(q_cos, q_sin);

        auto out_weights = utils::make_constant(ov::element::f32, ov::Shape{HIDDEN_SIZE, HIDDEN_SIZE}, weights_data);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(rotated, out_weights);

        function = std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(matmul)},
                                               params,
                                               "RopeTablePrecision");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        ov::Tensor grid{ov::element::f32, targetInputStaticShapes[0]};
        auto* grid_data = grid.data<float>();
        for (size_t i = 0; i < grid.get_size(); ++i) {
            grid_data[i] = static_cast<float>(i);
        }
        inputs.insert({funcInputs[0].get_node_shared_ptr(), grid});

        utils::InputGenerateData in_data;
        in_data.start_from = -1;
        in_data.range = 2;
        in_data.resolution = 256;
        auto hidden =
            utils::create_and_fill_tensor(funcInputs[1].get_element_type(), targetInputStaticShapes[1], in_data);
        inputs.insert({funcInputs[1].get_node_shared_ptr(), hidden});
    }

    static constexpr size_t GRID = 64;
    static constexpr size_t BANDS = 4;
    static constexpr size_t HIDDEN_SIZE = GRID * BANDS;
};

TEST_P(RopeTablePrecisionCPUTest, CompareWithRefs) {
    const auto infer_prc = std::get<0>(GetParam());
    if (!ov::intel_cpu::hasHardwareSupport(infer_prc)) {
        GTEST_SKIP() << "No " << infer_prc << " support";
    }
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_RopeTablePrecision,
                         RopeTablePrecisionCPUTest,
                         testing::Combine(testing::Values(ov::element::bf16, ov::element::f16),
                                          testing::Values(AngleChain::SHIFTED_GRID, AngleChain::CENTERED_GRID),
                                          testing::Values(InputShape{ov::PartialShape{-1, 256},
                                                                     {ov::Shape{64, 256}, ov::Shape{32, 256}}},
                                                          InputShape{ov::PartialShape{64, 256}, {ov::Shape{64, 256}}})),
                         RopeTablePrecisionCPUTest::getTestCaseName);

}  // namespace ov::test
