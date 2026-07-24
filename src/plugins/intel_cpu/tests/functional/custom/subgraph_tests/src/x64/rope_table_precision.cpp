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

namespace ov::test {

// Decomposed LTX-Video rope table applied to a projection: the angle chain must stay f32 under bf16
// enforcement, MatMuls stay bf16. The static case also exercises Snippets tokenization.
using RopeTablePrecisionParams = std::tuple<ov::element::Type,  // inference precision
                                            InputShape>;        // hidden states shape

class RopeTablePrecisionCPUTest : public testing::WithParamInterface<RopeTablePrecisionParams>,
                                  public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RopeTablePrecisionParams>& obj) {
        const auto& [infer_prc, shape] = obj.param;
        std::ostringstream result;
        result << "inferPRC=" << infer_prc << "_" << (shape.first.is_dynamic() ? "dynamic" : "static");
        result << "_IS=" << ov::test::utils::partialShape2str({shape.first});
        return result.str();
    }

protected:
    void SetUp() override {
        const auto& [infer_prc, hidden_shape] = GetParam();
        targetDevice = utils::DEVICE_CPU;
        configuration.insert({ov::hint::inference_precision.name(), infer_prc});
        // low precision angles give O(1) errors; honest rounding stays well under this
        rel_threshold = 0.05;
        abs_threshold = 0.5;

        // the rope table size is fixed; only the hidden states shape varies
        init_input_shapes({InputShape{ov::PartialShape{1, GRID, 1}, {ov::Shape{1, GRID, 1}}}, hidden_shape});

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[1])};

        // ~500 rad max after the freq multiply: fatal for bf16 (~2 rad steps), exact for f32 sin/cos
        auto freqs = utils::make_constant(ov::element::f32, ov::Shape{1, 1, BANDS}, std::vector<float>{1, 2, 4, 8});
        auto angles = std::make_shared<ov::op::v1::Multiply>(params[0], freqs);
        auto shifted = std::make_shared<ov::op::v1::Add>(
            angles,
            utils::make_constant(ov::element::f32, ov::Shape{}, std::vector<float>{-1.0F}));
        auto order = utils::make_constant(ov::element::i32, ov::Shape{3}, std::vector<int>{0, 2, 1});
        auto transpose = std::make_shared<ov::op::v1::Transpose>(shifted, order);
        auto target_shape =
            utils::make_constant(ov::element::i32, ov::Shape{2}, std::vector<int>{1, static_cast<int>(HIDDEN_SIZE)});
        auto reshape = std::make_shared<ov::op::v1::Reshape>(transpose, target_shape, false);
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
    // gate on the plugin's own capability: bf16 is also supported without avx512 (avx2_vnni_2)
    const auto capabilities = core->get_property(targetDevice, ov::device::capabilities);
    if (std::find(capabilities.begin(), capabilities.end(), "BF16") == capabilities.end()) {
        GTEST_SKIP() << "No BF16 support";
    }
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_RopeTablePrecision,
                         RopeTablePrecisionCPUTest,
                         testing::Combine(testing::Values(ov::element::bf16),
                                          testing::Values(InputShape{ov::PartialShape{-1, 256},
                                                                     {ov::Shape{64, 256}, ov::Shape{32, 256}}},
                                                          InputShape{ov::PartialShape{64, 256}, {ov::Shape{64, 256}}})),
                         RopeTablePrecisionCPUTest::getTestCaseName);

}  // namespace ov::test
