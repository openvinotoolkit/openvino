// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/softmax.hpp"

namespace ov {
namespace test {
/*
 *                    input2
 *                      |
 *  Constant(i8)     Softmax
 *       |            /
 *    Convert     Multiply
 *       |          /
 *    Multiply  Convert   input1(u8/i8)
 *         \     /          |
 *          Gather     FakeQuantize
 *              \       /
 *               \     /
 *               MatMul
 *
 * With a per-row weights scale and non-constant indices, LPT can't move the dequantization through Gather,
 * so GatherCompressed must be kept for i8/u8 weights as well. A per-column scale is still handled by LPT.
 */
enum class GatherScaleShape { SCALAR, PER_ROW, PER_COLUMN };

static std::ostream& operator<<(std::ostream& os, GatherScaleShape scale_shape) {
    switch (scale_shape) {
    case GatherScaleShape::SCALAR:
        return os << "scalar";
    case GatherScaleShape::PER_ROW:
        return os << "per_row";
    case GatherScaleShape::PER_COLUMN:
        return os << "per_column";
    }
    OPENVINO_THROW("Unexpected GatherScaleShape");
}

static Shape get_scale_shape(GatherScaleShape scale_shape, const Shape& weights_shape) {
    switch (scale_shape) {
    case GatherScaleShape::PER_ROW:
        return {weights_shape[0], 1};
    case GatherScaleShape::PER_COLUMN:
        return {1, weights_shape[1]};
    case GatherScaleShape::SCALAR:
        return {1, 1};
    }
    OPENVINO_THROW("Unexpected GatherScaleShape");
}

using DisableGatherCompressedForQuantizedModelParams =
    std::tuple<element::Type, InputShape, InputShape, GatherScaleShape>;
class DisableGatherCompressedForQuantizedModel : public testing::WithParamInterface<DisableGatherCompressedForQuantizedModelParams>,
                                                 virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<DisableGatherCompressedForQuantizedModelParams>& obj) {
        const auto& [weight_prec, inputShape1, inputShape2, scale_shape] = obj.param;
        std::ostringstream result;
        result << "weight_prec=" << weight_prec << "_" << "inputShape1=" << inputShape1 << "_"
               << "inputShape2=" << inputShape2 << "_" << "scale=" << scale_shape;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        const auto& [weight_prec, inputShape1, inputShape2, scale_shape] = GetParam();
        init_input_shapes({inputShape1, inputShape2});

        targetDevice = utils::DEVICE_CPU;
        auto type = element::f32;

        auto input1 = std::make_shared<op::v0::Parameter>(type, inputDynamicShapes[0]);
        auto input2 = std::make_shared<op::v0::Parameter>(type, inputDynamicShapes[1]);

        auto shared_il = op::v0::Constant::create(type, {1, 1, 1, 1}, {0.f});
        auto shared_ih = op::v0::Constant::create(type, {1, 1, 1, 1}, {12.5f});
        auto shared_ol = op::v0::Constant::create(type, {1, 1, 1, 1}, {0.f});
        auto shared_oh = op::v0::Constant::create(type, {1, 1, 1, 1}, {12.5f});
        auto fq = std::make_shared<op::v0::FakeQuantize>(input1, shared_il, shared_ih, shared_ol, shared_oh, 256);

        // Weights
        auto weights_shape = Shape{64, 64};
        auto weights = utils::make_constant(weight_prec, weights_shape, utils::InputGenerateData(-1, 2, 32768));
        auto convert = std::make_shared<op::v0::Convert>(weights, element::f32);
        const auto scale_const_shape = get_scale_shape(scale_shape, weights_shape);
        auto scale = scale_shape == GatherScaleShape::SCALAR
                         ? op::v0::Constant::create(type, scale_const_shape, {0.625})
                         : utils::make_constant(type, scale_const_shape, utils::InputGenerateData(1, 1, 1000));
        auto multiply = std::make_shared<op::v1::Multiply>(convert, scale);
        // Indics
        auto softmax = std::make_shared<op::v1::Softmax>(input2, 0);
        auto multiply2 = std::make_shared<op::v1::Multiply>(softmax, op::v0::Constant::create(type, {1}, {64}));
        auto indics = std::make_shared<op::v0::Convert>(multiply2, element::i64);
        // Gather
        auto gather =
            std::make_shared<op::v8::Gather>(multiply, indics, op::v0::Constant::create(element::i32, Shape{1}, {0}));

        auto matMul = std::make_shared<ov::op::v0::MatMul>(fq, gather, false, true);

        function = std::make_shared<Model>(matMul, ParameterVector{input1, input2});
    }

    void check_results() {
        const auto& test_param = GetParam();
        const auto compressed_weights_precision = std::get<0>(test_param);
        const auto scale_shape = std::get<3>(test_param);

        const auto runtime_model = compiledModel.get_runtime_model();
        const auto matmul = runtime_model->get_result()->get_input_node_shared_ptr(0);

        std::shared_ptr<const ov::Node> gather = nullptr;
        bool have_gather = false;
        bool have_gather_compressed = false;
        for (const auto& n : runtime_model->get_ordered_ops()) {
            const auto type = n->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            if (type == "Gather") {
                gather = n;
                // Gather has >=4 inputs means it is GatherCompressed.
                if (n->get_input_size() >= 4) {
                    have_gather_compressed = true;
                } else {
                    have_gather = true;
                }
            }
        }

        const auto matmul_runtime_precision =
            matmul->get_rt_info().at(ov::exec_model_info::RUNTIME_PRECISION).as<ov::element::Type>();
        switch (compressed_weights_precision) {
        case element::i8:
        case element::u8:
            switch (scale_shape) {
            case GatherScaleShape::PER_ROW:
                // LPT can't move a per-row scale through Gather with non-constant indices, so GatherCompressed
                // dequantizes the compressed weights itself and MatMul is executed in a real precision
                // (f32, or bf16 on bf16-capable hosts).
                EXPECT_TRUE(have_gather_compressed);
                EXPECT_FALSE(have_gather);
                ASSERT_NE(gather, nullptr);
                EXPECT_EQ(gather->get_input_element_type(0), compressed_weights_precision);
                EXPECT_TRUE(matmul->get_input_element_type(1).is_real());
                EXPECT_TRUE(matmul_runtime_precision.is_real());
                break;
            case GatherScaleShape::PER_COLUMN:
                // LPT moves the dequantization after Gather, so Gather works on the compressed weights. The
                // per-column scale is along the MatMul reduction axis, so LPT keeps it before MatMul, which is
                // executed in a real precision.
                EXPECT_TRUE(have_gather);
                EXPECT_FALSE(have_gather_compressed);
                ASSERT_NE(gather, nullptr);
                EXPECT_EQ(gather->get_output_element_type(0), compressed_weights_precision);
                EXPECT_TRUE(matmul->get_input_element_type(1).is_real());
                EXPECT_TRUE(matmul_runtime_precision.is_real());
                break;
            case GatherScaleShape::SCALAR:
                EXPECT_TRUE(have_gather);
                if (compressed_weights_precision == element::i8) {
                    EXPECT_EQ(matmul->get_input_element_type(1), element::i8);
                    // FakeQuantize(matmul's input(0))'s output precision is u8
                    EXPECT_EQ(matmul_runtime_precision, element::u8);
                }
                // Current oneDNN MutMul official support precision: Source(u8, s8), Weights(s8).
                // So reorder will be inserted when weights is not s8, don't need to check matmul's input(1) precision.
                break;
            }
            break;
        case element::u4:
        case element::i4:
            EXPECT_TRUE(have_gather_compressed);
            break;
        default:
            break;
        }
    }
};

TEST_P(DisableGatherCompressedForQuantizedModel, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    check_results();
}

namespace {

const std::vector<InputShape> inputShapes1 = {{{-1, 3, -1, -1}, {{1, 3, 64, 64}}}};
const std::vector<InputShape> inputShapes2 = {{{}, {{32}}}};
const std::vector<element::Type> weightsPrecisions = {element::i8, element::u8, element::u4, element::i4};

INSTANTIATE_TEST_SUITE_P(smoke_DisableGatherCompressedForQuantizedModel_basic,
                         DisableGatherCompressedForQuantizedModel,
                         ::testing::Combine(::testing::ValuesIn(weightsPrecisions),
                                            ::testing::ValuesIn(inputShapes1),
                                            ::testing::ValuesIn(inputShapes2),
                                            ::testing::Values(GatherScaleShape::SCALAR)),
                         DisableGatherCompressedForQuantizedModel::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_DisableGatherCompressedForQuantizedModel_non_scalar_scale,
                         DisableGatherCompressedForQuantizedModel,
                         ::testing::Combine(::testing::Values(element::i8, element::u8),
                                            ::testing::ValuesIn(inputShapes1),
                                            ::testing::ValuesIn(inputShapes2),
                                            ::testing::Values(GatherScaleShape::PER_ROW, GatherScaleShape::PER_COLUMN)),
                         DisableGatherCompressedForQuantizedModel::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
