// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/data_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

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
 */
using DisableGatherCompressedForQuantizedModelParams = std::tuple<element::Type, InputShape, InputShape>;
class DisableGatherCompressedForQuantizedModel : public testing::WithParamInterface<DisableGatherCompressedForQuantizedModelParams>,
                                                 virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<DisableGatherCompressedForQuantizedModelParams> obj) {
        element::Type weight_prec;
        InputShape inputShape1, inputShape2;
        std::tie(weight_prec, inputShape1, inputShape2) = obj.param;
        std::ostringstream result;
        result << "weight_prec=" << weight_prec << "_" << "inputShape1=" << inputShape1 << "_"
               << "inputShape2=" << inputShape2;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        element::Type weight_prec;
        InputShape inputShape1, inputShape2;
        std::tie(weight_prec, inputShape1, inputShape2) = GetParam();

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
        auto multiply = std::make_shared<op::v1::Multiply>(convert, op::v0::Constant::create(type, {1, 1}, {0.625}));
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

        const auto runtime_model = compiledModel.get_runtime_model();
        const auto matmul = runtime_model->get_result()->get_input_node_shared_ptr(0);

        bool have_gather = false;
        bool have_gather_compressed = false;
        for (const auto& n : runtime_model->get_ordered_ops()) {
            const auto type = n->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            if (type == "Gather") {
                // Gather has >=4 inputs means it is GatherCompressed.
                if (n->get_input_size() >= 4) {
                    have_gather_compressed = true;
                } else {
                    have_gather = true;
                }
            }
        }

        switch (compressed_weights_precision) {
        case element::i8:
            EXPECT_TRUE(have_gather);
            EXPECT_EQ(matmul->get_input_element_type(1), element::i8);
            // FakeQuantize(matmul's input(0))'s output precision is u8
            EXPECT_EQ(matmul->get_rt_info().at(ov::exec_model_info::RUNTIME_PRECISION).as<ov::element::Type>(),
                      element::u8);
            break;
        case element::u8:
            EXPECT_TRUE(have_gather);
            // Current oneDNN MutMul official support precision: Source(u8, s8), Weights(s8).
            // So reorder will be inserted when weights is not s8, don't need to check matmul's input(1) precision.
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
                                            ::testing::ValuesIn(inputShapes2)),
                         DisableGatherCompressedForQuantizedModel::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
