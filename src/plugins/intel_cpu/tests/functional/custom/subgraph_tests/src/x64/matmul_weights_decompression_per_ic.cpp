// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/subtract.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

// Subgraph pattern:
//
//    Weights(WP)
//        |
//    Convert(f32)   ZP [IC, 1] (opt.)
//            \       /
//            Subtract(opt.)  Scale [IC, 1]
//                   \       /
//    Data(f32)      Multiply
//          \         /
//            MatMul
using MatMulWeightsDecompressionPerICParams = std::tuple<ov::Shape,  // data shape
                                                         bool>;      // with per-IC zero point

class MatMulWeightsDecompressionPerIC : public testing::WithParamInterface<MatMulWeightsDecompressionPerICParams>,
                                        virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulWeightsDecompressionPerICParams>& obj) {
        const auto& [data_shape, with_zp] = obj.param;
        std::ostringstream result;
        result << "IS=" << data_shape << "_withZp=" << with_zp;
        return result.str();
    }

protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        const auto& [data_shape, with_zp] = GetParam();
        const size_t IC = data_shape.back();
        const size_t OC = 64;

        init_input_shapes(static_shapes_to_test_representation({data_shape}));
        auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);

        const auto weights_precision = with_zp ? ov::element::u8 : ov::element::i8;
        auto weights = utils::make_constant(weights_precision,
                                            ov::Shape{IC, OC},
                                            utils::InputGenerateData(with_zp ? 0 : -8, 16, 1, 1));
        std::shared_ptr<ov::Node> decompressed = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
        if (with_zp) {
            auto zp = utils::make_constant(ov::element::u8, ov::Shape{IC, 1}, utils::InputGenerateData(6, 4, 1, 2));
            auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp, ov::element::f32);
            decompressed = std::make_shared<ov::op::v1::Subtract>(decompressed, zp_convert);
        }
        auto scale = utils::make_constant(ov::element::f32, ov::Shape{IC, 1}, utils::InputGenerateData(0, 1, 100, 3));
        auto mul = std::make_shared<ov::op::v1::Multiply>(decompressed, scale);
        auto matmul = std::make_shared<ov::op::v0::MatMul>(data, mul);

        function = std::make_shared<ov::Model>(ov::OutputVector{matmul},
                                               ov::ParameterVector{data},
                                               "MatMulWeightsDecompressionPerIC");
        configuration.insert({ov::hint::inference_precision.name(), ov::element::bf16});
        abs_threshold = 0.5f;
    }
};

TEST_P(MatMulWeightsDecompressionPerIC, CompareWithRefs) {
    if (!ov::with_cpu_x86_bfloat16()) {
        GTEST_SKIP();
    }
    run();
}

namespace {

INSTANTIATE_TEST_SUITE_P(
    smoke_MatMulWeightsDecompressionPerIC,
    MatMulWeightsDecompressionPerIC,
    ::testing::Combine(::testing::Values(ov::Shape{1, 1, 64}, ov::Shape{1, 16, 64}, ov::Shape{1, 64, 768}),
                       ::testing::Values(false, true)),
    MatMulWeightsDecompressionPerIC::getTestCaseName);

}  // namespace

}  // namespace ov::test
