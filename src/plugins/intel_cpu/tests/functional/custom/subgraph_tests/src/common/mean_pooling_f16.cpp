// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/precision_support.h"

namespace ov::test {

// Masked mean pooling without eps, sum(hidden * mask, 1) / mask.sum(1), as in HF ModernBERT and wav2vec2 heads.
// The sum over a long sequence exceeds the f16 range although the mean does not: under f16 inference it must be
// accumulated and divided in f32.
class MeanPoolingF16CPUTest : public testing::WithParamInterface<size_t>, public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<size_t>& obj) {
        return "length=" + std::to_string(obj.param);
    }

protected:
    static constexpr size_t BATCH = 2;
    static constexpr size_t HIDDEN = 128;
    // with the longer length, the second sequence is padded beyond this count, which is not exact in f16
    static constexpr size_t PADDED_LENGTH = 3001;

    void SetUp() override {
        const auto length = GetParam();
        targetDevice = utils::DEVICE_CPU;
        configuration.insert({ov::hint::inference_precision.name(), ov::element::f16});
        // the pooled f32 value is rounded to f16 once (half of the f16 step at 55 is 0.0156); a count rounded to f16
        // gives up to 0.034, f16 accumulation errors and overflows are far above
        abs_threshold = 0.025;
        rel_threshold = 0;

        init_input_shapes(static_shapes_to_test_representation({{BATCH, length, HIDDEN}, {BATCH, length}}));
        auto hidden = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        auto mask = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, inputDynamicShapes[1]);

        auto sequence_axis = utils::make_constant(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        auto expanded_mask = std::make_shared<ov::op::v0::Convert>(
            std::make_shared<ov::op::v0::Unsqueeze>(
                mask,
                utils::make_constant(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1})),
            ov::element::f32);
        auto sum =
            std::make_shared<ov::op::v1::ReduceSum>(std::make_shared<ov::op::v1::Multiply>(hidden, expanded_mask),
                                                    sequence_axis,
                                                    false);
        auto count =
            std::make_shared<ov::op::v0::Convert>(std::make_shared<ov::op::v1::ReduceSum>(mask, sequence_axis, true),
                                                  ov::element::f32);
        auto mean = std::make_shared<ov::op::v1::Divide>(sum, count);
        function =
            std::make_shared<ov::Model>(ov::OutputVector{mean}, ov::ParameterVector{hidden, mask}, "MeanPooling");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& model_inputs = function->inputs();
        // values around 55, exact in f16: the sum of 1200 tokens already exceeds the f16 range
        utils::InputGenerateData hidden_data(54, 2, 32);
        inputs.insert({model_inputs[0].get_node_shared_ptr(),
                       utils::create_and_fill_tensor(ov::element::f32, targetInputStaticShapes[0], hidden_data)});
        const auto& mask_shape = targetInputStaticShapes[1];
        ov::Tensor mask(ov::element::i64, mask_shape);
        auto* data = mask.data<int64_t>();
        const auto length = mask_shape[1];
        for (size_t b = 0; b < mask_shape[0]; b++) {
            const auto valid = b == 0 ? length : std::min(length, PADDED_LENGTH);
            for (size_t i = 0; i < length; i++) {
                data[b * length + i] = i < valid ? 1 : 0;
            }
        }
        inputs.insert({model_inputs[1].get_node_shared_ptr(), mask});
    }
};

TEST_P(MeanPoolingF16CPUTest, CompareWithRefs) {
    if (!ov::intel_cpu::hasHardwareSupport(ov::element::f16)) {
        GTEST_SKIP() << "f16 inference precision is not supported on this platform";
    }
    run();
}

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_MeanPoolingF16_CPU,
                         MeanPoolingF16CPUTest,
                         ::testing::Values(1024, 4096),
                         MeanPoolingF16CPUTest::getTestCaseName);

}  // namespace

}  // namespace ov::test
