// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits>
#include <string>

#include "internal_properties.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov::test {

/* `(1 - mask) * f32_lowest` shared by two attention blocks is reassociated into `mask * f32_max + f32_lowest`.
   f32_max overflows to +inf in bf16 and Softmax produces NaN (inf - inf). Covered with and without Snippets.

            mask[1,4]
               |
           Unsqueeze
               |
           Unsqueeze
               |
      Subtract(1, x)
               |
   Multiply(x, f32_lowest)
          /          \
         /            \
  Add(Q0 x K^T)   Add(Q1 x K^T)
        |              |
     Softmax        Softmax
        |              |
      x V            x V
*/
class BF16AttentionMask : public testing::WithParamInterface<bool>, virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<bool>& obj) {
        return obj.param ? "Snippets=ON" : "Snippets=OFF";
    }

protected:
    void SetUp() override {
        if (!ov::with_cpu_x86_bfloat16()) {
            GTEST_SKIP() << "Native bf16 support is required";
        }
        targetDevice = utils::DEVICE_CPU;
        configuration[ov::hint::inference_precision.name()] = ov::element::bf16;
        if (!GetParam()) {
            configuration.insert(ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE));
        }
        abs_threshold = 0.1f;

        const Shape qkv_shape{1, 1, 4, 4};
        auto query0 = std::make_shared<op::v0::Parameter>(element::f32, qkv_shape);
        auto query1 = std::make_shared<op::v0::Parameter>(element::f32, qkv_shape);
        auto key = std::make_shared<op::v0::Parameter>(element::f32, qkv_shape);
        auto value = std::make_shared<op::v0::Parameter>(element::f32, qkv_shape);
        auto mask = std::make_shared<op::v0::Parameter>(element::f32, Shape{1, 4});

        auto axes = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto expanded = std::make_shared<op::v0::Unsqueeze>(std::make_shared<op::v0::Unsqueeze>(mask, axes), axes);
        auto complement =
            std::make_shared<op::v1::Subtract>(op::v0::Constant::create(element::f32, Shape{1, 1, 1, 1}, {1.f}),
                                               expanded);
        auto extended_mask = std::make_shared<op::v1::Multiply>(
            complement,
            op::v0::Constant::create(element::f32, Shape{1, 1, 1, 1}, {std::numeric_limits<float>::lowest()}));

        OutputVector results;
        for (const auto& query : {query0, query1}) {
            auto scores = std::make_shared<op::v0::MatMul>(query, key, false, true);
            auto attention = std::make_shared<op::v1::Softmax>(std::make_shared<op::v1::Add>(scores, extended_mask), 3);
            auto output = std::make_shared<op::v0::MatMul>(attention, value);
            results.push_back(std::make_shared<op::v0::Convert>(output, element::bf16));
        }
        function = std::make_shared<Model>(results, ParameterVector{query0, query1, key, value, mask});
    }

    void generate_inputs(const std::vector<ov::Shape>&) override {
        inputs.clear();
        const auto& parameters = function->get_parameters();
        for (const auto& parameter : parameters) {
            ov::Tensor tensor{element::f32, parameter->get_shape()};
            auto* data = tensor.data<float>();
            // all positions of the mask are unmasked (1), so every Softmax row must stay finite
            const bool is_mask = parameter == parameters.back();
            for (size_t i = 0; i < tensor.get_size(); ++i) {
                data[i] = is_mask ? 1.f : static_cast<float>(i % 7) / 10.f;
            }
            inputs.insert({parameter, tensor});
        }
    }
};

TEST_P(BF16AttentionMask, CompareWithRefs) {
    run();
}

INSTANTIATE_TEST_SUITE_P(smoke_BF16AttentionMask,
                         BF16AttentionMask,
                         ::testing::Values(true, false),
                         BF16AttentionMask::getTestCaseName);

}  // namespace ov::test
