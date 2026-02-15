// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "common_test_utils/node_builders/activation.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

/* Convolution with u8 output type should be able to fuse i8 sum (write to) without reorder

       Param1     Param2
         |          |
       FQ_U8      FQ_U8
         |          |
       Conv1      Conv2
         |          |
       FQ_I8      FQ_I8
         \          /
          \        /
           ADD(Sum)
              |
              |
            ReLU
              |
            FQ_U8
              |
            Conv
              |
            FQ_I8
              |
            Result
*/

class ConvU8FuseSumI8 : virtual public SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        ov::element::Type netPrecision = ov::element::f32;

        targetDevice = ov::test::utils::DEVICE_CPU;

        auto make_i8_fake_quantize = [&](std::shared_ptr<ov::Node> input, ov::element::Type dataType) {
            return ov::test::utils::make_fake_quantize(input, dataType, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
        };

        auto make_u8_fake_quantize = [&](std::shared_ptr<ov::Node> input, ov::element::Type dataType) {
            return ov::test::utils::make_fake_quantize(input, dataType, 256, {}, {0.0f}, {2.55f}, {0.0f}, {2.55f});
        };

        auto make_quantized_weights = [&make_i8_fake_quantize](const Shape& shape, ov::element::Type dataType) {
            auto weights = ov::op::v0::Constant::create(dataType, shape, std::vector<float>{-0.0512377955019474});
            return make_i8_fake_quantize(weights, dataType);
        };

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netPrecision, ov::Shape{1, 3, 8, 8}),
                                   std::make_shared<ov::op::v0::Parameter>(netPrecision, ov::Shape{1, 3, 8, 8})};

        auto create_int8_conv_block = [&](std::shared_ptr<ov::Node> input) {
            auto fq_input = make_u8_fake_quantize(input, netPrecision);
            auto fq_weights = make_quantized_weights({3, 3, 4, 4}, netPrecision);

            auto conv = std::make_shared<ov::op::v1::Convolution>(fq_input,
                                                                  fq_weights,
                                                                  Strides{1, 1},
                                                                  CoordinateDiff{0, 0},
                                                                  CoordinateDiff{0, 0},
                                                                  Strides{1, 1},
                                                                  ov::op::PadType::SAME_UPPER);

            return make_i8_fake_quantize(conv, netPrecision);
        };

        auto sum = ov::test::utils::make_eltwise(create_int8_conv_block(params[0]),
                                                 create_int8_conv_block(params[1]),
                                                 ov::test::utils::EltwiseTypes::ADD);

        auto activation = ov::test::utils::make_activation(sum, netPrecision, ov::test::utils::ActivationTypes::Relu);
        auto final_fq = create_int8_conv_block(activation);

        auto result = std::make_shared<ov::op::v0::Result>(final_fq);
        function = std::make_shared<ov::Model>(result, params, "ConvU8FuseSumI8");
    }
};

TEST_F(ConvU8FuseSumI8, smoke_CompareWithRefs) {
    run();
    // 2 input reorders (abcd -> acdb) before 2 convolutions + one output reorderr (acdb -> abcd) before result
    // No reorder (i8 -> u8) is expected on the port for a fused sum
    CPUTestUtils::CheckNumberOfNodesWithType(compiledModel, "Reorder", 3);
}

}  // namespace test
}  // namespace ov
