// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/node_builders/group_convolution.hpp"
#include "utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

class ConvDWConv : virtual public ov::test::SubgraphBaseTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto precision = ov::element::f32;
        ov::test::InputShape input_shape{{}, {{1, 32, 112, 112}}};
        init_input_shapes({input_shape});


        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, shape));
        }
        auto conv_weights = ov::test::utils::make_constant(precision, std::vector<size_t>{32, 32, 1, 1});
        auto conv = ov::test::utils::make_convolution(params[0],
                                                      conv_weights,
                                                      precision,
                                                      std::vector<size_t>{1, 1},
                                                      std::vector<size_t>{1, 1},
                                                      ov::CoordinateDiff{0, 0},
                                                      ov::CoordinateDiff{0, 0},
                                                      std::vector<size_t>{1, 1},
                                                      ov::op::PadType::EXPLICIT,
                                                      32,
                                                      true);

        auto dw_conv_weights = ov::test::utils::make_constant(precision, std::vector<size_t>{32, 1, 1, 3, 3});
        auto dw_conv = ov::test::utils::make_group_convolution(conv,
                                                               dw_conv_weights,
                                                               precision,
                                                               std::vector<size_t>{1, 1},
                                                               ov::CoordinateDiff{1, 1},
                                                               ov::CoordinateDiff{1, 1},
                                                               std::vector<size_t>{1, 1},
                                                               ov::op::PadType::EXPLICIT);
        auto bias_const = ov::test::utils::make_constant(precision, {1, 32 , 1, 1});
        auto bias = std::make_shared<ov::opset10::Add>(dw_conv, bias_const);
        function = std::make_shared<ov::Model>(bias, params, "ConvDWConv");
    }
};

TEST_F(ConvDWConv, smoke_CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
