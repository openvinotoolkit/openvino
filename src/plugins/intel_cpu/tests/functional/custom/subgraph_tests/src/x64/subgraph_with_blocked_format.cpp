// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "openvino/opsets/opset8.hpp"

namespace ov {
namespace test {

class SubgraphWithBlockedFormat : virtual public SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        abs_threshold = 1e-2;

        auto type = element::f32;
        auto param = std::make_shared<ov::opset8::Parameter>(type, Shape{1, 32, 64, 32});
        auto weights_tensor = ov::test::utils::create_and_fill_tensor_real_distribution(type, Shape{32, 32, 1, 1}, 1, 10, 1);
        auto weights = std::make_shared<ov::op::v0::Constant>(weights_tensor);
        auto conv = std::make_shared<ov::opset8::Convolution>(param,
                                                              weights,
                                                              Strides{1, 1},
                                                              CoordinateDiff{0, 0},
                                                              CoordinateDiff{0, 0},
                                                              Strides{1, 1});
        auto mean =
            std::make_shared<ov::opset8::ReduceMean>(conv,
                                                     ov::opset8::Constant::create(element::i32, Shape{2}, {2, 3}),
                                                     true);
        auto reshape_before =
            std::make_shared<ov::op::v1::Reshape>(mean,
                                                  ov::opset8::Constant::create(element::i32, Shape{3}, {0, 16, -1}),
                                                  true);
        auto mvn = std::make_shared<ov::opset8::MVN>(reshape_before,
                                                     ov::opset8::Constant::create(element::i32, Shape{1}, {2}),
                                                     false,
                                                     0.1,
                                                     op::MVNEpsMode::INSIDE_SQRT);
        auto reshape_after =
            std::make_shared<ov::op::v1::Reshape>(mvn, std::make_shared<ov::op::v3::ShapeOf>(mean), false);
        auto mul = std::make_shared<ov::opset8::Multiply>(
            reshape_after,
            ov::test::utils::make_constant(type, Shape{32, 1, 1}));
        auto add = std::make_shared<ov::opset8::Add>(
            mul,
            ov::test::utils::make_constant(type, Shape{32, 1, 1}));
        auto sigmoid = std::make_shared<ov::opset8::Sigmoid>(add);
        auto mul2 = std::make_shared<ov::opset8::Multiply>(conv, sigmoid);

        function = std::make_shared<ov::Model>(mul2, ParameterVector{param});
    }

    void check_results() {
        auto runtime_function = compiledModel.get_runtime_model();
        int nodes_found = 0;
        for (const auto& n : runtime_function->get_ordered_ops()) {
            auto layer_type = n->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            if (layer_type == "Subgraph") {
                nodes_found++;
                auto output_layout = n->get_rt_info().at(ov::exec_model_info::OUTPUT_LAYOUTS).as<std::string>();
                // convolution maybe chooses 'nhwc' and the subgraph will follow it
                ASSERT_TRUE(output_layout == "aBcd8b" || output_layout == "aBcd16b" || output_layout == "acdb");
            }
        }
        ASSERT_GT(nodes_found, 0);
    }
};

TEST_F(SubgraphWithBlockedFormat, smoke_CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    check_results();
}

}  // namespace test
}  // namespace ov
