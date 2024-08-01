// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
//  These tests are designed for correctness of reshape's in-place implementation.
/*
 * Case 1:
 * Subgraph
 *
 *         params[0]   params[1]
 *             |          |
 * constant  shapeOf     /
 *      \      |        /
 *       broadcast     /
 *            \       /
 *             \     /
 *             reshape
 *                |
 *              result
 *
 
 *
 *  Due to non-const target shape parameter (params[1]), reshape node
 *  is non-constant node even though the input tensor is constant node.
 *
 *  some logic protecting constant data from being corrupted by
 *  the in-place consumer may breaks the in-place assumption, and reshape
 *  should be able to handle this case correctly.
 */

class InPlaceReshapeFromConstantCheck : public SubgraphBaseTest {
protected:
    void SetUp() override {
        const auto rtPrc = ov::element::f32;
        const ov::Shape inpShape = {21660, 4};
        const ov::Shape secShape = {4};
        targetStaticShapes = {{inpShape, secShape}};
        targetDevice = ov::test::utils::DEVICE_CPU;
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(rtPrc, inpShape),
                                   std::make_shared<ov::op::v0::Parameter>(ov::element::i32, secShape)};
        auto shape = std::make_shared<ov::op::v3::ShapeOf>(params[0]);
        auto c = std::make_shared<ov::op::v0::Constant>(rtPrc, ov::Shape{}, std::vector<float>{1.0f});
        auto broadcast = std::make_shared<ov::op::v3::Broadcast>(c, shape);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(broadcast, params[1], false);
        ov::ResultVector results{std::make_shared<ov::op::v0::Result>(reshape->output(0))};
        function = std::make_shared<ov::Model>(results, params, "reshape_check");
    }
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::Tensor tensor;
            if (i == 1) {
                tensor = ov::Tensor{ov::element::i32, targetInputStaticShapes[i]};
                auto inputData = tensor.data<ov::element_type_traits<ov::element::i32>::value_type>();
                const std::vector<unsigned> data = {38, 38, 15, 4};
                for (size_t j = 0lu; j < data.size(); ++j) {
                    inputData[j] = data[j];
                }
            } else {
                if (funcInput.get_element_type().is_real()) {
                    ov::test::utils::InputGenerateData in_data;
                    in_data.start_from = 0;
                    in_data.range = 10;
                    in_data.resolution = 1000;
                    tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i], in_data);
                } else {
                    tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                }
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_F(InPlaceReshapeFromConstantCheck, smoke_CPU_InPlaceReshapeFromConstantCheck) {
    run();
}

/* Case 2:
 * Subgraph
 *
 *         params[0]   params[1]
 *                \     /
 *                 \   /
 *                  add---reshape2---result2
 *                   |
 *                reshape1
 *                   |
 *                  MVN
 *                   |
 *                result1
 *
 *  The same memory is shared between the `result2` input and `MVN` output. The CPU graph inplace memory conflict
 *  resolution logic must prevent `result2` data being rewritten by the MVN node.
 */

class InPlaceReshapeShareInputCheck : public SubgraphBaseTest {
protected:
    void SetUp() override {
        const auto rtPrc = ov::element::f32;
        const ov::Shape inpShape = {1, 16, 16};
        targetStaticShapes = {{inpShape, inpShape}};
        targetDevice = ov::test::utils::DEVICE_CPU;
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(rtPrc, inpShape),
                                   std::make_shared<ov::op::v0::Parameter>(rtPrc, inpShape)};

        auto add = std::make_shared<ov::op::v1::Add>(params[0], params[1]);
        std::vector<int> newShape1 = {1, 1, 16, 16};
        auto targetShape1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, newShape1);
        auto reshape1 = std::make_shared<ov::op::v1::Reshape>(add, targetShape1, false);
        auto mvn = std::make_shared<ov::op::v6::MVN>(reshape1,
                                                     ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {2, 3}),
                                                     true,
                                                     0.1,
                                                     ov::op::MVNEpsMode::INSIDE_SQRT);
        auto res1 = std::make_shared<ov::op::v0::Result>(mvn);

        std::vector<int> newShape2 = {1, 4, 8, 8};
        auto targetShape2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, newShape2);
        auto reshape2 = std::make_shared<ov::op::v1::Reshape>(add, targetShape2, false);

        auto res2 = std::make_shared<ov::op::v0::Result>(reshape2);

        function = std::make_shared<ov::Model>(ov::ResultVector{res1, res2}, params, "reshape_share_input_check");
    }
};

TEST_F(InPlaceReshapeShareInputCheck, smoke_CPU_InPlaceReshapeShareInputCheck) {
    run();
}

}  // namespace test
}  // namespace ov
