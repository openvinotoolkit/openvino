// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/partial_shape.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include <common_test_utils/ov_tensor_utils.hpp>

/*
The main purpose of the tests is to test cyclic inplace resolution in order to make sure that output edges are referenced whenever possible.
*/
// using namespace CPUTestUtils;
namespace ov {
namespace test {

using VectorShapes = std::vector<InputShape>;

class InplaceSubgraph : virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<VectorShapes> obj) {
        VectorShapes& inputShapes = obj.param;

        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << ov::test::utils::vec2str(itr);
                }
            }
            result << ")";
        }
        return result.str();
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        ov::ParameterVector params;
        params.push_back(std::make_shared<ov::op::v0::Parameter>(precision, PartialShape{2, 6}));
        const auto& param = params[0];
        // std::pair<ov::PartialShape, std::vector<ov::Shape>>;
        const std::vector<InputShape> shapes{
            std::pair<ov::PartialShape, std::vector<ov::Shape>> {
                PartialShape{2, 6}, {ov::Shape{2, 6}}
            }
        };
        init_input_shapes(shapes);

        const PartialShape multiply_const_shape{2, 6};

        auto mul_tensor = ov::test::utils::create_and_fill_tensor(precision, multiply_const_shape.to_shape());
        auto multiply_const_input = std::make_shared<ov::op::v0::Constant>(mul_tensor);
        auto add_tensor = ov::test::utils::create_and_fill_tensor(precision, multiply_const_shape.to_shape());
        auto add_const_input = std::make_shared<ov::op::v0::Constant>(add_tensor);

        auto multiply_1 = std::make_shared<ov::op::v1::Multiply>(param, multiply_const_input);
        auto add = std::make_shared<ov::op::v1::Add>(multiply_1, add_const_input);
        auto multiply_2 = std::make_shared<ov::op::v1::Divide>(add, multiply_const_input);
        auto add_2 = std::make_shared<ov::op::v1::Add>(multiply_2, multiply_const_input);
        auto mamtmul_input = std::make_shared<ov::op::v0::Constant>(ov::test::utils::create_and_fill_tensor(precision, Shape{4, 6}));
        auto matmul = std::make_shared<ov::op::v0::MatMul>(add_2, mamtmul_input, false, true);

        auto result_0 = std::make_shared<ov::op::v0::Result>(matmul);  // Ho [batch_size, num_directions, hidden_size]

        function = std::make_shared<ov::Model>(ov::ResultVector{result_0}, params, "Subgraph0");
    }

protected:
    const ov::element::Type precision = ov::element::f32;
};

TEST_F(InplaceSubgraph, smoke_CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov
