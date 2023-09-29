// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/ov_tensor_utils.hpp>
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

/*This test runs the following subgraph:
    param1(input)  param2(K)
        |             |
        |          Multiply(simulates K calculation)
        \           /
         \         /
          \       /
            Top_K
              |
              |
            Result

The main purpose of this test is triggering the code path when the K value is not only a parameter,
but a variable calculated inside the model
*/

using namespace InferenceEngine;
using namespace ov::test;

namespace SubgraphTestsDefinitions {

class TopKVariableK : public SubgraphBaseTest {
public:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        const ov::Shape inpShape = {10, 6};
        const ov::Shape kShape = {};
        targetStaticShapes = {{inpShape, kShape}};

        ov::ParameterVector input_params;
        input_params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inpShape));
        input_params.push_back(std::make_shared<ov::op::v0::Parameter>(ov::element::i64, kShape));

        input_params[0]->set_friendly_name("Param_0");
        input_params[1]->set_friendly_name("Param_K");

        auto k_multiplier = ngraph::builder::makeConstant<int64_t>(ov::element::i64, {}, {-2});

        auto multiply = ngraph::builder::makeEltwise(input_params[1], k_multiplier, utils::EltwiseTypes::MULTIPLY);
        auto mode = ov::op::TopKMode::MAX;
        auto sort = ov::op::TopKSortType::SORT_VALUES;
        auto topk =
            std::make_shared<ov::op::v11::TopK>(input_params[0], multiply, 0, mode, sort, ElementType::i32, false);

        ngraph::ResultVector results;
        for (size_t i = 0; i < topk->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(topk->output(i)));
        }

        function = std::make_shared<ngraph::Function>(results, input_params, "TopK");
    }
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();
        for (size_t i = 0; i < funcInputs.size(); ++i) {
            const auto& funcInput = funcInputs[i];
            ov::runtime::Tensor tensor;
            if (i == 1) {
                tensor = ov::runtime::Tensor{ov::element::i64, targetInputStaticShapes[i]};
                auto inputData = tensor.data<ov::element_type_traits<ov::element::i64>::value_type>();
                inputData[0] = -2;
            } else {
                if (funcInput.get_element_type().is_real()) {
                    tensor = utils::create_and_fill_tensor(funcInput.get_element_type(),
                                                           targetInputStaticShapes[i],
                                                           10,
                                                           0,
                                                           1000);
                } else {
                    tensor = utils::create_and_fill_tensor(funcInput.get_element_type(), targetInputStaticShapes[i]);
                }
            }
            inputs.insert({funcInput.get_node_shared_ptr(), tensor});
        }
    }
};

TEST_F(TopKVariableK, smoke_TopK_Variable_K) {
    constexpr size_t iter_num = 10;
    for (size_t i = 0; i < iter_num; ++i) {
        run();
    }
}
} // namespace SubgraphTestsDefinitions