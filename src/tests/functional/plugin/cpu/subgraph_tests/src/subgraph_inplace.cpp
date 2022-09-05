// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils/cpu_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/builders.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

class SubgraphInplaceTest : virtual public ov::test::SubgraphBaseTest, public CPUTestsBase {
protected:
    void SetUp() override {
        ov::Shape inputShape = {900, 1, 256};
        std::vector<size_t> weightsShape_0 = {1, 1, 256};
        std::vector<size_t> weightsShape_1 = {256, 256};
        std::vector<float> reduce_all = {0, 1, 2};

        ov::element::Type e_type = ov::element::f32;
        targetDevice = CommonTestUtils::DEVICE_CPU;

        init_input_shapes(ov::test::static_shapes_to_test_representation({inputShape}));
        auto parameters = ngraph::builder::makeParams(e_type, {inputShape});

        auto mvn = ngraph::builder::makeMVN(parameters[0], true, true, 9.9e-6);

        std::vector<float> data;
        auto weights_0 = ngraph::builder::makeConstant(e_type, weightsShape_0, data, true);
        auto add = ngraph::builder::makeEltwise(mvn, weights_0, ngraph::helpers::EltwiseTypes::ADD);
        auto hswish = ngraph::builder::makeActivation(add, e_type, ngraph::helpers::ActivationTypes::HSwish);

        auto weights_1 = ngraph::builder::makeConstant(e_type, weightsShape_1, data, true);
        auto mm = ngraph::builder::makeMatMul(add, weights_1);

        auto reduce_min_axes = ngraph::builder::makeConstant(ov::element::i32, {3}, reduce_all);
        auto reduce_min = ngraph::builder::makeReduce(add, reduce_min_axes, false, ngraph::helpers::ReductionType::Min);

        auto reduce_max_axes = ngraph::builder::makeConstant(ov::element::i32, {3}, reduce_all);
        auto reduce_max = ngraph::builder::makeReduce(hswish, reduce_max_axes, false, ngraph::helpers::ReductionType::Max);

        function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{mm, reduce_min, add, reduce_max}, parameters, "SubgraphInplace");
    }
};

namespace  {

/* Disable inplace if Subgraph has several outputs.
              MVN                                              MVN
               |                                                |
        Add + Constant                   ->                  Subgraph
       /       |       \      \                      /       |      |       \
   MatMul ReduceMean ReduceMin HSwish              MatMul ReduceMean ReduceMin ReduceMax
                                |
                              ReduceMax
*/
TEST_F(SubgraphInplaceTest, smoke_SubgraphInplace_CPU) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    run();
    CheckNumberOfNodesWithType(compiledModel, "Subgraph", 1);
}
} // namespace
} // namespace SubgraphTestsDefinitions
