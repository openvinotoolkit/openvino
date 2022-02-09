// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/layer_test_utils.hpp"
#include <ngraph/opsets/opset8.hpp>
#include <exec_graph_info.hpp>


namespace SubgraphTestsDefinitions {

using namespace ngraph;

class GatherAddAvgpool : virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        inPrc = InferenceEngine::Precision::U8;
        outPrc = InferenceEngine::Precision::FP32;
        auto type = element::f32;
        auto param = std::make_shared<opset8::Parameter>(type, Shape{1, 3, 64, 64});
        auto gather = std::make_shared<opset8::Gather>(param,
                                                       op::Constant::create(element::i32, Shape{3}, {2, 1, 0}),
                                                       op::Constant::create(element::i32, Shape{1}, {1}));
        auto add = std::make_shared<opset8::Add>(gather, op::Constant::create(type, Shape{1, 3, 1, 1}, {3}));
        auto avgpool = std::make_shared<opset8::AvgPool>(add, Strides{1, 1}, Shape{0, 0}, Shape{0, 0}, Shape{2, 2}, false);
        function = std::make_shared<Function>(avgpool, ParameterVector{param});
    }

    void TearDown() override {
        auto exec_model = executableNetwork.GetExecGraphInfo().getFunction();

        int eltwise_nodes_found = 0;
        int pool_nodes_found = 0;
        for (const auto& n : exec_model->get_ordered_ops()) {
            auto layer_type = n->get_rt_info().at(ExecGraphInfoSerialization::LAYER_TYPE).as<std::string>();
            auto output_layout = n->get_rt_info().at(ExecGraphInfoSerialization::OUTPUT_LAYOUTS).as<std::string>();
            if (layer_type == "Subgraph") {
                eltwise_nodes_found++;
                ASSERT_EQ("abcd", output_layout);
            } else if (layer_type == "Pooling") {
                pool_nodes_found++;
                ASSERT_TRUE(output_layout == "aBcd8b" || output_layout == "aBcd16b");
            }
        }
        ASSERT_GT(eltwise_nodes_found, 0);
        ASSERT_GT(pool_nodes_found, 0);
    }
};

TEST_F(GatherAddAvgpool, smoke_CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
}

} // namespace SubgraphTestsDefinitions
