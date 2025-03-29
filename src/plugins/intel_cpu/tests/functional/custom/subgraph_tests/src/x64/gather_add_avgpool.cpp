// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

/*
   In cases like: Parameter->Gather->Subgraph->AvgPool when input blob precision is forced to U8.
   there is a precision mismatch between Gather and Subgraph,
   since Gather 'inherits' input precision (U8) and Subgraph works on FP32.
   That affects Subgraph's output layout. In this case Subgraph
   exposes 3 supported descriptors: (nhwc, fp32), (nChw8c, fp32), (nchw, fp32).
   Since none of the descriptors matches parent node (because of precision mismatch)
   the first one (nhwc) is picked instead of nchw. Subgraph's layout also affects
   AvgPool layout and for this node also nhwc is picked instead of more preferable
   nChw8c or nChw16c.
   To address the issue, there is a WA in intel_cpu::Graph::Replicate - we skip propagating
   input's precision if its child has Subgraph consumers.
   Same scenario happens when we have Eltwise instead of Subgraph - to be addressed in #78939.
*/
class GatherAddAvgpool : virtual public SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        inType = ov::element::u8;
        outType = ov::element::f32;
        auto type = ov::element::f32;
        auto param = std::make_shared<ov::op::v0::Parameter>(type, Shape{1, 3, 64, 64});
        auto gather =
            std::make_shared<ov::op::v8::Gather>(param,
                                                 ov::op::v0::Constant::create(element::i32, Shape{3}, {2, 1, 0}),
                                                 ov::op::v0::Constant::create(element::i32, Shape{1}, {1}));
        auto add =
            std::make_shared<ov::op::v1::Add>(gather, ov::op::v0::Constant::create(type, Shape{1, 3, 1, 1}, {3}));
        auto avgpool =
            std::make_shared<ov::op::v1::AvgPool>(add, Strides{1, 1}, Shape{0, 0}, Shape{0, 0}, Shape{2, 2}, false);
        function = std::make_shared<ov::Model>(avgpool, ov::ParameterVector{param});
    }

    void check_results() {
        auto exec_model = compiledModel.get_runtime_model();

        int eltwise_nodes_found = 0;
        int pool_nodes_found = 0;
        for (const auto& n : exec_model->get_ordered_ops()) {
            auto layer_type = n->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            auto output_layout = n->get_rt_info().at(ov::exec_model_info::OUTPUT_LAYOUTS).as<std::string>();
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
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    run();
    check_results();
}

}  // namespace test
}  // namespace ov
