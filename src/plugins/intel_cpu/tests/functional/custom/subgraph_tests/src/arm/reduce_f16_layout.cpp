// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/precision_support.h"

namespace ov::test {

// Under f16 the ACL Reduce runs Sum/Mean/Prod on f32 descriptors: the Reduce must keep the planar layout of its
// producer, so that the Reorder in front of it only converts the precision and does not transpose the tensor.
class ReduceF16LayoutCPUTest : public SubgraphBaseStaticTest {
protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        configuration.insert({ov::hint::inference_precision.name(), ov::element::f16});
        abs_threshold = 1e-2;

        init_input_shapes(static_shapes_to_test_representation({{1, 64, 16, 16}}));
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputDynamicShapes[0]);
        auto scaled = std::make_shared<ov::op::v1::Multiply>(
            input,
            utils::make_constant(ov::element::f32, ov::Shape{}, std::vector<float>{0.5f}));
        auto mean = std::make_shared<ov::op::v1::ReduceMean>(
            scaled,
            utils::make_constant(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{2, 3}),
            true);
        function = std::make_shared<ov::Model>(ov::OutputVector{mean}, ov::ParameterVector{input}, "ReduceF16Layout");
    }
};

TEST_F(ReduceF16LayoutCPUTest, CompareWithRefs) {
    if (!ov::intel_cpu::hasHardwareSupport(ov::element::f16)) {
        GTEST_SKIP() << "f16 inference precision is not supported on this platform";
    }
    run();
    // up to the Reduce, no node (the Reorder in front of it included) changes the planar layout of the producer. The
    // layout after the Reduce is chosen by its consumer.
    bool reduce_found = false;
    for (const auto& node : compiledModel.get_runtime_model()->get_ordered_ops()) {
        const auto& rt_info = node->get_rt_info();
        const auto type = rt_info.at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
        if (type == "Const") {
            continue;
        }
        EXPECT_EQ(rt_info.at(ov::exec_model_info::OUTPUT_LAYOUTS).as<std::string>(), "abcd") << type;
        if (type == "Reduce") {
            reduce_found = true;
            break;
        }
    }
    EXPECT_TRUE(reduce_found);
}

}  // namespace ov::test
