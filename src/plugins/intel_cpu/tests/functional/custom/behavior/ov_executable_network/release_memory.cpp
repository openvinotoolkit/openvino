// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/runtime/core.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/node_builders/constant.hpp"

using namespace ov::test;

namespace {
class MemoryReleaseTest : public testing::WithParamInterface<bool>, public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<bool>& obj) {
        bool dyn_shapes = obj.param;
        return dyn_shapes ? "dyn_shapes" : "static_shapes";
    }

public:
    void SetUp() override {
        auto net_prc = ov::element::f32;
        targetDevice = utils::DEVICE_CPU;

        bool dyn_shapes = this->GetParam();

        InputShape input_shape;

        if (dyn_shapes) {
            input_shape = {{1, 2048, -1}, {{1, 2048, 7}, {1, 2048, 10}}};
        } else {
            input_shape = {{}, {{1, 2048, 7}}};
        }

        init_input_shapes({input_shape});

        auto param = std::make_shared<ov::opset10::Parameter>(net_prc, inputDynamicShapes.front());

        //convolution params
        static const ov::Shape kernel_1x1 = {1};
        static const ov::Shape kernel_3x3 = {3};
        static const ov::Shape dilations_1x1 = {1};
        static const ov::Shape strides_1x1 = {1};

        static const ov::op::PadType pad_type = ov::op::PadType::EXPLICIT;

        static const std::vector<ptrdiff_t> zero_pads_begin = {0};
        static const std::vector<ptrdiff_t> zero_pads_end = {0};

        static const std::vector<ptrdiff_t> unit_pads_begin = {1};
        static const std::vector<ptrdiff_t> unit_pads_end = {1};

        auto relu0 = std::make_shared<ov::opset10::Relu>(param);

        auto conv1 = utils::make_convolution(relu0,
                                             net_prc,
                                             kernel_1x1,
                                             strides_1x1,
                                             zero_pads_begin,
                                             zero_pads_end,
                                             dilations_1x1,
                                             pad_type,
                                             512,
                                             true);

        auto relu1 = std::make_shared<ov::opset10::Relu>(conv1);

        auto conv2 = utils::make_convolution(relu1,
                                             net_prc,
                                             kernel_3x3,
                                             strides_1x1,
                                             unit_pads_begin,
                                             unit_pads_end,
                                             dilations_1x1,
                                             pad_type,
                                             512,
                                             true);

        auto relu2 = std::make_shared<ov::opset10::Relu>(conv2);

        auto conv3 = utils::make_convolution(relu2,
                                             net_prc,
                                             kernel_1x1,
                                             strides_1x1,
                                             zero_pads_begin,
                                             zero_pads_end,
                                             dilations_1x1,
                                             pad_type,
                                             2048,
                                             true);

        auto add = std::make_shared<ov::opset10::Add>(conv3, relu0);

        auto axis = utils::make_constant(ov::element::i32, {1}, std::vector<int32_t>({2}));

        auto reduce = std::make_shared<ov::opset10::ReduceMean>(add, axis, true);

        function = std::make_shared<ov::Model>(ov::OutputVector{reduce}, ov::ParameterVector{param});
    }
};

TEST_P(MemoryReleaseTest, ConsequitiveRelease) {
    compile_model();
    for (const auto& targetStaticShapeVec : targetStaticShapes) {
        generate_inputs(targetStaticShapeVec);
        validate();
    }
    compiledModel.release_memory();
    for (const auto& targetStaticShapeVec : targetStaticShapes) {
        generate_inputs(targetStaticShapeVec);
        validate();
    }
}

INSTANTIATE_TEST_SUITE_P(smoke_release_memory,
                         MemoryReleaseTest,
                         ::testing::Values(true, false),
                         MemoryReleaseTest::getTestCaseName);

}  // namespace

// TBD:
// a few infer requests one graph
// a few infer request a few graphs
// a few infer request parallel release calls