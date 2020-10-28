// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <fstream>

#include "functional_test_utils/layer_test_utils.hpp"
#include "ie_system_conf.h"

#include <ngraph/ngraph.hpp>
#include <legacy/ngraph_ops/fully_connected.hpp>

namespace LayerTestsDefinitions {

using InferenceEngine::Precision;
using InferenceEngine::SizeVector;

class MemoryConv : public testing::WithParamInterface<LayerTestsUtils::basicParams>,
                   public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<LayerTestsUtils::basicParams> obj) {
        Precision netPrecision;
        SizeVector inputShapes, newInputShapes;
        std::string targetDevice;
        std::tie(netPrecision, inputShapes, targetDevice) = obj.param;

        std::ostringstream result;
        result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        result << "netPRC=" << netPrecision.name() << "_";
        result << "targetDevice=" << targetDevice;
        return result.str();
    }

protected:
    void SetUp() {
        SizeVector ie_shape;
        std::tie(inPrc, ie_shape, targetDevice) = this->GetParam();

        using namespace ngraph;
        using std::make_shared;

        Shape shape = ie_shape;
        size_t C = shape[1];
        element::Type type = ngraph::element::f32;

        auto input = make_shared<op::v0::Parameter>(type, shape);
        auto mem_i = make_shared<op::v0::Constant>(type, shape, 0);
        auto mem_r = make_shared<op::v3::ReadValue>(mem_i, "id");

        auto mul = make_shared<op::v0::Multiply>(mem_r, input);
        auto sig = make_shared<op::v0::Sigmoid>(mul);

        auto fc1_w = make_shared<op::v0::Constant>(type, Shape{C, C}, 1);
        auto fc1_b = make_shared<op::v0::Constant>(type, Shape{C}, 1);
        auto fc1 = make_shared<op::FullyConnected>(sig, fc1_w, fc1_b, shape);

        auto fc2_w = make_shared<op::v0::Constant>(type, Shape{C, C}, 1);
        auto fc2_b = make_shared<op::v0::Constant>(type, Shape{C}, 1);
        auto fc2 = make_shared<op::FullyConnected>(fc1, fc2_w, fc2_b, shape);

        auto mem_w = make_shared<op::v3::Assign>(fc1, "id");

        // WA. Limitation of ngraph. control_dependency are required.
        mem_w->add_control_dependency(mem_r);
        fc2->add_control_dependency(mem_w);

        function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector      {fc2},
                ngraph::ParameterVector {input},
                "SimpleNet");
    }
};

TEST_P(MemoryConv, CheckTypeConversion) {
    if (!InferenceEngine::with_cpu_x86_bfloat16())
        GTEST_SKIP();

    auto ie = PluginCache::get().ie();
    auto net = InferenceEngine::CNNNetwork(function);
    auto exe_net = ie->LoadNetwork(net, "CPU");
    auto inf_reg = exe_net.CreateInferRequest();

    // check data type via exec graph
    auto exec_graph = exe_net.GetExecGraphInfo();
    auto exec_ops = exec_graph.getFunction()->get_ops();
    std::shared_ptr<ngraph::Node> mem_r, mem_w;

    for (auto &node : exec_ops) {
        auto var = node->get_rt_info()["layerType"];
        auto s_val = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(var);
        if (s_val->get() == "MemoryOutput")
            mem_w = node;
        if (s_val->get() == "MemoryInput")
            mem_r = node;
    }

    ASSERT_NE(nullptr, mem_r);
    ASSERT_EQ(ngraph::element::bf16, mem_r->output(0).get_element_type());

    ASSERT_NE(nullptr, mem_w);
    ASSERT_EQ(ngraph::element::bf16, mem_w->input(0).get_element_type());
}

INSTANTIATE_TEST_CASE_P(smoke_CPU, MemoryConv,
                        ::testing::Combine(
                                ::testing::Values<Precision>(Precision::BF16, Precision::FP32),
                                ::testing::Values(SizeVector{1, 200}),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        MemoryConv::getTestCaseName);

}  // namespace LayerTestsDefinitions
