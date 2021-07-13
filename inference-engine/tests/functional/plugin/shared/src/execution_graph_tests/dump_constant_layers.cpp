// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <execution_graph_tests/dump_constant_layers.hpp>
#include "functional_test_utils/skip_tests_config.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <inference_engine.hpp>

namespace ExecutionGraphTests {

std::string ExecGraphDumpConstantLayers::getTestCaseName(testing::TestParamInfo<std::string> obj) {
    std::ostringstream result;
    std::string targetDevice = obj.param;
    result << "TargetDevice=" << targetDevice;
    return result.str();
}

inline std::pair<std::vector<std::shared_ptr<ngraph::Node>>, std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>>
                                                generateExecGraph(const std::string dName) {
    auto deviceName = dName;
    ngraph::Shape shape = {3, 2};
    ngraph::element::Type type = ngraph::element::f32;

    using std::make_shared;
    using namespace ngraph::opset3;

    //  const const in1   in2             //
    //     \   |     | \  /                //
    //      mul      |  mul                //
    //       \       |  /                  //
    //        \      sum                   //
    //         \     /                     //
    //           sum                       //
    //             \                       //
    //               \     Const   in3     //
    //                 \      \    /       //
    //                   \   Squeeze       //
    //                     \   /           //
    //                      Sum            //
    //                       |             //
    //                      out            //

    //-----[1]-----//
    auto const1 = make_shared<Constant>(type, shape, 2);
    auto const2 = make_shared<Constant>(type, shape, 3);
    auto mul2   = make_shared<ngraph::op::v1::Multiply>(const1, const2);

    //-----[2]-----//
    auto input1 = make_shared<Parameter>(type, shape);
    auto input2 = make_shared<Parameter>(type, shape);
    auto mul1   = make_shared<ngraph::op::v1::Multiply>(input1, input2);
    auto sum1   = make_shared<ngraph::op::v1::Add>(mul1, input1);
    auto sum2   = make_shared<ngraph::op::v1::Add>(mul2, sum1);

    //-----[3]-----//
    auto input3 = make_shared<Parameter>(type, ngraph::Shape{1, 1, 3, 2});
    auto const3 = make_shared<Constant>(type, ngraph::Shape{2}, std::vector<int64_t>{0, 1});
    auto squeeze = make_shared<Squeeze>(input3, const3);
    auto sum3   = make_shared<ngraph::op::v1::Add>(sum2, squeeze);

    auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector {sum3},
            ngraph::ParameterVector{input1, input2, input3},
            "SimpleNet");

    auto ie  = InferenceEngine::Core();
    auto net = InferenceEngine::CNNNetwork(function);
    auto execNet   = ie.LoadNetwork(net, deviceName);
    auto execGraph = execNet.GetExecGraphInfo();
    auto execOps   = execGraph.getFunction()->get_ops();

    std::pair<std::vector<std::shared_ptr<ngraph::Node>>,
              std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>>
              result = {execOps, execNet.CreateInferRequest().GetPerformanceCounts()};

    return result;
}

TEST_P(ExecGraphDumpConstantLayers, DumpConstantLayers) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    auto result = generateExecGraph(this->GetParam());
    bool dumpCheckConst = true;
    bool performanceCheckConst = true;

#ifdef CPU_DEBUG_CAPS
    if (strcmp(std::getenv("OV_CPU_DUMP_CONSTANT_NODES"), "YES") == 0) {
        dumpCheckConst = !dumpCheckConst;
        performanceCheckConst = !performanceCheckConst;
    }
#endif

    auto execOps = result.first;
    for (auto &node : execOps) {
        auto var = node->get_rt_info()["layerType"];
        auto val = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(var);
        if (val->get() == "Const") {
            dumpCheckConst = !dumpCheckConst;
            break;
        }
    }

    auto performanceMap = result.second;
    for (const auto& it : performanceMap) {
        if ((it.first.find("Constant_") != std::string::npos) || (std::string(it.second.layer_type) == "Constant")) {
            performanceCheckConst = !performanceCheckConst;
            break;
        }
    }
    ASSERT_TRUE(dumpCheckConst && performanceCheckConst);
}

} // namespace ExecutionGraphTests
