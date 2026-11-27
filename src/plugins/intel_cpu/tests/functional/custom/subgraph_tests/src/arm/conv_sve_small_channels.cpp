// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

// On SVE CPUs oneDNN's ACL GEMM convolutions decline OC * IC <= 2048: those shapes must run on oneDNN's SVE kernels,
// not the reference implementation, and shapes ACL accepts must stay on ACL.
using ConvSveSmallChannelsParams = std::tuple<ov::Shape,     // input shape
                                              size_t,        // output channels
                                              size_t,        // kernel size
                                              size_t,        // stride
                                              std::string>;  // expected implementation; empty: any non-reference

class ConvSveSmallChannelsTest : public testing::WithParamInterface<ConvSveSmallChannelsParams>,
                                 virtual public SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvSveSmallChannelsParams>& obj) {
        const auto& [inputShape, outChannels, kernel, stride, expectedImpl] = obj.param;
        std::ostringstream result;
        result << "IS=" << inputShape << "_OC=" << outChannels << "_K=" << kernel << "_S=" << stride
               << "_Impl=" << (expectedImpl.empty() ? "not_ref" : expectedImpl);
        return result.str();
    }

protected:
    void SetUp() override {
        if (!ov::with_cpu_sve()) {
            GTEST_SKIP() << "The CPU has no SVE";
        }
        const auto& [inputShape, outChannels, kernel, stride, expectedImpl] = this->GetParam();
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto pad = static_cast<ptrdiff_t>((kernel - 1) / 2);
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShape);
        auto conv = ov::test::utils::make_convolution(param,
                                                      ov::element::f32,
                                                      {kernel, kernel},
                                                      {stride, stride},
                                                      {pad, pad},
                                                      {pad, pad},
                                                      {1, 1},
                                                      ov::op::PadType::EXPLICIT,
                                                      outChannels);
        auto bias = ov::test::utils::make_constant(ov::element::f32, ov::Shape{1, outChannels, 1, 1});
        auto relu = std::make_shared<ov::op::v0::Relu>(std::make_shared<ov::op::v1::Add>(conv, bias));
        function = std::make_shared<ov::Model>(ov::OutputVector{relu->output(0)},
                                               ov::ParameterVector{param},
                                               "ConvSveSmallChannels");
    }
};

TEST_P(ConvSveSmallChannelsTest, CompareWithRefs) {
    run();

    const auto& expectedImpl = std::get<4>(this->GetParam());
    size_t convolutions = 0;
    for (const auto& node : compiledModel.get_runtime_model()->get_ops()) {
        const auto& rtInfo = node->get_rt_info();
        if (rtInfo.at(ov::exec_model_info::LAYER_TYPE).as<std::string>() != "Convolution") {
            continue;
        }
        convolutions++;
        const auto implType = rtInfo.at(ov::exec_model_info::IMPL_TYPE).as<std::string>();
        if (expectedImpl.empty()) {
            EXPECT_EQ(implType.find("ref"), std::string::npos) << "Convolution runs on " << implType;
        } else {
            EXPECT_EQ(implType, expectedImpl);
        }
    }
    EXPECT_EQ(convolutions, 1U);
}

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_ConvSveSmallChannels,
                         ConvSveSmallChannelsTest,
                         ::testing::Values(
                             // OC * IC = 1024
                             ConvSveSmallChannelsParams{{1, 64, 20, 16}, 16, 1, 1, ""},
                             ConvSveSmallChannelsParams{{1, 64, 20, 16}, 16, 3, 1, ""},
                             // OC * IC = 48, a stride 2 first layer
                             ConvSveSmallChannelsParams{{1, 3, 64, 48}, 16, 3, 2, ""},
                             // OC * IC = 128, an even kernel
                             ConvSveSmallChannelsParams{{1, 16, 20, 16}, 8, 2, 1, ""},
                             // OC * IC = 8192: ACL accepts it
                             ConvSveSmallChannelsParams{{1, 64, 20, 16}, 128, 1, 1, "gemm_acl_f32"}),
                         ConvSveSmallChannelsTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
