// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/node_builders/group_convolution.hpp"
#include "openvino/op/add.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/fusing_test_utils.hpp"

namespace ov::test {

using ConvDWConvTestParams = std::tuple<element::Type_t,
                                        InputShape,
                                        size_t,                             // Output channels
                                        CPUTestUtils::fusingSpecificParams
                                        >;

class ConvDWConv : public testing::WithParamInterface<ConvDWConvTestParams>,
                   public SubgraphBaseTest,
                   public CPUTestUtils::CpuTestWithFusing {
protected:
    void SetUp() override {
        targetDevice = utils::DEVICE_CPU;
        const auto& [precision, input_shape, out_channels, fusing_params] = this->GetParam();
        std::tie(postOpMgrPtr, fusedOps) = fusing_params;

        init_input_shapes({input_shape});

        ov::ParameterVector params;
        for (auto&& shape : inputDynamicShapes) {
            params.push_back(std::make_shared<op::v0::Parameter>(precision, shape));
        }
        const size_t in_channels = targetStaticShapes[0][0][1];
        auto conv_weights = utils::make_constant(precision, std::vector<size_t>{out_channels, in_channels, 1, 1});
        auto conv = utils::make_convolution(params[0],
                                            conv_weights,
                                            precision,
                                            std::vector<size_t>{1, 1},
                                            std::vector<size_t>{1, 1},
                                            ov::CoordinateDiff{0, 0},
                                            ov::CoordinateDiff{0, 0},
                                            std::vector<size_t>{1, 1},
                                            ov::op::PadType::EXPLICIT,
                                            out_channels,
                                            true);

        auto dw_conv_weights = utils::make_constant(precision, std::vector<size_t>{out_channels, 1, 1, 3, 3});
        auto dw_conv = utils::make_group_convolution(conv,
                                                     dw_conv_weights,
                                                     precision,
                                                     std::vector<size_t>{1, 1},
                                                     ov::CoordinateDiff{1, 1},
                                                     ov::CoordinateDiff{1, 1},
                                                     std::vector<size_t>{1, 1},
                                                     ov::op::PadType::EXPLICIT);
        auto bias_const = utils::make_constant(precision, {1, out_channels, 1, 1});
        auto bias = std::make_shared<ov::op::v1::Add>(dw_conv, bias_const);

        function = create_ov_model(precision, params, bias, "ConvDWConv");
    }

public:
    static std::string get_test_case_name(const testing::TestParamInfo<ConvDWConvTestParams>& obj) {
        const auto& [precision, input_shape, out_channels, fusing_params] = obj.param;

        std::ostringstream result;

        result << "IS=";
        result << utils::partialShape2str({input_shape.first});
        result << "_TS=(";
        const int64_t last = static_cast<int64_t>(input_shape.second.size()) - 1L;
        for (int64_t i = 0; i < last; i++) {
            result << utils::vec2str(input_shape.second) << "_";
        }
        if (last >= 0L) {
            result << utils::vec2str(input_shape.second[last]);
        }

        result << ")_OC=" << out_channels;
        result << "_InPrc=" << precision;

        result << CpuTestWithFusing::getTestCaseName(fusing_params);

        return result.str();
    }

    void check_fusing(ov::CompiledModel& exec_network) const {
        if (!exec_network || !ov::with_cpu_x86_avx2() || ov::with_cpu_x86_avx512_core()) {
            return;
        }

        auto exec_graph = exec_network.get_runtime_model();
        ASSERT_NE(nullptr, exec_graph);

        uint8_t conv_num = 0U;
        auto get_exec_value = [](const std::string& param_name, const RTMap& rt_info) -> std::string {
            auto it = rt_info.find(param_name);
            OPENVINO_ASSERT(rt_info.end() != it);
            return it->second.as<std::string>();
        };
        for (const auto& node : exec_graph->get_ops()) {
            if (get_exec_value(ov::exec_model_info::LAYER_TYPE, node->get_rt_info()) == "Convolution") {
                conv_num++;
                ASSERT_EQ(5, node->inputs().size());
            }
        }

        ASSERT_TRUE(conv_num == 1U) << "Only one convolution node is expected after fusing.";
    }
};

TEST_P(ConvDWConv, CompareWithRefs) {
    run();
    check_fusing(compiledModel);
}

static const std::vector<InputShape> input_shape = {{{}, {{1, 24, 240, 320}}}};  // Only static shapes are supported by this fusing.

static const std::vector<CPUTestUtils::fusingSpecificParams> fusing_params_set = {
        CPUTestUtils::emptyFusingSpec,
        CPUTestUtils::fusingPRelu1D,
        CPUTestUtils::fusingFQPerChannelSigmoidFQPerChannel
    };

INSTANTIATE_TEST_SUITE_P(smoke_,
                         ConvDWConv,
                         ::testing::Combine(::testing::Values(element::f32),
                                            ::testing::ValuesIn(input_shape),
                                            ::testing::Values(40),
                                            ::testing::ValuesIn(fusing_params_set)),
                         ConvDWConv::get_test_case_name);

}  // namespace ov::test
