// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

/* Verify simple quantized convolution subgraph.
   No reference implementations are expected to be used.

       Param1
         |
       FQ_U8
         |
       Conv1
         |
       PreLU
         |
      Result

*/

typedef std::tuple<CPUSpecificParams, fusingSpecificParams> ConvU8I8FP32Params;

class ConvU8I8FP32 : public testing::WithParamInterface<ConvU8I8FP32Params>,
                     virtual public SubgraphBaseStaticTest,
                     public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvU8I8FP32Params>& obj) {
        const auto& [cpuParams, fusingParams] = obj.param;
        std::ostringstream result;
        result << "CPU_";
        result << CPUTestsBase::getTestCaseName(cpuParams);
        result << CpuTestWithFusing::getTestCaseName(fusingParams);
        return result.str();
    }

    void SetUp() override {
        const auto& [cpuParams, fusingParams] = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        ov::element::Type netPrecision = ov::element::f32;

        targetDevice = ov::test::utils::DEVICE_CPU;

        auto make_i8_fake_quantize = [&](std::shared_ptr<ov::Node> input, ov::element::Type dataType) {
            return ov::test::utils::make_fake_quantize(input, dataType, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
        };

        auto make_u8_fake_quantize = [&](std::shared_ptr<ov::Node> input, ov::element::Type dataType) {
            return ov::test::utils::make_fake_quantize(input, dataType, 256, {}, {0.0f}, {2.55f}, {0.0f}, {2.55f});
        };

        auto make_quantized_weights = [&make_i8_fake_quantize](const Shape& shape, ov::element::Type dataType) {
            auto weights = ov::op::v0::Constant::create(dataType, shape, std::vector<float>{-0.0512377955019474});
            return make_i8_fake_quantize(weights, dataType);
        };

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netPrecision, ov::Shape{1, 3, 8, 8})};

        auto fq_input = make_u8_fake_quantize(params[0], netPrecision);
        auto fq_weights = make_quantized_weights({3, 3, 4, 4}, netPrecision);

        auto conv = std::make_shared<ov::op::v1::Convolution>(fq_input,
                                                              fq_weights,
                                                              Strides{1, 1},
                                                              CoordinateDiff{0, 0},
                                                              CoordinateDiff{0, 0},
                                                              Strides{1, 1},
                                                              ov::op::PadType::SAME_UPPER);

        auto result = std::make_shared<ov::op::v0::Result>(conv);

        function = makeNgraphFunction(netPrecision, params, conv, "Convolution");
    }
};

TEST_P(ConvU8I8FP32, smoke_CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Convolution");
}

INSTANTIATE_TEST_SUITE_P(
    smoke_Conv,
    ConvU8I8FP32,
    ::testing::Combine(::testing::ValuesIn(filterCPUInfo(
                           {CPUSpecificParams{{}, {}, {"jit_sse42"}, {"jit_sse42_I8"}},  // verify i8 SSE42 just in case
                            CPUSpecificParams{{}, {}, {"jit_avx2"}, {"jit_avx2_I8"}},
                            CPUSpecificParams{{}, {}, {"brgconv_avx512"}, {"brgconv_avx512_I8"}}})),
                       ::testing::Values(fusingPReluPerTensor)),
    ConvU8I8FP32::getTestCaseName);

}  // namespace test
}  // namespace ov
