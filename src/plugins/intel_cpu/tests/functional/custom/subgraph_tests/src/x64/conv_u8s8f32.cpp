// Copyright (C) 2018-2026 Intel Corporation
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
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include "openvino/runtime/system_conf.hpp"
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

        function = create_ov_model(netPrecision, params, conv, "Convolution");
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

/* Verify AVX2 INT8 saturation protection (issue #38972).
   When full 8-bit weights outside [-64, 63] are executed on pure AVX2 (non-VNNI),
   vpmaddubsw intermediate sums can saturate to 32767 (positive) or -32768 (negative).
   LPT callback must ensure accurate execution (falling back to FP32 on pre-VNNI AVX2).
   Safe weights within [-64, 63] must continue using optimized INT8 execution. */
class ConvU8I8AVX2Base : public testing::WithParamInterface<CPUSpecificParams>,
                         virtual public SubgraphBaseStaticTest,
                         public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<CPUSpecificParams>& obj) {
        std::ostringstream result;
        result << "CPU_" << CPUTestsBase::getTestCaseName(obj.param);
        return result.str();
    }

    void buildGraph(const std::vector<float>& weightValues, const std::string& modelName) {
        targetDevice = ov::test::utils::DEVICE_CPU;
        std::tie(inFmts, outFmts, priority, selectedType) = this->GetParam();

        ov::element::Type netPrecision = ov::element::f32;

        auto make_i8_fake_quantize = [&](std::shared_ptr<ov::Node> input, ov::element::Type dataType) {
            return ov::test::utils::make_fake_quantize(input, dataType, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
        };

        auto make_u8_fake_quantize = [&](std::shared_ptr<ov::Node> input, ov::element::Type dataType) {
            return ov::test::utils::make_fake_quantize(input, dataType, 256, {}, {0.0f}, {2.55f}, {0.0f}, {2.55f});
        };

        // 2 input channels, 1 output channel, 1x1 kernel
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netPrecision, ov::Shape{1, 2, 4, 4})};
        auto fq_input = make_u8_fake_quantize(params[0], netPrecision);

        auto weights = ov::op::v0::Constant::create(netPrecision, ov::Shape{1, 2, 1, 1}, weightValues);
        auto fq_weights = make_i8_fake_quantize(weights, netPrecision);

        auto conv = std::make_shared<ov::op::v1::Convolution>(fq_input,
                                                              fq_weights,
                                                              Strides{1, 1},
                                                              CoordinateDiff{0, 0},
                                                              CoordinateDiff{0, 0},
                                                              Strides{1, 1},
                                                              ov::op::PadType::SAME_UPPER);

        function = create_ov_model(netPrecision, params, conv, modelName);
    }
};

class ConvU8I8AVX2PositiveSaturation : public ConvU8I8AVX2Base {
public:
    void SetUp() override {
        buildGraph({1.27f, 1.27f}, "ConvolutionAVX2PositiveSaturation");
    }
};

TEST_P(ConvU8I8AVX2PositiveSaturation, smoke_CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Convolution");
}

class ConvU8I8AVX2NegativeSaturation : public ConvU8I8AVX2Base {
public:
    void SetUp() override {
        buildGraph({-1.28f, -1.28f}, "ConvolutionAVX2NegativeSaturation");
    }
};

TEST_P(ConvU8I8AVX2NegativeSaturation, smoke_CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Convolution");
}

class ConvU8I8AVX2SafeWeights : public ConvU8I8AVX2Base {
public:
    void SetUp() override {
        buildGraph({0.63f, -0.64f}, "ConvolutionAVX2SafeWeights");
    }
};

TEST_P(ConvU8I8AVX2SafeWeights, smoke_CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "Convolution");
}

namespace {
bool is_vnni_supported() {
    auto is_isa_capped_to_avx2 = [](const char* env_var_name) {
        const char* val = std::getenv(env_var_name);
        if (!val)
            return false;
        std::string s(val);
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return s == "avx2" || s == "avx" || s == "sse41";
    };
    if (is_isa_capped_to_avx2("ONEDNN_MAX_CPU_ISA") || is_isa_capped_to_avx2("DNNL_MAX_CPU_ISA")) {
        return false;
    }
    return ov::with_cpu_x86_avx2_vnni() || ov::with_cpu_x86_avx512_core_vnni() || ov::with_cpu_x86_avx512_core_amx_int8();
}

const auto unsafeParams = []() {
    const bool has_vnni = is_vnni_supported();
    return filterCPUInfo({CPUSpecificParams{{}, {}, {"jit_avx2"}, has_vnni ? "jit_avx2.*_I8" : "jit_avx2.*_FP32"}});
}();

const auto safeParams = []() {
    return filterCPUInfo({CPUSpecificParams{{}, {}, {"jit_avx2"}, "jit_avx2.*_I8"}});
}();
}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_ConvAVX2PositiveSaturation,
    ConvU8I8AVX2PositiveSaturation,
    ::testing::ValuesIn(unsafeParams),
    ConvU8I8AVX2PositiveSaturation::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_ConvAVX2NegativeSaturation,
    ConvU8I8AVX2NegativeSaturation,
    ::testing::ValuesIn(unsafeParams),
    ConvU8I8AVX2NegativeSaturation::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_ConvAVX2SafeWeights,
    ConvU8I8AVX2SafeWeights,
    ::testing::ValuesIn(safeParams),
    ConvU8I8AVX2SafeWeights::getTestCaseName);

}  // namespace test
}  // namespace ov
