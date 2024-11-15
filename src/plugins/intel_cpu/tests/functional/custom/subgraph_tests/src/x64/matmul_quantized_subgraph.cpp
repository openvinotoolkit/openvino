// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

#include <algorithm>

using namespace CPUTestUtils;

namespace ov {
namespace test {

using ElementType = ov::element::Type_t;
using MatmulBrgemmInt8TestParams = std::tuple<ov::Shape,          // input shape
                                              bool,               // true: FullyConnected false: Matmul
                                              bool,               // ture: FullyConnected primitive implement type check
                                              ElementType,        // input u8/s8
                                              ElementType,        // output f32/u8/s8
                                              CPUSpecificParams,  // brgemm/jit primitive implement type
                                              ov::AnyMap          // Additional config
                                              >;

// subgraph:
//   fq->MatMul/FullyConnected->[fq]
// can cover brgemm avx2:
//   (u8/s8 + s8)->f32
//   (u8/s8 + s8)->u8/s8
class MatmulBrgemmInt8Test : public testing::WithParamInterface<MatmulBrgemmInt8TestParams>, public CpuTestWithFusing,
                      virtual public ov::test::SubgraphBaseStaticTest {
public:
    static std::string getTestCaseName(testing::TestParamInfo<MatmulBrgemmInt8TestParams> obj) {
        ov::Shape supportedInputShapes;
        bool isFC;
        bool isFCPrimCheck;
        ov::AnyMap additionalConfig;
        ElementType inType;
        ElementType outType;
        CPUSpecificParams cpuParams;
        std::tie(supportedInputShapes, isFC, isFCPrimCheck, inType, outType, cpuParams, additionalConfig) = obj.param;

        std::ostringstream result;
        result << "IS=" << supportedInputShapes.to_string() << "_";
        result << (isFC ? "FullyConnected" : "MatMul") << "_";
        result << "FCPrimCheck=" << isFCPrimCheck << "_";
        result << "InputType=" << inType << "_";
        result << "OutputType=" << outType << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                if (item.second == ov::element::bf16 || item.second == ov::element::f16)
                    result << "_" << item.first << "=" << item.second.as<std::string>();
            }
        }

        return result.str();
    }

protected:
    bool isFC;
    bool isFCPrimCheck;
    ov::AnyMap additionalConfig;
    std::string nameMatmul = "TestedMatmul";
    ElementType inType;
    ElementType outType;
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        ov::Shape inShapes;
        CPUSpecificParams cpuParams;
        std::tie(inShapes, isFC, isFCPrimCheck, inType, outType, cpuParams, additionalConfig) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        const auto ngPrec = ov::element::f32;
        ov::ParameterVector inputParams {std::make_shared<ov::op::v0::Parameter>(ngPrec, ov::Shape(inShapes))};

        std::shared_ptr<ov::Node> fq1;
        std::shared_ptr<ov::Node> matMul;
        std::shared_ptr<ov::Node> nodeBeforeConv;
        selectedType = makeSelectedTypeStr(selectedType, ElementType::i8);
        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        if (inType == ElementType::u8)
            fq1 = ov::test::utils::make_fake_quantize(inputParams[0], ngPrec, 256, {}, {0.0f}, {2.55f}, {0.0f}, {2.55f});
        else
            fq1 = ov::test::utils::make_fake_quantize(inputParams[0], ngPrec, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});

        if (isFC) {
            ov::Shape weightShape = inShapes;
            std::shared_ptr<ov::Node> biasWeightsNode;
            if (!isFCPrimCheck)
                std::swap(weightShape[0], weightShape[1]);
            else
                std::swap(weightShape[1], weightShape[2]);
            auto weightsNode = ov::test::utils::make_constant(ngPrec, weightShape);
            auto fq2 =
                ov::test::utils::make_fake_quantize(weightsNode, ngPrec, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
            auto fc = std::make_shared<ov::op::v0::MatMul>(fq1, fq2, false, false);
            fc->get_rt_info() = getCPUInfo();
            fc->set_friendly_name(nameMatmul);
            if (!isFCPrimCheck) {
                biasWeightsNode = ov::test::utils::make_constant(ngPrec, ov::Shape{});
            } else {
                auto fcShape = fc->get_output_shape(0);
                ov::Shape biasShape(fcShape.size(), 1);
                biasShape.back() = fcShape.back();
                biasWeightsNode = ov::test::utils::make_constant(ngPrec, biasShape);
            }
            matMul = std::make_shared<ov::op::v1::Add>(fc, biasWeightsNode);
        } else {
            auto fq2 = ov::test::utils::make_fake_quantize(inputParams[0], ngPrec, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
            matMul = std::make_shared<ov::op::v0::MatMul>(fq1, fq2, false, true);
            matMul->get_rt_info() = getCPUInfo();
            matMul->set_friendly_name(nameMatmul);
        }
        if (outType == ElementType::u8)
            nodeBeforeConv = ov::test::utils::make_fake_quantize(matMul, ngPrec, 256, {}, {0.0f}, {2.55f}, {0.0f}, {2.55f});
        else if (outType == ElementType::i8)
            nodeBeforeConv = ov::test::utils::make_fake_quantize(matMul, ngPrec, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
        else
            nodeBeforeConv = matMul;

        // matmul->fq->matmul can cover x8*s8->x8 case
        auto filterWeightsShape = matMul->get_output_shape(0);
        auto filterWeightsNode = ov::test::utils::make_constant(ov::element::f32, filterWeightsShape);
        auto fq3 = ov::test::utils::make_fake_quantize(filterWeightsNode, ngPrec, 256, {}, {-1.28f}, {1.27f}, {-1.28f}, {1.27f});
        // only matmul avx2 support s8*s8 input
        auto matMul2 = std::make_shared<ov::op::v0::MatMul>(nodeBeforeConv, fq3, false, false);

        function = makeNgraphFunction(ngPrec, inputParams, matMul2, "MatmulBrgemmInt8");
    }

    void check_node(std::shared_ptr<const ov::Model> function, const std::string& nodeName) {
        ASSERT_NE(nullptr, function);
        for (const auto &node : function->get_ops()) {
            const auto & rtInfo = node->get_rt_info();
            auto getExecValue = [&rtInfo](const std::string & paramName) -> std::string {
                auto it = rtInfo.find(paramName);
                OPENVINO_ASSERT(rtInfo.end() != it);
                return it->second.as<std::string>();
            };
            if (node->get_friendly_name() == nodeName) {
                auto primType = getExecValue(ov::exec_model_info::IMPL_TYPE);
                ASSERT_TRUE(primTypeCheck(primType)) << "primType is unexpected: " << primType << " Expected: " << selectedType;
                ASSERT_EQ(node->get_output_element_type(0), outType);
                ASSERT_EQ(node->get_input_element_type(0), inType);
            }
        }
    }
};

TEST_P(MatmulBrgemmInt8Test, CompareWithRefs) {
    // matmulBrgemmInt8 only cover avx2_vnni
    if (ov::with_cpu_x86_avx512_core_amx_fp16() && !isFCPrimCheck) {
        GTEST_SKIP();
    }
    if ((ov::with_cpu_x86_avx512_core() || !ov::with_cpu_x86_avx2_vnni()) && !ov::with_cpu_x86_avx512_core_amx_fp16())
        GTEST_SKIP();

    run();
    if (!!compiledModel) {
        auto exec = compiledModel.get_runtime_model();
        check_node(exec, nameMatmul);
    }
}

namespace {

const std::vector<ov::Shape> supportedInputShapes = {
    {16, 32},
    {17, 15},
};

const std::vector<CPUSpecificParams>matmulSpecificFilterParams = {
    {{}, {}, {"brgemm_avx2"}, "brgemm_avx2"},
    {{}, {}, {"jit_gemm"}, "jit_gemm"}
};

INSTANTIATE_TEST_SUITE_P(smoke_matmulBrgemmInt8,
                         MatmulBrgemmInt8Test,
                         ::testing::Combine(::testing::ValuesIn(supportedInputShapes),
                                            ::testing::ValuesIn({true, false}),
                                            ::testing::Values(false),
                                            ::testing::ValuesIn({ElementType::u8, ElementType::i8}),
                                            ::testing::ValuesIn({ElementType::f32, ElementType::u8, ElementType::i8}),
                                            ::testing::ValuesIn(matmulSpecificFilterParams),
                                            ::testing::Values(ov::AnyMap{})),
                         MatmulBrgemmInt8Test::getTestCaseName);

std::vector<ov::AnyMap> additionalConfig = {{ov::hint::inference_precision(ov::element::f32)},
                                            {ov::hint::inference_precision(ov::element::bf16)},
                                            {ov::hint::inference_precision(ov::element::f16)}};

const std::vector<CPUSpecificParams>fullyconnectedSpecificFilterParams = {
    {{}, {}, {"brgemm_avx512_amx"}, "brgemm_avx512_amx"},
    {{}, {}, {"jit_gemm"}, "jit_gemm"}
};

INSTANTIATE_TEST_SUITE_P(smoke_fullyconnected_prim_impl_type_check,
                         MatmulBrgemmInt8Test,
                         ::testing::Combine(::testing::Values(ov::Shape{1, 32, 36}),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(ElementType::u8),
                                            ::testing::Values(ElementType::u8),
                                            ::testing::ValuesIn(fullyconnectedSpecificFilterParams),
                                            ::testing::ValuesIn(additionalConfig)),
                         MatmulBrgemmInt8Test::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
