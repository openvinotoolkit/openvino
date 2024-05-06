// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/runtime/intel_cpu/properties.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/base/utils/generate_inputs.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

struct ShapeRelatedParams {
    std::vector<InputShape> inputShapes;
    std::pair<bool, bool> transpose;
};

typedef std::tuple<
        ShapeRelatedParams,
        ElementType,                        // Input precision
        ElementType,                        // Weights precision
        ElementType,                        // Output precision
        fusingSpecificParams,
        CPUSpecificParams,
        ov::AnyMap, // Additional config
        float                               // Weights sparse rate
> MatMulSparseParamSet;

class MatMulSparseCPUTest : public testing::WithParamInterface<MatMulSparseParamSet>,
                            virtual public SubgraphBaseTest, public CpuTestWithFusing {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MatMulSparseParamSet>& obj) {
        ShapeRelatedParams shapeRelatedParams;
        ElementType inType, weiType, outType;
        fusingSpecificParams fusingParams;
        CPUSpecificParams cpuParams;
        ov::AnyMap additionalConfig;
        float weiSparseRate;
        std::tie(shapeRelatedParams, inType, weiType, outType, fusingParams, cpuParams, additionalConfig,
            weiSparseRate) = obj.param;

        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : shapeRelatedParams.inputShapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : shapeRelatedParams.inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                auto itr = shape.second.begin();
                do {
                    result << ov::test::utils::vec2str(*itr);
                } while (++itr != shape.second.end() && result << "_");
            }
            result << ")_";
        }
        result << "transpose_a=" << shapeRelatedParams.transpose.first << "_";
        result << "transpose_b=" << shapeRelatedParams.transpose.second << "_";
        result << "inType=" << inType << "_";
        result << "weiType=" << weiType << "_";
        result << "outType=" << outType << "_";
        result << CpuTestWithFusing::getTestCaseName(fusingParams);
        result << CPUTestsBase::getTestCaseName(cpuParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                result << "_" << item.first << "=" << item.second.as<std::string>();
            }
        }
        result << "_weiSparseRate=" << weiSparseRate;

        return result.str();
    }

protected:
     std::string cpuNodeType;

    template<typename T>
    void transpose(T& shape) {
        OPENVINO_ASSERT(shape.size() > 1);
        std::swap(*(shape.end() - 1), *(shape.end() - 2));
    }

    std::vector<int8_t> inline generateSparseVector(size_t vec_len,
                float sparseRate = 0.0f,
                int8_t upTo = 10,
                int8_t startFrom = 1,
                int32_t seed = 1) {
        std::vector<int8_t> res(vec_len);
        std::mt19937 gen(seed);
        std::uniform_int_distribution<long> dist(static_cast<long>(startFrom), static_cast<long>(upTo));

        std::mt19937 gen_f(123);
        std::uniform_real_distribution<float> dist_f(0.f, 1.f);

        size_t countZero = 0;

        res[0] = startFrom;
        res[vec_len - 1] = upTo;
        for (size_t i = 1; i < vec_len - 1; i++) {
            if (dist_f(gen_f) > sparseRate) {
                res[i] = static_cast<int8_t>(dist(gen));
            } else {
                res[i] = 0;
                countZero++;
            }
        }

        std::cout << "Sparse rate = " << countZero * 100 / vec_len << "%" << std::endl;

        return res;
    }

    std::shared_ptr<Node> makeMatMulRelaxed(const Output<Node>& A,
                                            const ov::PartialShape& inShapeB,
                                            ElementType weiType,
                                            bool transpose_a,
                                            bool transpose_b,
                                            const std::vector<int8_t>& weiData) {
        auto inputParamsFP32 = std::make_shared<ov::op::v0::Parameter>(element::f32, A.get_partial_shape());
        auto tensor = ov::test::utils::create_and_fill_tensor(element::f32, inShapeB.to_shape());
        auto matrixBFP32 = std::make_shared<ov::op::v0::Constant>(tensor);

        auto matMulRelaxed = std::make_shared<ov::op::TypeRelaxed<ov::op::v0::MatMul>>(
            ov::op::v0::MatMul(inputParamsFP32, matrixBFP32, transpose_a, transpose_b),
            element::f32);

        auto matrixB = std::make_shared<ov::op::v0::Constant>(weiType, inShapeB.get_shape(), weiData);

        auto matMul = matMulRelaxed->copy_with_new_inputs({A, matrixB});

        return matMul;
    }

    void SetUp() override {
        abs_threshold = 0.5f;

        ShapeRelatedParams shapeRelatedParams;
        ElementType inType, weiType, outType;
        fusingSpecificParams fusingParams;
        CPUSpecificParams cpuParams;
        ov::AnyMap additionalConfig;
        float weiSparseRate;

        std::tie(shapeRelatedParams, inType, weiType, outType, fusingParams, cpuParams, additionalConfig,
            weiSparseRate) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        configuration.insert(additionalConfig.begin(), additionalConfig.end());
        targetDevice = ov::test::utils::DEVICE_CPU;

        init_input_shapes(shapeRelatedParams.inputShapes);

        bool transpA = shapeRelatedParams.transpose.first;
        bool transpB = shapeRelatedParams.transpose.second;

        if (transpA) {
            transpose(inputDynamicShapes[0]);
            for (auto& shapes : targetStaticShapes) {
                transpose(shapes[0]);
            }
        }
        if (transpB) {
            transpose(inputDynamicShapes[1]);
            for (auto& shapes : targetStaticShapes) {
                transpose(shapes[1]);
            }
        }

        const auto& inShapeA = inputDynamicShapes[0];
        const auto& inShapeB = inputDynamicShapes[1];

        std::tie(postOpMgrPtr, fusedOps) = fusingParams;

        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        cpuNodeType = "FullyConnected";
        selectedType = makeSelectedTypeStr(selectedType, element::i8);

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(inType, inShapeA)};

        auto tensor = ov::test::utils::create_and_fill_tensor(element::f32, inShapeB.to_shape());
        auto matrixB = std::make_shared<ov::op::v0::Constant>(tensor);

        auto weiData = generateSparseVector(ov::shape_size(inShapeB.get_shape()), weiSparseRate);
        auto matMul = makeMatMulRelaxed(params[0], inShapeB, weiType, transpA, transpB, weiData);

        function = makeNgraphFunction(element::f32, params, matMul, cpuNodeType);

        checkFusingPosition = false;

        functionRefs = function->clone();
        convert_precisions.insert({ov::element::i8, ov::element::f32});
        convert_precisions.insert({ov::element::u8, ov::element::f32});
    }
};

TEST_P(MatMulSparseCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, cpuNodeType);
}

namespace {

/* ============= Common params ============= */

std::vector<CPUSpecificParams> filterSpecificParams(bool sparseExpected) {
    std::vector<CPUSpecificParams> specificParams;
    if (with_cpu_x86_avx512_core_amx()) {
        if (sparseExpected) {
            specificParams.push_back(CPUSpecificParams{{}, {}, {"brgemm_avx512_amx"}, "brgemm_avx512_amx_sparse"});
        } else {
            specificParams.push_back(CPUSpecificParams{{}, {}, {"brgemm_avx512_amx"}, "brgemm_avx512_amx"});
        }
    }

    return specificParams;
}

/* ============= FullyConnected ============= */
namespace fullyConnected {

// cpu (sparse) configs
const ov::AnyMap emptyConfig = {};
const ov::AnyMap SparseRate50 = {{ov::intel_cpu::sparse_weights_decompression_rate(0.5)}};
const ov::AnyMap SparseRate80 = {{ov::intel_cpu::sparse_weights_decompression_rate(0.8)}};

const std::vector<ShapeRelatedParams> IS2D_sparse_smoke = {
    {static_shapes_to_test_representation({{64, 64}, {64, 64}}), {false, true}},
    {static_shapes_to_test_representation({{71, 64}, {64, 64}}), {false, true}},
    {static_shapes_to_test_representation({{3, 128}, {128, 64}}), {false, true}},
    {static_shapes_to_test_representation({{71, 64}, {64, 128}}), {false, true}},

    {
        {
            {{-1, -1}, {{20, 64}, {20, 64}}},
            {{64, 128}, {{64, 128}, {64, 128}}}
        },
        {false, true}
    },

    {
        {
            {{{0, 100}, {0, 64}}, {{20, 64}, {14, 64}, {20, 64}, {14, 64}}},
            {{64, 128}, {{64, 128}, {64, 128}, {64, 128}, {64, 128}}}
        },
        {false, true}
    },
    {static_shapes_to_test_representation({{1, 4096}, {4096, 16384}}), {false, true}},
};

const auto testParams2D_i8_smoke = ::testing::Combine(::testing::ValuesIn(IS2D_sparse_smoke),
                                                   ::testing::Values(ElementType::i8, ElementType::u8),
                                                   ::testing::Values(ElementType::i8),
                                                   ::testing::Values(ElementType::f32),
                                                   ::testing::Values(emptyFusingSpec),
                                                   ::testing::ValuesIn(filterSpecificParams(false)),
                                                   ::testing::Values(emptyConfig, SparseRate80),
                                                   ::testing::Values(0.7));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_I8, MatMulSparseCPUTest, testParams2D_i8_smoke,
    MatMulSparseCPUTest::getTestCaseName);

const auto testParams2D_i8_sparse_smoke = ::testing::Combine(::testing::ValuesIn(IS2D_sparse_smoke),
                                                   ::testing::Values(ElementType::i8, ElementType::u8),
                                                   ::testing::Values(ElementType::i8),
                                                   ::testing::Values(ElementType::f32),
                                                   ::testing::Values(emptyFusingSpec),
                                                   ::testing::ValuesIn(filterSpecificParams(true)),
                                                   ::testing::Values(SparseRate50),
                                                   ::testing::Values(0.7));

INSTANTIATE_TEST_SUITE_P(smoke_FC_2D_I8_sparse, MatMulSparseCPUTest, testParams2D_i8_sparse_smoke,
    MatMulSparseCPUTest::getTestCaseName);

const std::vector<ShapeRelatedParams> IS3D_sparse_smoke = {
    {static_shapes_to_test_representation({{1, 64, 64}, {64, 64}}), {false, true}},
    {static_shapes_to_test_representation({{3, 71, 64}, {64, 64}}), {false, true}},
    {static_shapes_to_test_representation({{3, 5, 128}, {128, 64}}), {false, true}},
    {static_shapes_to_test_representation({{1, 71, 64}, {64, 128}}), {false, true}},

    {
        {
            {{-1, -1, 64}, {{1, 5, 64}, {1, 10, 64}, {1, 5, 64}, {1, 10, 64}}},
            {{64, 128}, {{64, 128}, {64, 128}}}
        },
        {false, true}
    },

    // todo: [av] investigate "Primitive descriptor was not found" error for this case
    // {
    //     {
    //         {{{0, 60}, {0, 60}, {0, 64}}}, {{1, 3, 64}, {1, 7, 64}}},
    //         {{64, 64}, {{64, 64}, {64, 64}}}
    //     },
    //     {false, true}
    // },
};

const auto testParams3D_i8_smoke = ::testing::Combine(::testing::ValuesIn(IS3D_sparse_smoke),
                                                   ::testing::Values(ElementType::i8, ElementType::u8),
                                                   ::testing::Values(ElementType::i8),
                                                   ::testing::Values(ElementType::f32),
                                                   ::testing::Values(emptyFusingSpec),
                                                   ::testing::ValuesIn(filterSpecificParams(false)),
                                                   ::testing::Values(emptyConfig, SparseRate80),
                                                   ::testing::Values(0.7));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D_I8, MatMulSparseCPUTest, testParams3D_i8_smoke,
    MatMulSparseCPUTest::getTestCaseName);

const auto testParams3D_i8_sparse_smoke = ::testing::Combine(::testing::ValuesIn(IS3D_sparse_smoke),
                                                   ::testing::Values(ElementType::i8, ElementType::u8),
                                                   ::testing::Values(ElementType::i8),
                                                   ::testing::Values(ElementType::f32),
                                                   ::testing::Values(emptyFusingSpec),
                                                   ::testing::ValuesIn(filterSpecificParams(true)),
                                                   ::testing::Values(SparseRate50),
                                                   ::testing::Values(0.7));

INSTANTIATE_TEST_SUITE_P(smoke_FC_3D_I8_sparse, MatMulSparseCPUTest, testParams3D_i8_sparse_smoke,
    MatMulSparseCPUTest::getTestCaseName);

} // namespace fullyConnected

} // namespace

}  // namespace test
}  // namespace ov
