// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <random>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/general_utils.h"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/topk.hpp"

using namespace CPUTestUtils;
using namespace ov::test;
using SortMode = ov::op::TopKMode;
using SortType = ov::op::TopKSortType;

namespace ov {
namespace test {

typedef std::tuple<int64_t,                     // keepK
                   int64_t,                     // axis
                   SortMode,                    // mode
                   std::tuple<SortType, bool>,  // sort and stable
                   ElementType,                 // Net type
                   ElementType,                 // Input type
                   ElementType,                 // Output type
                   InputShape                   // inputShape
                   >
    basicTopKParams;

typedef std::tuple<basicTopKParams, CPUSpecificParams, ov::AnyMap> TopKLayerCPUTestParamsSet;
class TopKLayerCPUTest : public testing::WithParamInterface<TopKLayerCPUTestParamsSet>,
                         virtual public SubgraphBaseTest,
                         public CPUTestsBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TopKLayerCPUTestParamsSet>& obj) {
        const auto &[basicParamsSet, cpuParams, additionalConfig] = obj.param;
        const auto &[keepK, axis, mode, sortTypeStable, netPrecision, inPrc, outPrc, inputShape] = basicParamsSet;
        SortType sort = std::get<0>(sortTypeStable);
        bool stable = std::get<1>(sortTypeStable);

        std::ostringstream result;
        result << "k=" << keepK << "_";
        result << "axis=" << axis << "_";
        result << "mode=" << mode << "_";
        result << "sort=" << sort << "_";
        result << "stable=" << (stable ? "True" : "False") << "_";
        result << "netPRC=" << netPrecision << "_";
        result << "inPRC=" << inPrc << "_";
        result << "outPRC=" << outPrc << "_";
        result << "IS=" << ov::test::utils::partialShape2str({inputShape.first}) << "_"
               << "TS=(";
        for (const auto& shape : inputShape.second) {
            result << ov::test::utils::vec2str(shape) << "_";
        }

        result << CPUTestsBase::getTestCaseName(cpuParams);

        if (!additionalConfig.empty()) {
            result << "_PluginConf";
            for (auto& item : additionalConfig) {
                if (item.second == ov::element::bf16)
                    result << "_" << item.first << "=" << item.second.as<std::string>();
            }
        }

        return result.str();
    }

protected:
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override {
        if (sort != SortType::NONE) {
            SubgraphBaseTest::compare(expected, actual);
            return;
        }

        ASSERT_GE(expected.size(), 2u);
        ASSERT_GE(actual.size(), 2u);
        ASSERT_EQ(expected[0].get_shape(), actual[0].get_shape());
        ASSERT_EQ(expected[1].get_shape(), actual[1].get_shape());
        ASSERT_EQ(expected[0].get_element_type(), actual[0].get_element_type());
        ASSERT_EQ(expected[1].get_element_type(), ov::element::i32);
        ASSERT_EQ(actual[1].get_element_type(), ov::element::i32);

        std::vector<ov::Tensor> exp_sorted = expected;
        std::vector<ov::Tensor> act_sorted = actual;

        sort_topk_pairs(exp_sorted[0], exp_sorted[1]);
        sort_topk_pairs(act_sorted[0], act_sorted[1]);

        SubgraphBaseTest::compare(exp_sorted, act_sorted);
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto &[basicParamsSet, cpuParams, additionalConfig] = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;
        const auto& [keepK, _axis, mode, sortTypeStable, _netPrecision, inPrc, outPrc, inputShape] = basicParamsSet;
        this->keepK = keepK;
        axis = _axis;
        netPrecision = _netPrecision;
        sort = std::get<0>(sortTypeStable);
        stable = std::get<1>(sortTypeStable);

        if (intel_cpu::contains_key_value(additionalConfig, {ov::hint::inference_precision.name(), ov::element::bf16}))
            netPrecision = ElementType::bf16;
        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        const auto primType = getPrimitiveType();
        selectedType = primType == "acl" ? "jit" : primType;
        if (!ov::with_cpu_x86_avx512_core() && netPrecision == ElementType::bf16) {
            updateSelectedType(selectedType, ElementType::f32, configuration);
        } else {
            updateSelectedType(selectedType, netPrecision, configuration);
        }

        staticShape = inputShape.first.rank() == 0;
        if (staticShape) {
            init_input_shapes({inputShape});
        } else {
            inputDynamicShapes = {inputShape.first, {}};
            for (size_t i = 0; i < inputShape.second.size(); ++i) {
                targetStaticShapes.push_back({inputShape.second[i], {}});
            }
        }

        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(netPrecision, inputDynamicShapes[0])};

        // static shape need specific const k to test different sorting algorithms, dynamic shape tests random param k
        std::shared_ptr<ov::op::v11::TopK> topk;
        if (staticShape) {
            auto k = std::make_shared<ov::op::v0::Constant>(ElementType::i64, ov::Shape{}, &keepK);
            topk = ov::as_type_ptr<ov::op::v11::TopK>(
                std::make_shared<ov::op::v11::TopK>(params[0], k, axis, mode, sort, ElementType::i32, stable));
        } else {
            auto k = std::make_shared<ov::op::v0::Parameter>(ElementType::i64, inputDynamicShapes[1]);
            params.push_back(k);
            topk = ov::as_type_ptr<ov::op::v11::TopK>(
                std::make_shared<ov::op::v11::TopK>(params[0], k, axis, mode, sort, ElementType::i32, stable));
        }

        topk->get_rt_info() = getCPUInfo();

        ov::ResultVector results;
        for (size_t i = 0; i < topk->get_output_size(); i++) {
            results.push_back(std::make_shared<ov::op::v0::Result>(topk->output(i)));
        }

        function = std::make_shared<ov::Model>(results, params, "TopK");
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        inputs.clear();
        const auto& funcInputs = function->inputs();

        // For unstable sorting, generate unrepeated input data to avoid a. and b. While for stable sorting,
        // repeating values are explicitly set.
        // a. Skip comparing of index results, because an element in actual index tensor can be different with
        //    its counterpart in expected index tensor
        // b. If SortType is SORT_INDICES or NONE, the test program still needs to apply std::sort for all pairs
        //    of 1xk value vectors in expected and actual output tensor before comparing them
        auto shape = targetInputStaticShapes.front();
        ov::Tensor tensor;
        tensor = ov::test::utils::create_and_fill_tensor(funcInputs[0].get_element_type(), shape);
        size_t size = tensor.get_size();

        if (netPrecision == ElementType::f32 || netPrecision == ElementType::i32) {
            std::vector<int> data(size);

            // For int32, deliberately set big numbers which are not accurately representable in fp32
            int start = netPrecision == ElementType::i32 ? pow(2, 30) + 1 : -static_cast<int>(size / 2);
            size_t set_size = sort == SortType::SORT_VALUES && stable ? size / 2 : size;
            std::iota(data.begin(), data.begin() + set_size, start);
            if (sort == SortType::SORT_VALUES && stable) {
                std::copy(data.begin(), data.begin() + set_size, data.begin() + set_size);
            }
            std::mt19937 gen(0);
            std::shuffle(data.begin(), data.end(), gen);

            if (netPrecision == ElementType::f32) {
                auto* rawBlobDataPtr = static_cast<float*>(tensor.data());
                for (size_t i = 0; i < size; ++i) {
                    rawBlobDataPtr[i] = static_cast<float>(data[i]);
                }
            } else {
                auto* rawBlobDataPtr = static_cast<int32_t*>(tensor.data());
                for (size_t i = 0; i < size; ++i) {
                    rawBlobDataPtr[i] = static_cast<int32_t>(data[i]);
                }
            }
        } else if (netPrecision == ElementType::bf16 || netPrecision == ElementType::f16) {
            size_t O = 1, A = 1, I = 1;
            A = shape[axis];
            for (int64_t i = 0; i < axis; i++)
                O *= shape[i];
            for (size_t i = axis + 1; i < shape.size(); i++)
                I *= shape[i];
            if (O * A * I != size)
                FAIL() << "Incorrect blob shape " << shape;

            const bool stable_values = sort == SortType::SORT_VALUES && stable;
            const size_t set_size = stable_values ? A / 2 : A;

            if (netPrecision == ElementType::bf16) {
                auto* rawBlobDataPtr = static_cast<ov::bfloat16*>(tensor.data());
                for (size_t o = 0; o < O; o++) {
                    for (size_t i = 0; i < I; i++) {
                        std::vector<int> data(A);
                        int start = -static_cast<int>(A / 2);
                        std::iota(data.begin(), data.begin() + set_size, start);
                        if (stable_values) {
                            std::copy(data.begin(), data.begin() + set_size, data.begin() + set_size);
                        }
                        const size_t seed = (o + 1) * (i + 1);
                        std::mt19937 gen(seed);
                        std::shuffle(data.begin(), data.end(), gen);
                        for (size_t a = 0; a < A; a++) {
                            rawBlobDataPtr[o * A * I + a * I + i] = static_cast<ov::bfloat16>(data[a]);
                        }
                    }
                }
            } else {  // f16
                auto* rawBlobDataPtr = static_cast<ov::float16*>(tensor.data());
                for (size_t o = 0; o < O; o++) {
                    for (size_t i = 0; i < I; i++) {
                        std::vector<int> data(A);
                        int start = -static_cast<int>(A / 2);
                        std::iota(data.begin(), data.begin() + set_size, start);
                        if (stable_values) {
                            std::copy(data.begin(), data.begin() + set_size, data.begin() + set_size);
                        }
                        const size_t seed = (o + 1) * (i + 1);
                        std::mt19937 gen(seed);
                        std::shuffle(data.begin(), data.end(), gen);
                        for (size_t a = 0; a < A; a++) {
                            rawBlobDataPtr[o * A * I + a * I + i] = static_cast<ov::float16>(data[a]);
                        }
                    }
                }
            }
        } else {
            FAIL() << "generate_inputs for " << netPrecision << " precision isn't supported";
        }
        inputs.insert({funcInputs[0].get_node_shared_ptr(), tensor});

        if (!staticShape) {
            generate_dynamic_k(funcInputs, targetInputStaticShapes);
        }
    }

private:
    void sort_topk_pairs(ov::Tensor& values, ov::Tensor& indices) const {
        ASSERT_EQ(values.get_shape(), indices.get_shape());

        const auto& shape = values.get_shape();
        const auto axis_u = static_cast<size_t>(axis);
        const size_t k = shape[axis_u];
        size_t outer_dim = 1;
        size_t inner_dim = 1;
        for (size_t i = 0; i < axis_u; ++i) {
            outer_dim *= shape[i];
        }
        for (size_t i = axis_u + 1; i < shape.size(); ++i) {
            inner_dim *= shape[i];
        }

        auto sort_impl = [&](auto* values_data, int32_t* indices_data) {
            using T = std::decay_t<decltype(*values_data)>;
            std::vector<std::pair<T, int32_t>> buf(k);
            for (size_t outer = 0; outer < outer_dim; ++outer) {
                const size_t base_outer = outer * k * inner_dim;
                for (size_t inner = 0; inner < inner_dim; ++inner) {
                    const size_t base = base_outer + inner;
                    for (size_t i = 0; i < k; ++i) {
                        const size_t offset = base + i * inner_dim;
                        buf[i] = {values_data[offset], indices_data[offset]};
                    }
                    std::sort(buf.begin(), buf.end(), [](const auto& lhs, const auto& rhs) {
                        if (lhs.first < rhs.first) {
                            return true;
                        }
                        if (rhs.first < lhs.first) {
                            return false;
                        }
                        return lhs.second < rhs.second;
                    });
                    for (size_t i = 0; i < k; ++i) {
                        const size_t offset = base + i * inner_dim;
                        values_data[offset] = buf[i].first;
                        indices_data[offset] = buf[i].second;
                    }
                }
            }
        };

        auto* idx_data = indices.data<int32_t>();
        switch (values.get_element_type()) {
        case ov::element::Type_t::f16:
            sort_impl(values.data<ov::float16>(), idx_data);
            break;
        case ov::element::Type_t::bf16:
            sort_impl(values.data<ov::bfloat16>(), idx_data);
            break;
        case ov::element::Type_t::f32:
            sort_impl(values.data<float>(), idx_data);
            break;
        case ov::element::Type_t::i32:
            sort_impl(values.data<int32_t>(), idx_data);
            break;
        default:
            FAIL() << "Unsupported tensor element type for TopK compare: " << values.get_element_type();
        }
    }

    void generate_dynamic_k(const std::vector<ov::Output<ov::Node>>& funcInputs,
                            const std::vector<ov::Shape>& targetInputStaticShapes) {
        const auto& kPrecision = funcInputs[1].get_element_type();
        const auto& kShape = targetInputStaticShapes[1];

        ov::test::utils::InputGenerateData in_data;
        const auto axis_dim = targetInputStaticShapes[0][axis];
        if (keepK > static_cast<int64_t>(axis_dim)) {
            FAIL() << "Dynamic k exceeds axis dim: k=" << keepK << " axis_dim=" << axis_dim;
        }
        in_data.start_from = keepK;
        in_data.range = 1;
        in_data.seed = inferRequestNum++;
        const auto kTensor = ov::test::utils::create_and_fill_tensor(kPrecision, kShape, in_data);

        inputs.insert({funcInputs[1].get_node_shared_ptr(), kTensor});
    }

private:
    int64_t axis;
    int64_t keepK = 0;
    SortType sort;
    bool stable;
    size_t inferRequestNum = 0;
    ElementType netPrecision;
    bool staticShape;
};
TEST_P(TopKLayerCPUTest, CompareWithRefs) {
    run();
    CheckPluginRelatedResults(compiledModel, "TopK");
}

namespace {

const std::vector<ElementType> netPrecisions = {
    ElementType::f32,
};

const std::vector<ElementType> netPrecisions_f16 = {
    ElementType::f16,
};

std::vector<ov::AnyMap> additionalConfig = {
    {{ov::hint::inference_precision(ov::element::f32)}},
#if defined(OPENVINO_ARCH_X86_64)
    {{ov::hint::inference_precision(ov::element::bf16)}}
#endif
};

#if defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)
std::vector<ov::AnyMap> additionalConfig_f16 = {
    {{ov::hint::inference_precision(ov::element::f16)}},
};
#else
std::vector<ov::AnyMap> additionalConfig_f16 = {{}};
#endif

const std::vector<int64_t> axes = {0, 1, 2, 3};
const std::vector<int64_t> k = {1, 5, 7, 18, 21};

const std::vector<SortMode> modes = {SortMode::MIN, SortMode::MAX};

const std::vector<std::tuple<SortType, bool>> sortTypeStable = {
    std::tuple<SortType, bool>{SortType::SORT_VALUES, false},
    std::tuple<SortType, bool>{SortType::SORT_VALUES, true},
    std::tuple<SortType, bool>{SortType::SORT_INDICES, false},
    std::tuple<SortType, bool>{SortType::NONE, false}};

std::vector<ov::test::InputShape> inputShapes = {
    {{}, {{21, 21, 21, 21}}},
};

std::vector<ov::test::InputShape> inputShapesDynamic = {
    {{21, {20, 25}, 21, {20, 25}}, {{21, 21, 21, 21}, {21, 22, 21, 23}}}};

const std::vector<int64_t> k_dynamic = {1, 5, 7, 18};

const std::vector<int64_t> k_vl_tail = {1, 3};
const std::vector<int64_t> axes_vl_tail = {3};
const std::vector<std::tuple<SortType, bool>> sortTypeStable_vl_tail = {
    std::tuple<SortType, bool>{SortType::SORT_VALUES, false},
    std::tuple<SortType, bool>{SortType::NONE, false},
};

std::vector<ov::test::InputShape> inputShapes_vl_tail = {
    {{}, {{1, 1, 1, 3}}},
    {{}, {{1, 1, 1, 4}}},
    {{}, {{1, 1, 1, 5}}},
    {{}, {{1, 1, 1, 8}}},
    {{}, {{1, 1, 1, 9}}},
    {{}, {{1, 1, 1, 16}}},
    {{}, {{1, 1, 1, 17}}},
};

std::vector<CPUSpecificParams> cpuParams_vl_tail = {CPUSpecificParams({nchw, x}, {nchw, nchw}, {}, {})};

std::vector<CPUSpecificParams> cpuParams = {CPUSpecificParams({nChw16c, x}, {nChw16c, nChw16c}, {}, {}),
                                            CPUSpecificParams({nChw8c, x}, {nChw8c, nChw8c}, {}, {}),
                                            CPUSpecificParams({nchw, x}, {nchw, nchw}, {}, {}),
                                            CPUSpecificParams({nhwc, x}, {nhwc, nhwc}, {}, {})};

INSTANTIATE_TEST_SUITE_P(smoke_TopK,
                         TopKLayerCPUTest,
                         ::testing::Combine(::testing::Combine(::testing::ValuesIn(k),
                                                               ::testing::ValuesIn(axes),
                                                               ::testing::ValuesIn(modes),
                                                               ::testing::ValuesIn(sortTypeStable),
                                                               ::testing::ValuesIn(netPrecisions),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::ValuesIn(inputShapes)),
                                            ::testing::ValuesIn(filterCPUSpecificParams(cpuParams)),
                                            ::testing::ValuesIn(additionalConfig)),
                         TopKLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TopK_dynamic,
                         TopKLayerCPUTest,
                         ::testing::Combine(::testing::Combine(::testing::ValuesIn(k_dynamic),
                                                               ::testing::ValuesIn(axes),
                                                               ::testing::ValuesIn(modes),
                                                               ::testing::ValuesIn(sortTypeStable),
                                                               ::testing::ValuesIn(netPrecisions),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::ValuesIn(inputShapesDynamic)),
                                            ::testing::ValuesIn(filterCPUSpecificParams(cpuParams)),
                                            ::testing::ValuesIn(additionalConfig)),
                         TopKLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TopK_f16,
                         TopKLayerCPUTest,
                         ::testing::Combine(::testing::Combine(::testing::ValuesIn(k),
                                                               ::testing::ValuesIn(axes),
                                                               ::testing::ValuesIn(modes),
                                                               ::testing::ValuesIn(sortTypeStable),
                                                               ::testing::ValuesIn(netPrecisions_f16),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::ValuesIn(inputShapes)),
                                            ::testing::ValuesIn(filterCPUSpecificParams(cpuParams)),
                                            ::testing::ValuesIn(additionalConfig_f16)),
                         TopKLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TopK_f16_dynamic,
                         TopKLayerCPUTest,
                         ::testing::Combine(::testing::Combine(::testing::ValuesIn(k_dynamic),
                                                               ::testing::ValuesIn(axes),
                                                               ::testing::ValuesIn(modes),
                                                               ::testing::ValuesIn(sortTypeStable),
                                                               ::testing::ValuesIn(netPrecisions_f16),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::ValuesIn(inputShapesDynamic)),
                                            ::testing::ValuesIn(filterCPUSpecificParams(cpuParams)),
                                            ::testing::ValuesIn(additionalConfig_f16)),
                         TopKLayerCPUTest::getTestCaseName);

const std::vector<int64_t> k_int32 = {1, 5, 7, 9};
const std::vector<int64_t> k_int32_dynamic = {1, 5};

std::vector<ov::test::InputShape> inputShapes_int32 = {
    {{}, {{9, 9, 9, 9}}},
};

std::vector<ov::test::InputShape> inputShapesDynamic_int32 = {
    {{9, {5, 10}, 9, {5, 10}}, {{9, 9, 9, 9}, {9, 10, 9, 10}}}};

INSTANTIATE_TEST_SUITE_P(smoke_TopK_int32,
                         TopKLayerCPUTest,
                         ::testing::Combine(::testing::Combine(::testing::ValuesIn(k_int32),
                                                               ::testing::ValuesIn(axes),
                                                               ::testing::ValuesIn(modes),
                                                               ::testing::ValuesIn(sortTypeStable),
                                                               ::testing::Values(ElementType::i32),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::ValuesIn(inputShapes_int32)),
                                            ::testing::ValuesIn(filterCPUSpecificParams(cpuParams)),
                                            ::testing::Values(additionalConfig[0])),
                         TopKLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TopK_int32_dynamic,
                         TopKLayerCPUTest,
                         ::testing::Combine(::testing::Combine(::testing::ValuesIn(k_int32_dynamic),
                                                               ::testing::ValuesIn(axes),
                                                               ::testing::ValuesIn(modes),
                                                               ::testing::ValuesIn(sortTypeStable),
                                                               ::testing::Values(ElementType::i32),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::ValuesIn(inputShapesDynamic_int32)),
                                            ::testing::ValuesIn(filterCPUSpecificParams(cpuParams)),
                                            ::testing::Values(additionalConfig[0])),
                         TopKLayerCPUTest::getTestCaseName);

std::vector<ov::test::InputShape> inputShapes_bubble_BLK_on_channel_horiz = {
    {{}, {{2, 2, 2, 2}}},
};

std::vector<ov::test::InputShape> inputShapesDynamic_bubble_BLK_on_channel_horiz = {
    {{2, {2, 3}, 2, 2}, {{2, 2, 2, 2}, {2, 3, 2, 2}}}};

INSTANTIATE_TEST_SUITE_P(
    smoke_TopK_bubble_BLK_on_channel_horiz,
    TopKLayerCPUTest,
    ::testing::Combine(::testing::Combine(::testing::Values(1),
                                          ::testing::Values(1),
                                          ::testing::ValuesIn(modes),
                                          ::testing::ValuesIn(sortTypeStable),
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::ValuesIn(inputShapes_bubble_BLK_on_channel_horiz)),
                       ::testing::Values(CPUSpecificParams({nChw16c, x}, {nChw16c, nChw16c}, {}, {})),
                       ::testing::ValuesIn(additionalConfig)),
    TopKLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_TopK_bubble_BLK_on_channel_horiz_dynamic,
    TopKLayerCPUTest,
    ::testing::Combine(::testing::Combine(::testing::Values(1),
                                          ::testing::Values(1),
                                          ::testing::ValuesIn(modes),
                                          ::testing::ValuesIn(sortTypeStable),
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::ValuesIn(inputShapesDynamic_bubble_BLK_on_channel_horiz)),
                       ::testing::Values(CPUSpecificParams({nChw16c, x}, {nChw16c, nChw16c}, {}, {})),
                       ::testing::ValuesIn(additionalConfig)),
    TopKLayerCPUTest::getTestCaseName);

std::vector<ov::test::InputShape> inputShapes_top1 = {
    {{}, {{1, 1, 2, 1}}},
};

std::vector<ov::test::InputShape> inputShapesDynamic_top1 = {{{1, 1, 2, {1, 2}}, {{1, 1, 2, 1}, {1, 1, 2, 2}}}};

INSTANTIATE_TEST_SUITE_P(
    smoke_Top1,
    TopKLayerCPUTest,
    ::testing::Combine(::testing::Combine(::testing::Values(1),
                                          ::testing::Values(3),
                                          ::testing::Values(SortMode::MAX),
                                          ::testing::Values(std::tuple<SortType, bool>(SortType::SORT_INDICES, false)),
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::ValuesIn(inputShapes_top1)),
                       ::testing::Values(CPUSpecificParams({nchw, x}, {nchw, nchw}, {}, {})),
                       ::testing::ValuesIn(additionalConfig)),
    TopKLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Top1_dynamic,
    TopKLayerCPUTest,
    ::testing::Combine(::testing::Combine(::testing::Values(1),
                                          ::testing::Values(3),
                                          ::testing::Values(SortMode::MAX),
                                          ::testing::Values(std::tuple<SortType, bool>(SortType::SORT_INDICES, false)),
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::ValuesIn(inputShapesDynamic_top1)),
                       ::testing::Values(CPUSpecificParams({nchw, x}, {nchw, nchw}, {}, {})),
    ::testing::ValuesIn(additionalConfig)),
    TopKLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_TopK_vl_tail,
    TopKLayerCPUTest,
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(k_vl_tail),
                                          ::testing::ValuesIn(axes_vl_tail),
                                          ::testing::ValuesIn(modes),
                                          ::testing::ValuesIn(sortTypeStable_vl_tail),
                                          ::testing::ValuesIn(netPrecisions),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::ValuesIn(inputShapes_vl_tail)),
                       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_vl_tail)),
                       ::testing::ValuesIn(additionalConfig)),
    TopKLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_TopK_f16_vl_tail,
    TopKLayerCPUTest,
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(k_vl_tail),
                                          ::testing::ValuesIn(axes_vl_tail),
                                          ::testing::ValuesIn(modes),
                                          ::testing::ValuesIn(sortTypeStable_vl_tail),
                                          ::testing::ValuesIn(netPrecisions_f16),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::ValuesIn(inputShapes_vl_tail)),
                       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_vl_tail)),
                       ::testing::ValuesIn(additionalConfig_f16)),
    TopKLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_TopK_int32_vl_tail,
    TopKLayerCPUTest,
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(k_vl_tail),
                                          ::testing::ValuesIn(axes_vl_tail),
                                          ::testing::ValuesIn(modes),
                                          ::testing::ValuesIn(sortTypeStable_vl_tail),
                                          ::testing::Values(ElementType::i32),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::Values(ElementType::dynamic),
                                          ::testing::ValuesIn(inputShapes_vl_tail)),
                       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_vl_tail)),
                       ::testing::Values(additionalConfig[0])),
    TopKLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace test
}  // namespace ov
