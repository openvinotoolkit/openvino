// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <random>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;
using namespace ov::test;
using SortMode = ov::op::TopKMode;
using SortType = ov::op::TopKSortType;

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
    static std::string getTestCaseName(testing::TestParamInfo<TopKLayerCPUTestParamsSet> obj) {
        basicTopKParams basicParamsSet;
        CPUSpecificParams cpuParams;
        ov::AnyMap additionalConfig;
        std::tie(basicParamsSet, cpuParams, additionalConfig) = obj.param;

        int64_t keepK, axis;
        SortMode mode;
        std::tuple<SortType, bool> sortTypeStable;
        ElementType netPrecision, inPrc, outPrc;
        InputShape inputShape;
        std::tie(keepK, axis, mode, sortTypeStable, netPrecision, inPrc, outPrc, inputShape) = basicParamsSet;
        SortType sort = std::get<0>(sortTypeStable);
        bool stable = std::get<1>(sortTypeStable);

        std::ostringstream result;
        bool staticShape = inputShape.first.rank() == 0;
        if (staticShape)
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
    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;

        basicTopKParams basicParamsSet;
        CPUSpecificParams cpuParams;
        ov::AnyMap additionalConfig;
        std::tie(basicParamsSet, cpuParams, additionalConfig) = this->GetParam();

        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        int64_t keepK;
        SortMode mode;
        std::tuple<SortType, bool> sortTypeStable;
        ElementType inPrc, outPrc;
        InputShape inputShape;
        std::tie(keepK, axis, mode, sortTypeStable, netPrecision, inPrc, outPrc, inputShape) = basicParamsSet;
        sort = std::get<0>(sortTypeStable);
        stable = std::get<1>(sortTypeStable);

        if (additionalConfig[ov::hint::inference_precision.name()] == ov::element::bf16)
            inPrc = outPrc = netPrecision = ElementType::bf16;
        else
            inPrc = outPrc = netPrecision;
        configuration.insert(additionalConfig.begin(), additionalConfig.end());

        selectedType = getPrimitiveType() + "_" + ov::element::Type(netPrecision).get_type_name();

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
            topk = std::dynamic_pointer_cast<ov::op::v11::TopK>(
                std::make_shared<ov::op::v11::TopK>(params[0], k, axis, mode, sort, ElementType::i32, stable));
        } else {
            auto k = std::make_shared<ov::op::v0::Parameter>(ElementType::i64, inputDynamicShapes[1]);
            params.push_back(k);
            topk = std::dynamic_pointer_cast<ov::op::v11::TopK>(
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
        } else if (netPrecision == ElementType::bf16) {
            size_t O = 1, A = 1, I = 1;
            A = shape[axis];
            for (int64_t i = 0; i < axis; i++)
                O *= shape[i];
            for (size_t i = axis + 1; i < shape.size(); i++)
                I *= shape[i];
            if (O * A * I != size)
                FAIL() << "Incorrect blob shape " << shape;

            auto* rawBlobDataPtr = static_cast<ov::bfloat16*>(tensor.data());
            for (size_t o = 0; o < O; o++) {
                for (size_t i = 0; i < I; i++) {
                    std::vector<int> data(A);
                    int start = -static_cast<int>(A / 2);
                    std::iota(data.begin(), data.end(), start);
                    const size_t seed = (o + 1) * (i + 1);
                    std::mt19937 gen(seed);
                    std::shuffle(data.begin(), data.end(), gen);
                    for (size_t a = 0; a < A; a++) {
                        rawBlobDataPtr[o * A * I + a * I + i] = static_cast<ov::bfloat16>(data[a]);
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
    void generate_dynamic_k(const std::vector<ov::Output<ov::Node>>& funcInputs,
                            const std::vector<ov::Shape>& targetInputStaticShapes) {
        const auto& kPrecision = funcInputs[1].get_element_type();
        const auto& kShape = targetInputStaticShapes[1];

        ov::test::utils::InputGenerateData in_data;
        in_data.start_from = 1;
        in_data.range = targetInputStaticShapes[0][axis];
        in_data.seed = inferRequestNum++;;
        const auto kTensor = ov::test::utils::create_and_fill_tensor(kPrecision, kShape, in_data);

        inputs.insert({funcInputs[1].get_node_shared_ptr(), kTensor});
    }

private:
    int64_t axis;
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

std::vector<ov::AnyMap> additionalConfig = {{{ov::hint::inference_precision(ov::element::f32)}},
                                            {{ov::hint::inference_precision(ov::element::bf16)}}};

const std::vector<int64_t> axes = {0, 1, 2, 3};
const std::vector<int64_t> k = {1, 5, 7, 18, 21};

const std::vector<SortMode> modes = {SortMode::MIN, SortMode::MAX};

const std::vector<std::tuple<SortType, bool>> sortTypeStable = {
    std::tuple<SortType, bool>{SortType::SORT_VALUES, false},
    std::tuple<SortType, bool>{SortType::SORT_VALUES, true},
    std::tuple<SortType, bool>{SortType::SORT_INDICES, false}};

std::vector<ov::test::InputShape> inputShapes = {
    {{}, {{21, 21, 21, 21}}},
};

std::vector<ov::test::InputShape> inputShapesDynamic = {
    {{21, {20, 25}, 21, {20, 25}}, {{21, 21, 21, 21}, {21, 22, 21, 23}}}};

std::vector<CPUSpecificParams> cpuParams = {CPUSpecificParams({nChw16c, x}, {nChw16c, nChw16c}, {}, {}),
                                            CPUSpecificParams({nchw, x}, {nchw, nchw}, {}, {}),
                                            CPUSpecificParams({nhwc, x}, {nhwc, nhwc}, {}, {})};

INSTANTIATE_TEST_SUITE_P(smoke_TopK,
                        TopKLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::ValuesIn(k),
                                                              ::testing::ValuesIn(axes),
                                                              ::testing::ValuesIn(modes),
                                                              ::testing::ValuesIn(sortTypeStable),
                                                              ::testing::ValuesIn(netPrecisions),
                                                              ::testing::Values(ElementType::undefined),
                                                              ::testing::Values(ElementType::undefined),
                                                              ::testing::ValuesIn(inputShapes)),
                                           ::testing::ValuesIn(filterCPUSpecificParams(cpuParams)),
                                           ::testing::ValuesIn(additionalConfig)),
                        TopKLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TopK_dynamic,
                        TopKLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::Values(1),
                                                              ::testing::ValuesIn(axes),
                                                              ::testing::ValuesIn(modes),
                                                              ::testing::ValuesIn(sortTypeStable),
                                                              ::testing::ValuesIn(netPrecisions),
                                                              ::testing::Values(ElementType::undefined),
                                                              ::testing::Values(ElementType::undefined),
                                                              ::testing::ValuesIn(inputShapesDynamic)),
                                           ::testing::ValuesIn(filterCPUSpecificParams(cpuParams)),
                                           ::testing::ValuesIn(additionalConfig)),
                        TopKLayerCPUTest::getTestCaseName);

const std::vector<int64_t> k_int32 = {1, 5, 7, 9};

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
                                                              ::testing::Values(ElementType::undefined),
                                                              ::testing::Values(ElementType::undefined),
                                                              ::testing::ValuesIn(inputShapes_int32)),
                                           ::testing::ValuesIn(filterCPUSpecificParams(cpuParams)),
                                           ::testing::Values(additionalConfig[0])),
                        TopKLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_TopK_int32_dynamic,
                        TopKLayerCPUTest,
                        ::testing::Combine(::testing::Combine(::testing::Values(1),
                                                              ::testing::ValuesIn(axes),
                                                              ::testing::ValuesIn(modes),
                                                              ::testing::ValuesIn(sortTypeStable),
                                                              ::testing::Values(ElementType::i32),
                                                              ::testing::Values(ElementType::undefined),
                                                              ::testing::Values(ElementType::undefined),
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
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::Values(ElementType::undefined),
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
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::Values(ElementType::undefined),
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
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::Values(ElementType::undefined),
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
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::Values(ElementType::undefined),
                                          ::testing::ValuesIn(inputShapesDynamic_top1)),
                       ::testing::Values(CPUSpecificParams({nchw, x}, {nchw, nchw}, {}, {})),
                       ::testing::ValuesIn(additionalConfig)),
    TopKLayerCPUTest::getTestCaseName);

}  // namespace