// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scatter_elements_update.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
using Reduction = ov::op::v12::ScatterElementsUpdate::Reduction;

struct ScatterElementsUpdateParams {
    ScatterElementsUpdateParams(reference_tests::Tensor paramData,
                                reference_tests::Tensor paramIndices,
                                reference_tests::Tensor paramUpdates,
                                reference_tests::Tensor paramAxis,
                                reference_tests::Tensor paramExpected,
                                const Reduction paramReduction = Reduction::NONE,
                                const bool paramUseInitValue = true)
        : input{std::move(paramData)},
          indices{std::move(paramIndices)},
          updates{std::move(paramUpdates)},
          axis{std::move(paramAxis)},
          expected{std::move(paramExpected)},
          reduction{paramReduction},
          use_init_value{paramUseInitValue} {}

    const reference_tests::Tensor input;
    const reference_tests::Tensor indices;
    const reference_tests::Tensor updates;
    const reference_tests::Tensor axis;
    const reference_tests::Tensor expected;
    const Reduction reduction;
    const bool use_init_value;
};

class ReferenceScatterElementsUpdateV3LayerTest : public testing::TestWithParam<ScatterElementsUpdateParams>,
                                                  public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.input.data, params.indices.data, params.updates.data, params.axis.data};
        refOutData = {params.expected.data};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ScatterElementsUpdateParams>& obj) {
        const auto& param = obj.param;
        std::ostringstream result;
        result << "data_sh=" << param.input.shape;
        result << "_data_pr=" << param.input.type;
        result << "_indx_sh=" << param.indices.shape;
        result << "_indx_pr=" << param.indices.type;
        result << "_updt_sh=" << param.updates.shape;
        result << "_updt_pr=" << param.updates.type;
        result << "_axis_sh=" << param.axis.shape;
        result << "_axis_pr=" << param.axis.type;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ScatterElementsUpdateParams& params) {
        const auto A = std::make_shared<op::v0::Parameter>(params.input.type, params.input.shape);
        const auto B = std::make_shared<op::v0::Parameter>(params.indices.type, params.indices.shape);
        const auto C = std::make_shared<op::v0::Parameter>(params.updates.type, params.updates.shape);
        const auto D = std::make_shared<op::v0::Parameter>(params.axis.type, params.axis.shape);
        auto scatterElts = std::make_shared<op::v3::ScatterElementsUpdate>(A, B, C, D);
        return std::make_shared<ov::Model>(NodeVector{scatterElts}, ParameterVector{A, B, C, D});
    }
};

class ReferenceScatterElementsUpdateV12LayerTest : public testing::TestWithParam<ScatterElementsUpdateParams>,
                                                   public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.input.data, params.indices.data, params.updates.data, params.axis.data};
        refOutData = {params.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<ScatterElementsUpdateParams>& obj) {
        static std::map<Reduction, std::string> reduction_as_string = {
            {Reduction::NONE, "none"},
            {Reduction::SUM, "sum"},
            {Reduction::PROD, "prod"},
            {Reduction::MIN, "min"},
            {Reduction::MAX, "max"},
            {Reduction::MEAN, "mean"},
        };
        const auto& param = obj.param;
        std::ostringstream result;
        result << ReferenceScatterElementsUpdateV3LayerTest::getTestCaseName(obj);
        result << "_reduction=" << reduction_as_string[param.reduction];
        result << "_use_init_value=" << std::boolalpha << param.use_init_value;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ScatterElementsUpdateParams& params) {
        const auto data = std::make_shared<op::v0::Parameter>(params.input.type, params.input.shape);
        const auto indices = std::make_shared<op::v0::Parameter>(params.indices.type, params.indices.shape);
        const auto updates = std::make_shared<op::v0::Parameter>(params.updates.type, params.updates.shape);
        const auto axis = std::make_shared<op::v0::Parameter>(params.axis.type, params.axis.shape);
        auto scatter_eu = std::make_shared<op::v12::ScatterElementsUpdate>(data,
                                                                           indices,
                                                                           updates,
                                                                           axis,
                                                                           params.reduction,
                                                                           params.use_init_value);
        return std::make_shared<ov::Model>(NodeVector{scatter_eu}, ParameterVector{data, indices, updates, axis});
    }
};

TEST_P(ReferenceScatterElementsUpdateV3LayerTest, CompareWithHardcodedRefs) {
    Exec();
}

TEST_P(ReferenceScatterElementsUpdateV12LayerTest, CompareWithHardcodedRefs) {
    Exec();
}

template <element::Type_t ET, element::Type_t ET_IND>
std::vector<ScatterElementsUpdateParams> generateScatterParams() {
    using T = typename element_type_traits<ET>::value_type;
    using T_INT = typename element_type_traits<ET_IND>::value_type;
    std::vector<ScatterElementsUpdateParams> scatterParams{
        // axis = 0
        ScatterElementsUpdateParams(
            reference_tests::Tensor({2, 2}, element::Type(ET), std::vector<T>{1, 2, 3, 4}),          // input
            reference_tests::Tensor({2, 2}, element::Type(ET_IND), std::vector<T_INT>{1, 1, 0, 0}),  // indices
            reference_tests::Tensor({2, 2}, element::Type(ET), std::vector<T>{10, 20, 30, 40}),      // updates
            reference_tests::Tensor({1}, element::Type(ET_IND), std::vector<T_INT>{0}),              // axis
            reference_tests::Tensor({2, 2}, element::Type(ET), std::vector<T>{30, 40, 10, 20})),     // expected
        // axis = 1
        ScatterElementsUpdateParams(
            reference_tests::Tensor({2, 1}, element::Type(ET), std::vector<T>{1, 2}),          // input
            reference_tests::Tensor({2, 1}, element::Type(ET_IND), std::vector<T_INT>{0, 0}),  // indices
            reference_tests::Tensor({2, 1}, element::Type(ET), std::vector<T>{10, 20}),        // updates
            reference_tests::Tensor({1}, element::Type(ET_IND), std::vector<T_INT>{1}),        // axis
            reference_tests::Tensor({2, 1}, element::Type(ET), std::vector<T>{10, 20})),       // expected
    };
    return scatterParams;
}

std::vector<ScatterElementsUpdateParams> generateScatterCombinedParams() {
    const std::vector<std::vector<ScatterElementsUpdateParams>> scatterTypeParams{
        // i16
        generateScatterParams<element::Type_t::i16, element::Type_t::i8>(),
        generateScatterParams<element::Type_t::i16, element::Type_t::u8>(),
        generateScatterParams<element::Type_t::i16, element::Type_t::i16>(),
        generateScatterParams<element::Type_t::i16, element::Type_t::u16>(),
        generateScatterParams<element::Type_t::i16, element::Type_t::i32>(),
        generateScatterParams<element::Type_t::i16, element::Type_t::u32>(),
        generateScatterParams<element::Type_t::i16, element::Type_t::i64>(),
        generateScatterParams<element::Type_t::i16, element::Type_t::u64>(),
        // i32
        generateScatterParams<element::Type_t::i32, element::Type_t::i8>(),
        generateScatterParams<element::Type_t::i32, element::Type_t::u8>(),
        generateScatterParams<element::Type_t::i32, element::Type_t::i16>(),
        generateScatterParams<element::Type_t::i32, element::Type_t::u16>(),
        generateScatterParams<element::Type_t::i32, element::Type_t::i32>(),
        generateScatterParams<element::Type_t::i32, element::Type_t::u32>(),
        generateScatterParams<element::Type_t::i32, element::Type_t::i64>(),
        generateScatterParams<element::Type_t::i32, element::Type_t::u64>(),
        // i64
        generateScatterParams<element::Type_t::i64, element::Type_t::i8>(),
        generateScatterParams<element::Type_t::i64, element::Type_t::u8>(),
        generateScatterParams<element::Type_t::i64, element::Type_t::i16>(),
        generateScatterParams<element::Type_t::i64, element::Type_t::u16>(),
        generateScatterParams<element::Type_t::i64, element::Type_t::i32>(),
        generateScatterParams<element::Type_t::i64, element::Type_t::u32>(),
        generateScatterParams<element::Type_t::i64, element::Type_t::i64>(),
        generateScatterParams<element::Type_t::i64, element::Type_t::u64>(),
        // u32
        generateScatterParams<element::Type_t::u32, element::Type_t::i8>(),
        generateScatterParams<element::Type_t::u32, element::Type_t::u8>(),
        generateScatterParams<element::Type_t::u32, element::Type_t::i16>(),
        generateScatterParams<element::Type_t::u32, element::Type_t::u16>(),
        generateScatterParams<element::Type_t::u32, element::Type_t::i32>(),
        generateScatterParams<element::Type_t::u32, element::Type_t::u32>(),
        generateScatterParams<element::Type_t::u32, element::Type_t::i64>(),
        generateScatterParams<element::Type_t::u32, element::Type_t::u64>(),
        // u64
        generateScatterParams<element::Type_t::u64, element::Type_t::i8>(),
        generateScatterParams<element::Type_t::u64, element::Type_t::u8>(),
        generateScatterParams<element::Type_t::u64, element::Type_t::i16>(),
        generateScatterParams<element::Type_t::u64, element::Type_t::u16>(),
        generateScatterParams<element::Type_t::u64, element::Type_t::i32>(),
        generateScatterParams<element::Type_t::u64, element::Type_t::u32>(),
        generateScatterParams<element::Type_t::u64, element::Type_t::i64>(),
        generateScatterParams<element::Type_t::u64, element::Type_t::u64>(),
        // f16
        generateScatterParams<element::Type_t::f16, element::Type_t::i8>(),
        generateScatterParams<element::Type_t::f16, element::Type_t::u8>(),
        generateScatterParams<element::Type_t::f16, element::Type_t::i16>(),
        generateScatterParams<element::Type_t::f16, element::Type_t::u16>(),
        generateScatterParams<element::Type_t::f16, element::Type_t::i32>(),
        generateScatterParams<element::Type_t::f16, element::Type_t::u32>(),
        generateScatterParams<element::Type_t::f16, element::Type_t::i64>(),
        generateScatterParams<element::Type_t::f16, element::Type_t::u64>(),
        // f32
        generateScatterParams<element::Type_t::f32, element::Type_t::i8>(),
        generateScatterParams<element::Type_t::f32, element::Type_t::u8>(),
        generateScatterParams<element::Type_t::f32, element::Type_t::i16>(),
        generateScatterParams<element::Type_t::f32, element::Type_t::u16>(),
        generateScatterParams<element::Type_t::f32, element::Type_t::i32>(),
        generateScatterParams<element::Type_t::f32, element::Type_t::u32>(),
        generateScatterParams<element::Type_t::f32, element::Type_t::i64>(),
        generateScatterParams<element::Type_t::f32, element::Type_t::u64>(),
    };
    std::vector<ScatterElementsUpdateParams> combinedParams;
    for (const auto& param : scatterTypeParams) {
        std::move(param.begin(), param.end(), std::back_inserter(combinedParams));
    }
    return combinedParams;
}

template <typename Indices_t, typename std::enable_if<std::is_signed<Indices_t>::value>::type* = nullptr>
Indices_t norm(int i, int d) {
    return static_cast<Indices_t>(i);
}
template <typename Indices_t, typename std::enable_if<std::is_unsigned<Indices_t>::value>::type* = nullptr>
Indices_t norm(int i, int d) {
    return static_cast<Indices_t>(i < 0 ? i + d : i);
}

template <element::Type_t DATA_ET, element::Type_t INDICES_ET>
std::vector<ScatterElementsUpdateParams> generate_scatter_eu_v12_params() {
    using Data_t = typename element_type_traits<DATA_ET>::value_type;
    using Indices_t = typename element_type_traits<INDICES_ET>::value_type;
    return {
        {{Shape{3, 2}, element::Type(DATA_ET), std::vector<Data_t>{11, 12, 13, 14, 15, 16}},           // data
         {Shape{1, 2}, element::Type(INDICES_ET), std::vector<Indices_t>{norm<Indices_t>(-1, 3), 1}},  // indices
         {Shape{1, 2}, element::Type(DATA_ET), std::vector<Data_t>{5, 24}},                            // updates
         {Shape{1}, element::Type(INDICES_ET), std::vector<Indices_t>{0}},                             // axis
         {Shape{3, 2}, element::Type(DATA_ET), std::vector<Data_t>{11, 12, 13, 24, 15, 16}},           // expected
         Reduction::MAX,
         true},
        {{Shape{2, 3}, element::Type(DATA_ET), std::vector<Data_t>{11, 12, 13, 14, 15, 16}},
         {Shape{2, 2}, element::Type(INDICES_ET), std::vector<Indices_t>{norm<Indices_t>(-3, 3), 1, 0, 2}},
         {Shape{2, 2}, element::Type(DATA_ET), std::vector<Data_t>{1, 22, 24, 6}},
         {Shape{1}, element::Type(INDICES_ET), std::vector<Indices_t>{1}},
         {Shape{2, 3}, element::Type(DATA_ET), std::vector<Data_t>{1, 22, 13, 24, 15, 6}},
         Reduction::MIN,
         false},
        {{Shape{1, 2, 3}, element::Type(DATA_ET), std::vector<Data_t>{11, 12, 13, 14, 15, 16}},
         {Shape{1, 1, 4}, element::Type(INDICES_ET), std::vector<Indices_t>{0, 1, 0, 2}},
         {Shape{1, 1, 4}, element::Type(DATA_ET), std::vector<Data_t>{23, 38, 32, 7}},
         {Shape{1}, element::Type(INDICES_ET), std::vector<Indices_t>{2}},
         {Shape{1, 2, 3}, element::Type(DATA_ET), std::vector<Data_t>{22, 25, 10, 14, 15, 16}},
         Reduction::MEAN,
         true},
        {{Shape{1, 2, 3}, element::Type(DATA_ET), std::vector<Data_t>{11, 12, 13, 14, 15, 16}},
         {Shape{1, 1, 4}, element::Type(INDICES_ET), std::vector<Indices_t>{0, 1, 0, 0}},
         {Shape{1, 1, 4}, element::Type(DATA_ET), std::vector<Data_t>{20, 33, 26, 29}},
         {Shape{1}, element::Type(INDICES_ET), std::vector<Indices_t>{2}},
         {Shape{1, 2, 3}, element::Type(DATA_ET), std::vector<Data_t>{25, 33, 13, 14, 15, 16}},
         Reduction::MEAN,
         false},
        {{Shape{2, 2, 1}, element::Type(DATA_ET), std::vector<Data_t>{1, 2, 3, 4}},
         {Shape{1, 5, 1}, element::Type(INDICES_ET), std::vector<Indices_t>{0, 0, 1, 1, 1}},
         {Shape{1, 5, 1}, element::Type(DATA_ET), std::vector<Data_t>{50, 51, 10, 20, 30}},
         {Shape{1}, element::Type(INDICES_ET), std::vector<Indices_t>{1}},
         {Shape{2, 2, 1}, element::Type(DATA_ET), std::vector<Data_t>{101, 60, 3, 4}},
         Reduction::SUM,
         false},
        {{Shape{3, 2}, element::Type(DATA_ET), std::vector<Data_t>{1, 2, 3, 4, 5, 6}},
         {Shape{4, 1}, element::Type(INDICES_ET), std::vector<Indices_t>{0, 0, 1, 2}},
         {Shape{4, 1}, element::Type(DATA_ET), std::vector<Data_t>{7, 7, 10, 5}},
         {Shape{1}, element::Type(INDICES_ET), std::vector<Indices_t>{0}},
         {Shape{3, 2}, element::Type(DATA_ET), std::vector<Data_t>{49, 2, 30, 4, 25, 6}},
         Reduction::PROD,
         true},
    };
}

std::vector<ScatterElementsUpdateParams> collect_scatter_eu_v12_params() {
    const std::vector<std::vector<ScatterElementsUpdateParams>> params{
        // i16
        generate_scatter_eu_v12_params<element::Type_t::i16, element::Type_t::i8>(),
        generate_scatter_eu_v12_params<element::Type_t::i16, element::Type_t::u8>(),
        generate_scatter_eu_v12_params<element::Type_t::i16, element::Type_t::i16>(),
        generate_scatter_eu_v12_params<element::Type_t::i16, element::Type_t::u16>(),
        generate_scatter_eu_v12_params<element::Type_t::i16, element::Type_t::i32>(),
        generate_scatter_eu_v12_params<element::Type_t::i16, element::Type_t::u32>(),
        generate_scatter_eu_v12_params<element::Type_t::i16, element::Type_t::i64>(),
        generate_scatter_eu_v12_params<element::Type_t::i16, element::Type_t::u64>(),
        // i32
        generate_scatter_eu_v12_params<element::Type_t::i32, element::Type_t::i8>(),
        generate_scatter_eu_v12_params<element::Type_t::i32, element::Type_t::u8>(),
        generate_scatter_eu_v12_params<element::Type_t::i32, element::Type_t::i16>(),
        generate_scatter_eu_v12_params<element::Type_t::i32, element::Type_t::u16>(),
        generate_scatter_eu_v12_params<element::Type_t::i32, element::Type_t::i32>(),
        generate_scatter_eu_v12_params<element::Type_t::i32, element::Type_t::u32>(),
        generate_scatter_eu_v12_params<element::Type_t::i32, element::Type_t::i64>(),
        generate_scatter_eu_v12_params<element::Type_t::i32, element::Type_t::u64>(),
        // i64
        generate_scatter_eu_v12_params<element::Type_t::i64, element::Type_t::i8>(),
        generate_scatter_eu_v12_params<element::Type_t::i64, element::Type_t::u8>(),
        generate_scatter_eu_v12_params<element::Type_t::i64, element::Type_t::i16>(),
        generate_scatter_eu_v12_params<element::Type_t::i64, element::Type_t::u16>(),
        generate_scatter_eu_v12_params<element::Type_t::i64, element::Type_t::i32>(),
        generate_scatter_eu_v12_params<element::Type_t::i64, element::Type_t::u32>(),
        generate_scatter_eu_v12_params<element::Type_t::i64, element::Type_t::i64>(),
        generate_scatter_eu_v12_params<element::Type_t::i64, element::Type_t::u64>(),
        // u32
        generate_scatter_eu_v12_params<element::Type_t::u32, element::Type_t::i8>(),
        generate_scatter_eu_v12_params<element::Type_t::u32, element::Type_t::u8>(),
        generate_scatter_eu_v12_params<element::Type_t::u32, element::Type_t::i16>(),
        generate_scatter_eu_v12_params<element::Type_t::u32, element::Type_t::u16>(),
        generate_scatter_eu_v12_params<element::Type_t::u32, element::Type_t::i32>(),
        generate_scatter_eu_v12_params<element::Type_t::u32, element::Type_t::u32>(),
        generate_scatter_eu_v12_params<element::Type_t::u32, element::Type_t::i64>(),
        generate_scatter_eu_v12_params<element::Type_t::u32, element::Type_t::u64>(),
        // u64
        generate_scatter_eu_v12_params<element::Type_t::u64, element::Type_t::i8>(),
        generate_scatter_eu_v12_params<element::Type_t::u64, element::Type_t::u8>(),
        generate_scatter_eu_v12_params<element::Type_t::u64, element::Type_t::i16>(),
        generate_scatter_eu_v12_params<element::Type_t::u64, element::Type_t::u16>(),
        generate_scatter_eu_v12_params<element::Type_t::u64, element::Type_t::i32>(),
        generate_scatter_eu_v12_params<element::Type_t::u64, element::Type_t::u32>(),
        generate_scatter_eu_v12_params<element::Type_t::u64, element::Type_t::i64>(),
        generate_scatter_eu_v12_params<element::Type_t::u64, element::Type_t::u64>(),
        // f16
        generate_scatter_eu_v12_params<element::Type_t::f16, element::Type_t::i8>(),
        generate_scatter_eu_v12_params<element::Type_t::f16, element::Type_t::u8>(),
        generate_scatter_eu_v12_params<element::Type_t::f16, element::Type_t::i16>(),
        generate_scatter_eu_v12_params<element::Type_t::f16, element::Type_t::u16>(),
        generate_scatter_eu_v12_params<element::Type_t::f16, element::Type_t::i32>(),
        generate_scatter_eu_v12_params<element::Type_t::f16, element::Type_t::u32>(),
        generate_scatter_eu_v12_params<element::Type_t::f16, element::Type_t::i64>(),
        generate_scatter_eu_v12_params<element::Type_t::f16, element::Type_t::u64>(),
        // f32
        generate_scatter_eu_v12_params<element::Type_t::f32, element::Type_t::i8>(),
        generate_scatter_eu_v12_params<element::Type_t::f32, element::Type_t::u8>(),
        generate_scatter_eu_v12_params<element::Type_t::f32, element::Type_t::i16>(),
        generate_scatter_eu_v12_params<element::Type_t::f32, element::Type_t::u16>(),
        generate_scatter_eu_v12_params<element::Type_t::f32, element::Type_t::i32>(),
        generate_scatter_eu_v12_params<element::Type_t::f32, element::Type_t::u32>(),
        generate_scatter_eu_v12_params<element::Type_t::f32, element::Type_t::i64>(),
        generate_scatter_eu_v12_params<element::Type_t::f32, element::Type_t::u64>(),
    };

    auto combined_params = generateScatterCombinedParams();
    for (const auto& param : params) {
        std::move(param.begin(), param.end(), std::back_inserter(combined_params));
    }
    return combined_params;
}

INSTANTIATE_TEST_SUITE_P(smoke_ScatterElementsUpdate,
                         ReferenceScatterElementsUpdateV3LayerTest,
                         ::testing::ValuesIn(generateScatterCombinedParams()),
                         ReferenceScatterElementsUpdateV3LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ScatterElementsUpdate,
                         ReferenceScatterElementsUpdateV12LayerTest,
                         ::testing::ValuesIn(collect_scatter_eu_v12_params()),
                         ReferenceScatterElementsUpdateV12LayerTest::getTestCaseName);
}  // namespace
