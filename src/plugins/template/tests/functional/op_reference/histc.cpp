// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/histc.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

#include "base_reference_test.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;

namespace {
struct HistcParams {
    Tensor data;
    int64_t bins;
    double min;
    double max;
    Tensor expected;
    std::string name;
};

class ReferenceHistcLayerTest : public testing::TestWithParam<HistcParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& p = GetParam();
        const auto in = std::make_shared<ov::op::v0::Parameter>(p.data.type, p.data.shape);
        const auto histc = std::make_shared<ov::op::v17::Histc>(in, p.bins, p.min, p.max);
        function = std::make_shared<ov::Model>(ov::OutputVector{histc}, ov::ParameterVector{in});
        inputData = {p.data.data};
        refOutData = {p.expected.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<HistcParams>& obj) {
        const auto& p = obj.param;
        return p.name + "_type=" + p.data.type.get_type_name() + "_shape=" + p.data.shape.to_string();
    }
};

TEST_P(ReferenceHistcLayerTest, CompareWithRefs) {
    Exec();
}

// Expected values match torch.histc.
template <ov::element::Type_t ET>
std::vector<HistcParams> generateParams() {
    using T = ov::fundamental_type_for<ET>;
    using ov::Shape;
    return {
        {{Shape{3}, ET, std::vector<T>{1, 2, 1}},
         4,
         0.,
         3.,
         {Shape{4}, ET, std::vector<T>{0, 2, 1, 0}},
         "explicit_range"},
        {{Shape{2, 2}, ET, std::vector<T>{1, 2, 1, 4}},
         4,
         0.,
         0.,
         {Shape{4}, ET, std::vector<T>{2, 1, 0, 1}},
         "data_range"},
        {{Shape{6}, ET, std::vector<T>{0, 1, 2, 3, 4, 5}},
         3,
         0.5,
         4.5,
         {Shape{3}, ET, std::vector<T>{1, 2, 1}},
         "fractional_range_out_of_range_ignored"},
        {{Shape{3}, ET, std::vector<T>{3, 3, 3}}, 4, 0., 0., {Shape{4}, ET, std::vector<T>{0, 0, 3, 0}}, "equal_data"},
        {{Shape{3}, ET, std::vector<T>{7, 1, 9}},
         2,
         4.,
         4.,
         {Shape{2}, ET, std::vector<T>{1, 2}},
         "min_eq_max_nonzero"},
        {{Shape{0}, ET, std::vector<T>{}}, 3, 0., 0., {Shape{3}, ET, std::vector<T>{0, 0, 0}}, "empty"},
        {{Shape{}, ET, std::vector<T>{5}}, 2, 0., 0., {Shape{2}, ET, std::vector<T>{0, 1}}, "scalar"},
        {{Shape{2, 3}, ET, std::vector<T>{0, 1, 2, 3, 4, 5}},
         1,
         0.,
         10.,
         {Shape{1}, ET, std::vector<T>{6}},
         "single_bin"},
        {{Shape{5}, ET, std::vector<T>{0, 1, 2, 3, 4}},
         4,
         0.,
         4.,
         {Shape{4}, ET, std::vector<T>{1, 1, 1, 2}},
         "all_on_bin_edges"},
        {{Shape{2}, ET, std::vector<T>{1, 3}},
         10,
         0.,
         0.,
         {Shape{10}, ET, std::vector<T>{1, 0, 0, 0, 0, 0, 0, 0, 0, 1}},
         "more_bins_than_elements"},
        {{Shape{3}, ET, std::vector<T>{5, 6, 7}}, 2, 0., 1., {Shape{2}, ET, std::vector<T>{0, 0}}, "all_out_of_range"},
        {{Shape{0, 4}, ET, std::vector<T>{}},
         2,
         5.,
         5.,
         {Shape{2}, ET, std::vector<T>{0, 0}},
         "empty_min_eq_max_nonzero"},
    };
}

template <ov::element::Type_t ET>
std::vector<HistcParams> generateFloatParams() {
    using T = ov::fundamental_type_for<ET>;
    using ov::Shape;
    const auto inf = std::numeric_limits<float>::infinity();
    auto params = generateParams<ET>();
    const std::vector<HistcParams> float_params{
        {{Shape{5}, ET, std::vector<T>{T(-1), T(0), T(0.5), T(1), T(2)}},
         2,
         0.,
         1.,
         {Shape{2}, ET, std::vector<T>{1, 2}},
         "max_goes_to_last_bin"},
        {{Shape{3}, ET, std::vector<T>{T(1), T(std::numeric_limits<float>::quiet_NaN()), T(2)}},
         2,
         0.,
         2.,
         {Shape{2}, ET, std::vector<T>{0, 2}},
         "nan_ignored"},
        {{Shape{2},
          ET,
          std::vector<T>{T(std::numeric_limits<float>::quiet_NaN()), T(std::numeric_limits<float>::quiet_NaN())}},
         2,
         0.,
         1.,
         {Shape{2}, ET, std::vector<T>{0, 0}},
         "all_nan_explicit_range"},
        {{Shape{4}, ET, std::vector<T>{T(-0.75), T(-0.25), T(0.25), T(0.75)}},
         4,
         -1.,
         1.,
         {Shape{4}, ET, std::vector<T>{1, 1, 1, 1}},
         "negative_range"},
        {{Shape{3}, ET, std::vector<T>{T(-inf), T(0.5), T(inf)}},
         2,
         0.,
         1.,
         {Shape{2}, ET, std::vector<T>{0, 1}},
         "infinities_ignored"},
        {{Shape{1}, ET, std::vector<T>{T(-0.0f)}}, 2, 0., 1., {Shape{2}, ET, std::vector<T>{1, 0}}, "negative_zero"},
    };
    params.insert(params.end(), float_params.begin(), float_params.end());
    return params;
}

std::vector<HistcParams> generateCombinedParams() {
    using ov::Shape;
    std::vector<std::vector<HistcParams>> all{generateFloatParams<ov::element::f32>(),
                                              generateFloatParams<ov::element::f64>(),
                                              generateFloatParams<ov::element::f16>(),
                                              generateFloatParams<ov::element::bf16>(),
                                              generateParams<ov::element::i8>(),
                                              generateParams<ov::element::i32>(),
                                              generateParams<ov::element::i64>(),
                                              generateParams<ov::element::u8>(),
                                              generateParams<ov::element::u32>()};
    // f32 range ends are rounded to f32, so 0.3f is not above max = 0.3.
    all.push_back({{{Shape{1}, ov::element::f32, std::vector<float>{0.3f}},
                    3,
                    0.,
                    0.3,
                    {Shape{3}, ov::element::f32, std::vector<float>{0, 0, 1}},
                    "max_rounded_to_type"},
                   // Range widening uses nextafter in f32, since 1e30 - 1 == 1e30.
                   {{Shape{2}, ov::element::f32, std::vector<float>{1e30f, 1e30f}},
                    2,
                    0.,
                    0.,
                    {Shape{2}, ov::element::f32, std::vector<float>{0, 2}},
                    "large_equal_data"},
                   {{Shape{3}, ov::element::i32, std::vector<int32_t>{-5, 0, 5}},
                    2,
                    -5.,
                    5.,
                    {Shape{2}, ov::element::i32, std::vector<int32_t>{1, 2}},
                    "signed_int"},
                   {{Shape{1}, ov::element::f32, std::vector<float>{std::nextafter(1.f, 0.f)}},
                    3,
                    0.,
                    1.,
                    {Shape{3}, ov::element::f32, std::vector<float>{0, 0, 1}},
                    "just_below_max"},
                   {{Shape{2}, ov::element::f64, std::vector<double>{1e300, 1e300}},
                    2,
                    0.,
                    0.,
                    {Shape{2}, ov::element::f64, std::vector<double>{0, 2}},
                    "large_equal_data"},
                   // Counts are kept in size_t, so they do not stall at 256 (bf16) or 2048 (f16) increments.
                   {{Shape{1000}, ov::element::bf16, std::vector<ov::bfloat16>(1000, 1)},
                    1,
                    0.,
                    2.,
                    {Shape{1}, ov::element::bf16, std::vector<ov::bfloat16>{1000}},
                    "count_above_bf16_step"},
                   {{Shape{4000}, ov::element::f16, std::vector<ov::float16>(4000, 1)},
                    1,
                    0.,
                    2.,
                    {Shape{1}, ov::element::f16, std::vector<ov::float16>{4000}},
                    "count_above_f16_step"},
                   {{Shape{300}, ov::element::u8, std::vector<uint8_t>(300, 0)},
                    1,
                    0.,
                    1.,
                    {Shape{1}, ov::element::u8, std::vector<uint8_t>{255}},
                    "count_saturates"},
                   {{Shape{200}, ov::element::i8, std::vector<int8_t>(200, 0)},
                    1,
                    0.,
                    1.,
                    {Shape{1}, ov::element::i8, std::vector<int8_t>{127}},
                    "count_saturates"},
                   {{Shape{4}, ov::element::u8, std::vector<uint8_t>{0, 5, 10, 200}},
                    2,
                    -10.,
                    10.,
                    {Shape{2}, ov::element::u8, std::vector<uint8_t>{0, 3}},
                    "unsigned_negative_min"},
                   {{Shape{2}, ov::element::i64, std::vector<int64_t>{-(int64_t{1} << 40), int64_t{1} << 40}},
                    2,
                    0.,
                    0.,
                    {Shape{2}, ov::element::i64, std::vector<int64_t>{1, 1}},
                    "large_int_data_range"}});
    std::vector<HistcParams> combined;
    for (const auto& params : all)
        combined.insert(combined.end(), params.begin(), params.end());
    return combined;
}

INSTANTIATE_TEST_SUITE_P(smoke_Histc_With_Hardcoded_Refs,
                         ReferenceHistcLayerTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceHistcLayerTest::getTestCaseName);
}  // namespace
