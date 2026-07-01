// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/histc.hpp"

#include <gtest/gtest.h>

#include <limits>

#include "base_reference_test.hpp"
#include "openvino/op/parameter.hpp"

namespace {
struct HistcParams {
    template <class IT>
    HistcParams(const ov::PartialShape& shape,
                const ov::element::Type& iType,
                const std::vector<IT>& iValues,
                int64_t bins,
                double min_val,
                double max_val,
                const std::vector<IT>& oValues)
        : pshape(shape),
          inType(iType),
          outType(iType),
          bins(bins),
          min_val(min_val),
          max_val(max_val),
          inputData(reference_tests::CreateTensor(iType, iValues)),
          refData(reference_tests::CreateTensor(ov::Shape{static_cast<size_t>(bins)}, iType, oValues)) {}

    ov::PartialShape pshape;
    ov::element::Type inType;
    ov::element::Type outType;
    int64_t bins;
    double min_val;
    double max_val;
    ov::Tensor inputData;
    ov::Tensor refData;
};

class ReferenceHistcLayerTest : public testing::TestWithParam<HistcParams>,
                                 public reference_tests::CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params.pshape, params.inType, params.bins, params.min_val, params.max_val);
        inputData = {params.inputData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<HistcParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "shape=" << param.pshape << "_";
        result << "iType=" << param.inType << "_";
        result << "bins=" << param.bins << "_";
        result << "min=" << param.min_val << "_";
        result << "max=" << param.max_val;
        return result.str();
    }

private:
    static std::shared_ptr<ov::Model> CreateFunction(const ov::PartialShape& input_shape,
                                                      const ov::element::Type& input_type,
                                                      int64_t bins,
                                                      double min_val,
                                                      double max_val) {
        const auto in = std::make_shared<ov::op::v0::Parameter>(input_type, input_shape);
        const auto histc = std::make_shared<ov::op::v17::Histc>(in, bins, min_val, max_val);
        return std::make_shared<ov::Model>(ov::OutputVector{histc}, ov::ParameterVector{in});
    }
};

TEST_P(ReferenceHistcLayerTest, CompareWithRefs) {
    Exec();
}

template <ov::element::Type_t IN_ET>
std::vector<HistcParams> generateHistcFloatParams() {
    using T = typename ov::element_type_traits<IN_ET>::value_type;

    std::vector<HistcParams> params{
        // Explicit range [0, 5), 5 bins, values evenly land one-per-bin.
        HistcParams(ov::PartialShape{5},
                    IN_ET,
                    std::vector<T>{T(0.0f), T(1.0f), T(2.0f), T(3.0f), T(4.0f)},
                    5,
                    0.0,
                    5.0,
                    std::vector<T>{T(1.0f), T(1.0f), T(1.0f), T(1.0f), T(1.0f)}),
        // Auto-range (min_val == max_val == 0): range is derived from data min/max.
        HistcParams(ov::PartialShape{4},
                    IN_ET,
                    std::vector<T>{T(1.0f), T(2.0f), T(3.0f), T(4.0f)},
                    4,
                    0.0,
                    0.0,
                    std::vector<T>{T(1.0f), T(1.0f), T(1.0f), T(1.0f)}),
        // Values outside [min_val, max_val] are excluded from all bins.
        HistcParams(ov::PartialShape{6},
                    IN_ET,
                    std::vector<T>{T(-10.0f), T(0.5f), T(1.5f), T(2.5f), T(3.5f), T(20.0f)},
                    4,
                    0.0,
                    4.0,
                    std::vector<T>{T(1.0f), T(1.0f), T(1.0f), T(1.0f)}),
        // Degenerate range (min_val == max_val): all in-range values fall in bin 0.
        HistcParams(ov::PartialShape{3},
                    IN_ET,
                    std::vector<T>{T(2.0f), T(2.0f), T(3.0f)},
                    3,
                    2.0,
                    2.0,
                    std::vector<T>{T(2.0f), T(0.0f), T(0.0f)}),
        // Multi-dimensional input is flattened before binning.
        HistcParams(ov::PartialShape{2, 2},
                    IN_ET,
                    std::vector<T>{T(0.0f), T(1.0f), T(2.0f), T(3.0f)},
                    2,
                    0.0,
                    4.0,
                    std::vector<T>{T(2.0f), T(2.0f)}),
    };
    return params;
}

std::vector<HistcParams> generateHistcCombinedParams() {
    const std::vector<std::vector<HistcParams>> typeParams{
        generateHistcFloatParams<ov::element::Type_t::f32>(),
        generateHistcFloatParams<ov::element::Type_t::f16>(),
        generateHistcFloatParams<ov::element::Type_t::bf16>(),
    };
    std::vector<HistcParams> combinedParams;
    for (const auto& p : typeParams) {
        combinedParams.insert(combinedParams.end(), p.begin(), p.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Histc_With_Hardcoded_Refs,
                         ReferenceHistcLayerTest,
                         testing::ValuesIn(generateHistcCombinedParams()),
                         ReferenceHistcLayerTest::getTestCaseName);

// Dedicated edge-case test: empty input tensor must produce an all-zero histogram.
class ReferenceHistcEmptyInputTest : public reference_tests::CommonReferenceTest {
public:
    void SetUp() {
        const auto in = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{0});
        function = std::make_shared<ov::Model>(
            ov::OutputVector{std::make_shared<ov::op::v17::Histc>(in, 4, 0.0, 1.0)},
            ov::ParameterVector{in});
        inputData = {reference_tests::CreateTensor(ov::element::f32, std::vector<float>{})};
        refOutData = {reference_tests::CreateTensor(ov::Shape{4}, ov::element::f32, std::vector<float>{0, 0, 0, 0})};
    }
};

TEST(HistcEdgeCases, EmptyInput) {
    ReferenceHistcEmptyInputTest test;
    test.SetUp();
    test.Exec();
}

}  // namespace
