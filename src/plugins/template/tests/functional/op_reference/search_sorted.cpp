// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/search_sorted.hpp"

#include <gtest/gtest.h>

#include "base_reference_test.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"

using namespace reference_tests;
using namespace ov;

namespace {

struct SearchSortedParams {
    PartialShape sortedShape;
    PartialShape valuesShape;
    bool rightMode;
    std::string testcaseName;
    reference_tests::Tensor sorted;
    reference_tests::Tensor values;
    reference_tests::Tensor expectedOutput;
};

template <typename T>
SearchSortedParams PrepareTestCaseParams(const PartialShape& sortedShape,
                                         const PartialShape& valuesShape,
                                         bool rightMode,
                                         const std::vector<T>& sortedData,
                                         const std::vector<T>& valuesData,
                                         const std::vector<int64_t>& expectedData,
                                         const std::string& testcaseName) {
    SearchSortedParams ret;
    const auto elementType = element::from<T>();

    ret.sortedShape = sortedShape;
    ret.valuesShape = valuesShape;
    ret.rightMode = rightMode;
    ret.testcaseName = testcaseName;
    ret.sorted = reference_tests::Tensor(elementType, sortedShape.to_shape(), sortedData);
    ret.values = reference_tests::Tensor(elementType, valuesShape.to_shape(), valuesData);
    ret.expectedOutput = reference_tests::Tensor(element::Type_t::i64, valuesShape.to_shape(), expectedData);

    return ret;
}

class ReferenceSearchSortedTest : public testing::TestWithParam<SearchSortedParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        const auto& params = GetParam();
        function = CreateFunction(params);
        inputData = {params.sorted.data, params.values.data};
        refOutData = {params.expectedOutput.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<SearchSortedParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "type=" << param.sorted.data.get_element_type();
        result << "_sortedShape=" << param.sortedShape;
        result << "_valuesShape=" << param.valuesShape;
        result << "_rightMode=" << param.rightMode;
        result << "_=" << param.testcaseName;

        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const SearchSortedParams& params) {
        const auto sorted =
            std::make_shared<op::v0::Parameter>(params.sorted.data.get_element_type(), params.sortedShape);
        const auto values =
            std::make_shared<op::v0::Parameter>(params.values.data.get_element_type(), params.valuesShape);

        const auto op = std::make_shared<op::v15::SearchSorted>(sorted, values, params.rightMode);

        return std::make_shared<Model>(NodeVector{op}, ParameterVector{sorted, values});
    }
};

TEST_P(ReferenceSearchSortedTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t ET>
std::vector<SearchSortedParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    std::vector<SearchSortedParams> params;

#define TEST_DATA(sorted_shape, values_shape, right_mode, sorted_data, values_data, expected_output_data, description) \
    params.push_back(PrepareTestCaseParams<T>(sorted_shape,                                                            \
                                              values_shape,                                                            \
                                              right_mode,                                                              \
                                              sorted_data,                                                             \
                                              values_data,                                                             \
                                              expected_output_data,                                                    \
                                              description));

#include "unit_test_utils/tests_data/search_sorted_data.h"
#undef TEST_DATA

    return params;
}

std::vector<SearchSortedParams> generateCombinedParams() {
    const std::vector<std::vector<SearchSortedParams>> generatedParams{generateParams<element::Type_t::i32>(),
                                                                       generateParams<element::Type_t::f32>()};
    std::vector<SearchSortedParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_SearchSorted_With_Hardcoded_Refs,
                         ReferenceSearchSortedTest,
                         testing::ValuesIn(generateCombinedParams()),
                         ReferenceSearchSortedTest::getTestCaseName);
}  // namespace
