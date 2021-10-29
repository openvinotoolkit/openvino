// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <gtest/gtest.h>

#include <transformations/init_node_info.hpp>
#include "lpt_ngraph_functions/get_dequantization_function.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "low_precision/network_helper.hpp"


namespace {
using namespace testing;
using namespace ngraph;
using namespace ngraph::pass;

typedef std::tuple<
    bool, // isConvert
    bool, // isSubtract
    size_t, // subDataInput
    // mulDataInput
    size_t> GetDequantizationTestValues;

class GetDequantizationTestTransformation : public LayerTransformation, public testing::WithParamInterface<GetDequantizationTestValues> {
public:
    void SetUp() override {
        bool isConvert;
        bool isSubtract;
        size_t subDataInput;
        size_t mulDataInput;
        std::tie(isConvert, isSubtract, subDataInput, mulDataInput) = this->GetParam();

        actualFunction = ngraph::builder::subgraph::GetDequantizationFunction::getOriginal(
            isConvert, isSubtract, subDataInput, mulDataInput);
        auto dequantization = ngraph::pass::low_precision::NetworkHelper::getDequantization(actualFunction->get_result());
        referenceFunction = ngraph::builder::subgraph::GetDequantizationFunction::getReference(dequantization);
    }

    static std::string getTestCaseName(testing::TestParamInfo<GetDequantizationTestValues> obj) {
        bool isConvert;
        bool isSubtract;
        size_t subDataInput;
        size_t mulDataInput;
        std::tie(isConvert, isSubtract, subDataInput, mulDataInput) = obj.param;

        std::ostringstream result;
        result <<
            (isConvert ? "convert_" : "without_convert_") <<
            (isSubtract ? "_subtract_with_data_input=" : "") <<
            (isSubtract ? std::to_string(subDataInput) : "without_subtract") <<
            (subDataInput == 0 ? "" : "_") <<
            "_multiply_with_data_input=" << mulDataInput;
        return result.str();
    }
};

std::vector<bool> isConvert = { true, false };

std::vector<bool> isSubtract = { true, false };

std::vector<size_t> subDataInput = { 0ul, 1ul };

std::vector<size_t> mulDataInput = { 0ul, 1ul };


TEST_P(GetDequantizationTestTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true);
    ASSERT_TRUE(res.first) << res.second;
}

INSTANTIATE_TEST_SUITE_P(smoke_LPT, GetDequantizationTestTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(isConvert),
        ::testing::ValuesIn(isSubtract),
        ::testing::ValuesIn(subDataInput),
        ::testing::ValuesIn(mulDataInput)),
    GetDequantizationTestTransformation::getTestCaseName);
} // namespace
