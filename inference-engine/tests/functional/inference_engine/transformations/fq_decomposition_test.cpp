// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/fq_decomposition.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"

using FakeQuantizeDecompositionBasicParams = std::tuple<ngraph::element::Type_t, // 'data' input precision
                                                        ngraph::Shape,           // data shape
                                                        ngraph::element::Type_t, // 'range' inputs precision
                                                        ngraph::Shape,           // il shape
                                                        ngraph::Shape,           // ih shape
                                                        ngraph::Shape,           // ol shape
                                                        ngraph::Shape,           // oh shape
                                                        size_t                   // levels
>;

using FakeQuantizeDecompositionParamsSet = std::tuple<FakeQuantizeDecompositionBasicParams,
                                                      std::pair<float, float>, // il and ih values
                                                      bool                     // should be decompos
>;

class FakeQuantizeDecompositionTest : public CommonTestUtils::TestsCommon, public ::testing::WithParamInterface<FakeQuantizeDecompositionParamsSet> {
public:
    static std::string getTestCaseName(::testing::TestParamInfo<FakeQuantizeDecompositionParamsSet> obj) {
        FakeQuantizeDecompositionBasicParams basic_params;
        std::pair<float, float> input_ranges_values;
        bool should_be_decompos;
        std::tie(basic_params, input_ranges_values, should_be_decompos) = obj.param;

        ngraph::Shape data_shape, il_shape, ih_shape, ol_shape, oh_shape;
        ngraph::element::Type_t data_prec, ranges_prec;
        size_t levels;
        std::tie(data_prec, data_shape, ranges_prec, il_shape, ih_shape, ol_shape, oh_shape, levels) = basic_params;

        std::ostringstream result;
        result << "DATA=" << CommonTestUtils::vec2str(data_shape) << "_";
        result << "DATA_PRC=" << ngraph::element::Type(data_prec) << "_";
        result << "IL=" << CommonTestUtils::vec2str(il_shape) << "_" << input_ranges_values.first << "_";
        result << "IH=" << CommonTestUtils::vec2str(ih_shape) << "_" << input_ranges_values.second << "_";
        result << "OL=" << CommonTestUtils::vec2str(ol_shape) << "_";
        result << "OH=" << CommonTestUtils::vec2str(oh_shape) << "_";
        result << "RANGES_PRC=" << ngraph::element::Type(ranges_prec) << "_";
        result << "LEVELS=" << levels;
        return result.str();
    }

protected:
    void SetUp() override {
        FakeQuantizeDecompositionBasicParams basic_params;
        std::pair<float, float> input_ranges_values;
        bool should_be_decompos;
        std::tie(basic_params, input_ranges_values, should_be_decompos) = this->GetParam();

        ngraph::Shape data_shape, il_shape, ih_shape, ol_shape, oh_shape;
        ngraph::element::Type_t data_prec, ranges_prec;
        size_t levels;
        std::tie(data_prec, data_shape, ranges_prec, il_shape, ih_shape, ol_shape, oh_shape, levels) = basic_params;

        bool need_convert = data_prec != ranges_prec;

        std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
        {
            const auto data = std::make_shared<ngraph::opset1::Parameter>(data_prec, ngraph::PartialShape(data_shape));
            const auto il = std::make_shared<ngraph::opset1::Constant>(ranges_prec, il_shape, input_ranges_values.first);
            const auto ih = std::make_shared<ngraph::opset1::Constant>(ranges_prec, ih_shape, input_ranges_values.second);
            const auto ol = std::make_shared<ngraph::opset1::Constant>(ranges_prec, ol_shape);
            const auto oh = std::make_shared<ngraph::opset1::Constant>(ranges_prec, oh_shape);

            const auto fq = std::make_shared<ngraph::opset1::FakeQuantize>(data, il, ih, ol, oh, levels);
            f = std::make_shared<ngraph::Function>(ngraph::NodeVector{fq}, ngraph::ParameterVector{data});

            ngraph::pass::Manager manager;
            manager.register_pass<ngraph::pass::InitNodeInfo>();
            manager.register_pass<ngraph::pass::FakeQuantizeDecomposition>();
            manager.run_passes(f);

            ASSERT_NO_THROW(check_rt_info(f));
        }

        {
            auto input_data = std::make_shared<ngraph::opset1::Parameter>(data_prec, ngraph::PartialShape(data_shape));
            ngraph::ParameterVector params;
            params.push_back(input_data);
            std::shared_ptr<ngraph::Node> data = input_data;
            const auto il = std::make_shared<ngraph::opset1::Constant>(ranges_prec, il_shape, input_ranges_values.first);
            const auto ih = std::make_shared<ngraph::opset1::Constant>(ranges_prec, ih_shape, input_ranges_values.second);
            const auto ol = std::make_shared<ngraph::opset1::Constant>(ranges_prec, ol_shape);
            const auto oh = std::make_shared<ngraph::opset1::Constant>(ranges_prec, oh_shape);

            if (should_be_decompos) {
                if (need_convert) {
                    data = std::make_shared<ngraph::opset1::Convert>(data, ranges_prec);
                }

                const auto max = std::make_shared<ngraph::opset1::Maximum>(data, il);
                const auto min = std::make_shared<ngraph::opset1::Minimum>(max, ih);

                const auto levels_minus_one = std::make_shared<ngraph::opset1::Constant>(ranges_prec, ngraph::Shape{}, levels - 1);

                const auto sub_in_high_low = std::make_shared<ngraph::opset1::Subtract>(ih, il);
                const auto isc = std::make_shared<ngraph::opset1::Divide>(levels_minus_one, sub_in_high_low);
                const auto ish = std::make_shared<ngraph::opset1::Multiply>(il, isc);

                const auto after_isc_apply = std::make_shared<ngraph::opset1::Multiply>(min, isc);
                const auto after_ish_apply = std::make_shared<ngraph::opset1::Subtract>(after_isc_apply, ish);

                const auto round = std::make_shared<ngraph::opset5::Round>(after_ish_apply, ngraph::opset5::Round::RoundMode::HALF_TO_EVEN);

                const auto sub_out_high_low = std::make_shared<ngraph::opset1::Subtract>(oh, ol);
                const auto osc = std::make_shared<ngraph::opset1::Divide>(sub_out_high_low, levels_minus_one);

                const auto after_osc_apply = std::make_shared<ngraph::opset1::Multiply>(round, osc);
                const auto after_out_low_add = std::make_shared<ngraph::opset1::Add>(after_osc_apply, ol);
                std::shared_ptr<ngraph::Node> result = after_out_low_add;

                if (need_convert) {
                    result = std::make_shared<ngraph::opset1::Convert>(result, data_prec);
                }

                f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{result}, params);
            } else {
                const auto fq = std::make_shared<ngraph::opset1::FakeQuantize>(data, il, ih, ol, oh, levels);
                f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fq}, params);
            }
        }

        const auto res = compare_functions(f, f_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
};

TEST_P(FakeQuantizeDecompositionTest, CompareFunctions) {}

const std::vector<ngraph::element::Type_t> precisions = {ngraph::element::Type_t::f16, ngraph::element::Type_t::f32};

const std::vector<size_t> levels = {16, 255, 256};

const std::vector<std::pair<float, float>> input_ranges_supported = {
    {-10.0f, 10.f}
};

const auto simple_fq_basic = ::testing::Combine(::testing::ValuesIn(precisions),
                                                ::testing::Values(ngraph::Shape{2, 3, 4, 5}),
                                                ::testing::ValuesIn(precisions),
                                                ::testing::Values(ngraph::Shape{1, 3, 1, 1}),
                                                ::testing::Values(ngraph::Shape{1, 3, 1, 1}),
                                                ::testing::Values(ngraph::Shape{1, 3, 1, 1}),
                                                ::testing::Values(ngraph::Shape{1, 3, 1, 1}),
                                                ::testing::ValuesIn(levels));

const auto broadcast_fq_basic = ::testing::Combine(::testing::ValuesIn(precisions),
                                                   ::testing::Values(ngraph::Shape{2, 3, 4, 5}),
                                                   ::testing::ValuesIn(precisions),
                                                   ::testing::Values(ngraph::Shape{1, 3, 4, 1}),
                                                   ::testing::Values(ngraph::Shape{1, 1, 4, 5}),
                                                   ::testing::Values(ngraph::Shape{1, 1, 1, 1}),
                                                   ::testing::Values(ngraph::Shape{1, 1, 1, 1}),
                                                   ::testing::ValuesIn(levels));

const auto elementwise_fq_basic = ::testing::Combine(::testing::ValuesIn(precisions),
                                                     ::testing::Values(ngraph::Shape{2, 3, 4, 5}),
                                                     ::testing::ValuesIn(precisions),
                                                     ::testing::Values(ngraph::Shape{2, 3, 4, 5}),
                                                     ::testing::Values(ngraph::Shape{2, 3, 4, 1}),
                                                     ::testing::Values(ngraph::Shape{2, 3, 4, 5}),
                                                     ::testing::Values(ngraph::Shape{2, 3, 4, 5}),
                                                     ::testing::ValuesIn(levels));

const auto broadcast_6D_fq_basic = ::testing::Combine(::testing::ValuesIn(precisions),
                                                      ::testing::Values(ngraph::Shape{2, 3, 4, 5, 6, 7}),
                                                      ::testing::ValuesIn(precisions),
                                                      ::testing::Values(ngraph::Shape{2, 3, 4, 1, 1, 1}),
                                                      ::testing::Values(ngraph::Shape{1, 3, 4, 5, 1, 1}),
                                                      ::testing::Values(ngraph::Shape{1, 1, 1, 5, 6, 7}),
                                                      ::testing::Values(ngraph::Shape{1, 1, 1, 5, 6, 7}),
                                                      ::testing::ValuesIn(levels));

INSTANTIATE_TEST_SUITE_P(SimpleFakeQuantize_Decomposition, FakeQuantizeDecompositionTest,
                        ::testing::Combine(
                            simple_fq_basic,
                            ::testing::ValuesIn(input_ranges_supported),
                            ::testing::Values(true)),
                        FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(BroadcastFakeQuantize_Decomposition, FakeQuantizeDecompositionTest,
                        ::testing::Combine(
                            broadcast_fq_basic,
                            ::testing::ValuesIn(input_ranges_supported),
                            ::testing::Values(true)),
                        FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ElementwiseFakeQuantize_Decomposition, FakeQuantizeDecompositionTest,
                        ::testing::Combine(
                            elementwise_fq_basic,
                            ::testing::ValuesIn(input_ranges_supported),
                            ::testing::Values(true)),
                        FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(FakeQuantize6D_Decomposition, FakeQuantizeDecompositionTest,
                        ::testing::Combine(
                            broadcast_6D_fq_basic,
                            ::testing::ValuesIn(input_ranges_supported),
                            ::testing::Values(true)),
                        FakeQuantizeDecompositionTest::getTestCaseName);

const std::vector<std::pair<float, float>> input_ranges_unsupported = {
    {10.0f, -10.f},
    {5.0f, 5.0f},
    {-5.0f, -5.0f}
};

INSTANTIATE_TEST_SUITE_P(SimpleFakeQuantize_NoDecomposition, FakeQuantizeDecompositionTest,
                        ::testing::Combine(
                            simple_fq_basic,
                            ::testing::ValuesIn(input_ranges_unsupported),
                            ::testing::Values(false)),
                        FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(BroadcastFakeQuantize_NoDecomposition, FakeQuantizeDecompositionTest,
                        ::testing::Combine(
                            broadcast_fq_basic,
                            ::testing::ValuesIn(input_ranges_unsupported),
                            ::testing::Values(false)),
                        FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ElementwiseFakeQuantize_NoDecomposition, FakeQuantizeDecompositionTest,
                        ::testing::Combine(
                            elementwise_fq_basic,
                            ::testing::ValuesIn(input_ranges_unsupported),
                            ::testing::Values(false)),
                        FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(FakeQuantize6D_NoDecomposition, FakeQuantizeDecompositionTest,
                        ::testing::Combine(
                            broadcast_6D_fq_basic,
                            ::testing::ValuesIn(input_ranges_unsupported),
                            ::testing::Values(false)),
                        FakeQuantizeDecompositionTest::getTestCaseName);
