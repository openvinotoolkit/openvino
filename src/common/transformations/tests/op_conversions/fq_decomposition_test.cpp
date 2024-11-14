// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/fq_decomposition.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

using FakeQuantizeDecompositionBasicParams = std::tuple<element::Type_t,  // 'data' input precision
                                                        Shape,            // data shape
                                                        element::Type_t,  // 'range' inputs precision
                                                        Shape,            // il shape
                                                        Shape,            // ih shape
                                                        Shape,            // ol shape
                                                        Shape,            // oh shape
                                                        size_t            // levels
                                                        >;

using FakeQuantizeDecompositionParamsSet = std::tuple<FakeQuantizeDecompositionBasicParams,
                                                      std::pair<float, float>,  // il and ih values
                                                      bool                      // should be decompos
                                                      >;

class FakeQuantizeDecompositionTest : public ov::test::TestsCommon,
                                      public ::testing::WithParamInterface<FakeQuantizeDecompositionParamsSet> {
public:
    static std::string getTestCaseName(::testing::TestParamInfo<FakeQuantizeDecompositionParamsSet> obj) {
        FakeQuantizeDecompositionBasicParams basic_params;
        std::pair<float, float> input_ranges_values;
        bool should_be_decompos;
        std::tie(basic_params, input_ranges_values, should_be_decompos) = obj.param;

        Shape data_shape, il_shape, ih_shape, ol_shape, oh_shape;
        element::Type_t data_prec, ranges_prec;
        size_t levels;
        std::tie(data_prec, data_shape, ranges_prec, il_shape, ih_shape, ol_shape, oh_shape, levels) = basic_params;

        std::ostringstream result;
        result << "DATA=" << ov::test::utils::vec2str(data_shape) << "_";
        result << "DATA_PRC=" << element::Type(data_prec) << "_";
        result << "IL=" << ov::test::utils::vec2str(il_shape) << "_" << input_ranges_values.first << "_";
        result << "IH=" << ov::test::utils::vec2str(ih_shape) << "_" << input_ranges_values.second << "_";
        result << "OL=" << ov::test::utils::vec2str(ol_shape) << "_";
        result << "OH=" << ov::test::utils::vec2str(oh_shape) << "_";
        result << "RANGES_PRC=" << element::Type(ranges_prec) << "_";
        result << "LEVELS=" << levels;
        return result.str();
    }

protected:
    void SetUp() override {
        FakeQuantizeDecompositionBasicParams basic_params;
        std::pair<float, float> input_ranges_values;
        bool should_be_decompos;
        std::tie(basic_params, input_ranges_values, should_be_decompos) = this->GetParam();

        Shape data_shape, il_shape, ih_shape, ol_shape, oh_shape;
        element::Type_t data_prec, ranges_prec;
        size_t levels;
        std::tie(data_prec, data_shape, ranges_prec, il_shape, ih_shape, ol_shape, oh_shape, levels) = basic_params;

        bool need_convert = data_prec != ranges_prec;

        std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
        {
            const auto data = std::make_shared<opset1::Parameter>(data_prec, PartialShape(data_shape));
            const auto il = std::make_shared<opset1::Constant>(ranges_prec, il_shape, input_ranges_values.first);
            const auto ih = std::make_shared<opset1::Constant>(ranges_prec, ih_shape, input_ranges_values.second);
            const auto ol = std::make_shared<opset1::Constant>(ranges_prec, ol_shape);
            const auto oh = std::make_shared<opset1::Constant>(ranges_prec, oh_shape);

            const auto fq = std::make_shared<opset1::FakeQuantize>(data, il, ih, ol, oh, levels);
            f = std::make_shared<ov::Model>(NodeVector{fq}, ParameterVector{data});

            pass::Manager manager;
            manager.register_pass<ov::pass::InitNodeInfo>();
            manager.register_pass<ov::pass::FakeQuantizeDecomposition>();
            manager.run_passes(f);

            OV_ASSERT_NO_THROW(check_rt_info(f));
        }

        {
            auto input_data = std::make_shared<opset1::Parameter>(data_prec, PartialShape(data_shape));
            ParameterVector params;
            params.push_back(input_data);
            std::shared_ptr<Node> data = input_data;
            const auto il = std::make_shared<opset1::Constant>(ranges_prec, il_shape, input_ranges_values.first);
            const auto ih = std::make_shared<opset1::Constant>(ranges_prec, ih_shape, input_ranges_values.second);
            const auto ol = std::make_shared<opset1::Constant>(ranges_prec, ol_shape);
            const auto oh = std::make_shared<opset1::Constant>(ranges_prec, oh_shape);

            if (should_be_decompos) {
                if (need_convert) {
                    data = std::make_shared<opset1::Convert>(data, ranges_prec);
                }

                const auto max = std::make_shared<opset1::Maximum>(data, il);
                const auto min = std::make_shared<opset1::Minimum>(max, ih);

                const auto levels_minus_one = std::make_shared<opset1::Constant>(ranges_prec, Shape{}, levels - 1);

                const auto sub_in_high_low = std::make_shared<opset1::Subtract>(ih, il);
                const auto isc = std::make_shared<opset1::Divide>(levels_minus_one, sub_in_high_low);
                const auto ish = std::make_shared<opset1::Multiply>(il, isc);

                const auto after_isc_apply = std::make_shared<opset1::Multiply>(min, isc);
                const auto after_ish_apply = std::make_shared<opset1::Subtract>(after_isc_apply, ish);

                const auto round =
                    std::make_shared<opset5::Round>(after_ish_apply, opset5::Round::RoundMode::HALF_TO_EVEN);

                const auto sub_out_high_low = std::make_shared<opset1::Subtract>(oh, ol);
                const auto osc = std::make_shared<opset1::Divide>(sub_out_high_low, levels_minus_one);

                const auto after_osc_apply = std::make_shared<opset1::Multiply>(round, osc);
                const auto after_out_low_add = std::make_shared<opset1::Add>(after_osc_apply, ol);
                std::shared_ptr<Node> result = after_out_low_add;

                if (need_convert) {
                    result = std::make_shared<opset1::Convert>(result, data_prec);
                }

                f_ref = std::make_shared<ov::Model>(NodeVector{result}, params);
            } else {
                const auto fq = std::make_shared<opset1::FakeQuantize>(data, il, ih, ol, oh, levels);
                f_ref = std::make_shared<ov::Model>(NodeVector{fq}, params);
            }
        }

        const auto res = compare_functions(f, f_ref);
        ASSERT_TRUE(res.first) << res.second;
    }
};

TEST_P(FakeQuantizeDecompositionTest, CompareFunctions) {}

const std::vector<element::Type_t> precisions = {element::Type_t::f16, element::Type_t::f32};

const std::vector<size_t> levels = {16, 255, 256};

const std::vector<std::pair<float, float>> input_ranges_supported = {{-10.0f, 10.f}};

const auto simple_fq_basic = ::testing::Combine(::testing::ValuesIn(precisions),
                                                ::testing::Values(Shape{2, 3, 4, 5}),
                                                ::testing::ValuesIn(precisions),
                                                ::testing::Values(Shape{1, 3, 1, 1}),
                                                ::testing::Values(Shape{1, 3, 1, 1}),
                                                ::testing::Values(Shape{1, 3, 1, 1}),
                                                ::testing::Values(Shape{1, 3, 1, 1}),
                                                ::testing::ValuesIn(levels));

const auto broadcast_fq_basic = ::testing::Combine(::testing::ValuesIn(precisions),
                                                   ::testing::Values(Shape{2, 3, 4, 5}),
                                                   ::testing::ValuesIn(precisions),
                                                   ::testing::Values(Shape{1, 3, 4, 1}),
                                                   ::testing::Values(Shape{1, 1, 4, 5}),
                                                   ::testing::Values(Shape{1, 1, 1, 1}),
                                                   ::testing::Values(Shape{1, 1, 1, 1}),
                                                   ::testing::ValuesIn(levels));

const auto elementwise_fq_basic = ::testing::Combine(::testing::ValuesIn(precisions),
                                                     ::testing::Values(Shape{2, 3, 4, 5}),
                                                     ::testing::ValuesIn(precisions),
                                                     ::testing::Values(Shape{2, 3, 4, 5}),
                                                     ::testing::Values(Shape{2, 3, 4, 1}),
                                                     ::testing::Values(Shape{2, 3, 4, 5}),
                                                     ::testing::Values(Shape{2, 3, 4, 5}),
                                                     ::testing::ValuesIn(levels));

const auto broadcast_6D_fq_basic = ::testing::Combine(::testing::ValuesIn(precisions),
                                                      ::testing::Values(Shape{2, 3, 4, 5, 6, 7}),
                                                      ::testing::ValuesIn(precisions),
                                                      ::testing::Values(Shape{2, 3, 4, 1, 1, 1}),
                                                      ::testing::Values(Shape{1, 3, 4, 5, 1, 1}),
                                                      ::testing::Values(Shape{1, 1, 1, 5, 6, 7}),
                                                      ::testing::Values(Shape{1, 1, 1, 5, 6, 7}),
                                                      ::testing::ValuesIn(levels));

INSTANTIATE_TEST_SUITE_P(SimpleFakeQuantize_Decomposition,
                         FakeQuantizeDecompositionTest,
                         ::testing::Combine(simple_fq_basic,
                                            ::testing::ValuesIn(input_ranges_supported),
                                            ::testing::Values(true)),
                         FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(BroadcastFakeQuantize_Decomposition,
                         FakeQuantizeDecompositionTest,
                         ::testing::Combine(broadcast_fq_basic,
                                            ::testing::ValuesIn(input_ranges_supported),
                                            ::testing::Values(true)),
                         FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ElementwiseFakeQuantize_Decomposition,
                         FakeQuantizeDecompositionTest,
                         ::testing::Combine(elementwise_fq_basic,
                                            ::testing::ValuesIn(input_ranges_supported),
                                            ::testing::Values(true)),
                         FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(FakeQuantize6D_Decomposition,
                         FakeQuantizeDecompositionTest,
                         ::testing::Combine(broadcast_6D_fq_basic,
                                            ::testing::ValuesIn(input_ranges_supported),
                                            ::testing::Values(true)),
                         FakeQuantizeDecompositionTest::getTestCaseName);

const std::vector<std::pair<float, float>> input_ranges_unsupported = {{10.0f, -10.f}, {5.0f, 5.0f}, {-5.0f, -5.0f}};

INSTANTIATE_TEST_SUITE_P(SimpleFakeQuantize_NoDecomposition,
                         FakeQuantizeDecompositionTest,
                         ::testing::Combine(simple_fq_basic,
                                            ::testing::ValuesIn(input_ranges_unsupported),
                                            ::testing::Values(false)),
                         FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(BroadcastFakeQuantize_NoDecomposition,
                         FakeQuantizeDecompositionTest,
                         ::testing::Combine(broadcast_fq_basic,
                                            ::testing::ValuesIn(input_ranges_unsupported),
                                            ::testing::Values(false)),
                         FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ElementwiseFakeQuantize_NoDecomposition,
                         FakeQuantizeDecompositionTest,
                         ::testing::Combine(elementwise_fq_basic,
                                            ::testing::ValuesIn(input_ranges_unsupported),
                                            ::testing::Values(false)),
                         FakeQuantizeDecompositionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(FakeQuantize6D_NoDecomposition,
                         FakeQuantizeDecompositionTest,
                         ::testing::Combine(broadcast_6D_fq_basic,
                                            ::testing::ValuesIn(input_ranges_unsupported),
                                            ::testing::Values(false)),
                         FakeQuantizeDecompositionTest::getTestCaseName);
