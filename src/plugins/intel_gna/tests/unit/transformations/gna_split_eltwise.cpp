// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/split_eltwise.hpp"

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include <legacy/ngraph_ops/eltwise.hpp>
#include <layers/gna_split_layer.hpp>

namespace testing {
namespace {

static std::shared_ptr<ngraph::Function> createFunction(const ngraph::Shape& input_shape,
                                                        bool with_const,
                                                        bool with_fq,
                                                        ELTWISE_TYPE type,
                                                        bool split) {
    std::shared_ptr<ngraph::Node> last_node, last_node0, last_node1;

    ngraph::ParameterVector parameters;
    auto input0 = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, input_shape);
    parameters.push_back(input0);
    last_node0 = input0;
    std::shared_ptr<ngraph::Node> input1;
    if (with_const) {
        auto const_input = ngraph::opset9::Constant::create(ngraph::element::f32, input_shape, {1});
        last_node1 = const_input;
    } else {
        auto input1 = std::make_shared<ngraph::opset9::Parameter>(ngraph::element::f32, input_shape);
        last_node1 = input1;
        parameters.push_back(input1);
    }

    auto add_fake_quantize = [&](const std::shared_ptr<ngraph::Node>& node) {
        auto input_low = ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1});
        auto input_high = ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {5});
        auto output_low = ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0});
        auto output_high = ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {10});
        return std::make_shared<ngraph::opset9::FakeQuantize>(node, input_low, input_high, output_low, output_high, 11);
    };

    if (with_fq) {
        auto fq_eltwise_input0 = add_fake_quantize(last_node0);
        last_node0 = fq_eltwise_input0;
        auto fq_eltwise_input1 = add_fake_quantize(last_node1);
        last_node1 = fq_eltwise_input1;
    }

    if (split) {
        auto split_sizes_per_axis = GNAPluginNS::AlignedSplitSizesPerAxis(input_shape);
        auto split0 = std::make_shared<ngraph::opset9::VariadicSplit>(last_node0,
                ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape({1}), std::vector<int64_t>{split_sizes_per_axis.first}),
                ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape({split_sizes_per_axis.second.size()}), split_sizes_per_axis.second));
        auto split1 = std::make_shared<ngraph::opset9::VariadicSplit>(last_node1,
                ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape({1}), std::vector<int64_t>{split_sizes_per_axis.first}),
                ngraph::opset9::Constant::create(ngraph::element::i64, ngraph::Shape({split_sizes_per_axis.second.size()}), split_sizes_per_axis.second));
        ov::NodeVector concat_inputs;
        for (size_t i = 0; i < split_sizes_per_axis.second.size(); i++) {
            auto eltwise_node_part = std::make_shared<ngraph::op::Eltwise>(split0->output(i), split1->output(i), type);
            concat_inputs.push_back(eltwise_node_part);
        }
        auto concat = std::make_shared<ngraph::opset9::Concat>(concat_inputs, split_sizes_per_axis.first);
        auto result = std::make_shared<ngraph::opset9::Result>(concat);
        return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, parameters);
    } else {
        auto eltwise = std::make_shared<ngraph::op::Eltwise>(last_node0, last_node1, type);
        auto result = std::make_shared<ngraph::opset9::Result>(eltwise);
        return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, parameters);
    }
}

typedef std::tuple<
        ngraph::Shape,
        bool,                               // with const
        bool,                               // with fq
        ELTWISE_TYPE                        // eltwise type
> EltwiseSplitParams;

static std::string getTestCaseName(testing::TestParamInfo<EltwiseSplitParams> obj) {
    ngraph::Shape shape;
    bool with_const;
    bool with_fq;
    ELTWISE_TYPE type;
    std::tie(shape, with_const, with_fq, type) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(shape) << "_";
    result << "wConst=" << with_const << "_";
    result << "wFQ=" << with_fq << "_";
    result << "type=";
    switch (type) {
    case ELTWISE_TYPE::Sum:
        result << "sum";
        break;
    case ELTWISE_TYPE::Sub:
        result << "sub";
        break;
    case ELTWISE_TYPE::Prod:
        result << "prod";
        break;
    default:
        break;
    }
    return result.str();
}

class SplitEltwiseTestSuiteFixture: public CommonTestUtils::TestsCommon,
                               public ::testing::WithParamInterface<EltwiseSplitParams> {
public:
    void SetUp() override;
public:
    std::shared_ptr<ngraph::Function> function, reference_function;
};

void SplitEltwiseTestSuiteFixture::SetUp() {
    ngraph::Shape shape;
    bool with_const;
    bool with_fq;
    ELTWISE_TYPE type;
    std::tie(shape, with_const, with_fq, type) = this->GetParam();
    function = createFunction(shape, with_const, with_fq, type, false);
    reference_function = createFunction(shape, with_const, with_fq, type, true);
}

void execute_test(std::shared_ptr<ngraph::Function> function,
                  std::shared_ptr<ngraph::Function> reference_function) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::SplitEltwise>();
    manager.run_passes(function);
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid) << result.message;
}

TEST_P(SplitEltwiseTestSuiteFixture, CompareFunctions) {
    execute_test(function, reference_function);
}

const std::vector<ov::Shape> inputShape = {
    {1, 67000},
    {1, 500000},
    {1, 936, 513},
    {1, 64, 64, 64},
    {1, 256, 64, 64}
};

INSTANTIATE_TEST_SUITE_P(SplitEltwiseTestSuite, SplitEltwiseTestSuiteFixture,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputShape),
                            ::testing::ValuesIn(std::vector<bool>{true, false}),                                                       // with const
                            ::testing::ValuesIn(std::vector<bool>{true, false}),                                                       // with fq
                            ::testing::ValuesIn(std::vector<ELTWISE_TYPE>{ELTWISE_TYPE::Sum, ELTWISE_TYPE::Sub, ELTWISE_TYPE::Prod})), // eltwise type
                            getTestCaseName);

} // namespace
} // namespace testing
