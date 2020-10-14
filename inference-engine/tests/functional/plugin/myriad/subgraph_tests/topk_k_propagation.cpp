// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_common.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>
#include <ngraph/function.hpp>
#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/operations/static_shape_topk.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape_topk.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>

namespace {

class DynamicToStaticTopKPropagation : public CommonTestUtils::TestsCommon,
        public testing::WithParamInterface<int64_t> {
public:
    void SetUp() override {
        const auto& k = GetParam();

        const auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::i64, ngraph::Shape{1000});
        const auto realK = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::i32, ngraph::Shape{1});
        const auto maxK = ngraph::opset4::Constant::create(ngraph::element::i32, {1}, {k});

        const auto concat = std::make_shared<ngraph::opset4::Concat>(ngraph::OutputVector{realK, maxK}, 0);

        const auto reduceMin = std::make_shared<ngraph::opset4::ReduceMin>(concat, ngraph::opset4::Constant::create(ngraph::element::i32, {1}, {0}), false);
        const auto builtSubgraph = buildSubgraph(reduceMin);

        const auto dsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(data, realK);
        const auto topK = std::make_shared<ngraph::opset4::TopK>(dsr, builtSubgraph, 0, "max", "value");

        ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(topK->output(0)),
                                     std::make_shared<ngraph::opset4::Result>(topK->output(1))};
        const auto function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{data, realK}, "TopKPropagationOfK");
        topK->set_output_type(0, dsr->get_input_element_type(0), ngraph::PartialShape::dynamic(1));

        const auto transformations = vpu::Transformations{{topK->type_info, vpu::dynamicToStaticShapeTopK}};
        ASSERT_NO_THROW(vpu::DynamicToStaticShape(transformations).run_on_function(function));

        ngraph::ResultVector processedResults;
        ASSERT_NO_THROW(processedResults = function->get_results());
        EXPECT_EQ(processedResults.size(), 2);

        const auto topKOutPartialShape = processedResults[0]->get_input_partial_shape(0);
        EXPECT_TRUE(topKOutPartialShape.is_static());

        const auto topKOutShape = topKOutPartialShape.get_shape();
        EXPECT_EQ(topKOutShape.size(), 1);
        EXPECT_EQ(topKOutShape[0], k);
    }

protected:
    virtual std::shared_ptr<ngraph::Node> buildSubgraph(std::shared_ptr<ngraph::Node> node) const {
        return node;
    }
};

const std::vector<int64_t> kVec = {0, 10, 100, 200, 500};

TEST_P(DynamicToStaticTopKPropagation, KPropagation) {
}

INSTANTIATE_TEST_CASE_P(smoke_NGraph, DynamicToStaticTopKPropagation, ::testing::ValuesIn(kVec));

class DynamicToStaticTopKPropagationReshape : public DynamicToStaticTopKPropagation {
protected:
    std::shared_ptr<ngraph::Node> buildSubgraph(std::shared_ptr<ngraph::Node> node) const override {
        return std::make_shared<ngraph::opset4::Reshape>(node, ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}), false);
    }
};

TEST_P(DynamicToStaticTopKPropagationReshape, KPropagation) {
}

INSTANTIATE_TEST_CASE_P(smoke_NGraph, DynamicToStaticTopKPropagationReshape, ::testing::ValuesIn(kVec));

class DynamicToStaticTopKPropagationSqueezeUnsqueeze : public DynamicToStaticTopKPropagation {
protected:
    std::shared_ptr<ngraph::Node> buildSubgraph(std::shared_ptr<ngraph::Node> node) const override {
        const auto unsqueeze = std::make_shared<ngraph::opset4::Unsqueeze>(node, ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
        return std::make_shared<ngraph::opset4::Squeeze>(unsqueeze, ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
    }
};

TEST_P(DynamicToStaticTopKPropagationSqueezeUnsqueeze, KPropagation) {
}

INSTANTIATE_TEST_CASE_P(smoke_NGraph, DynamicToStaticTopKPropagationSqueezeUnsqueeze, ::testing::ValuesIn(kVec));

class DynamicToStaticTopKPropagationConvert : public DynamicToStaticTopKPropagation {
protected:
    std::shared_ptr<ngraph::Node> buildSubgraph(std::shared_ptr<ngraph::Node> node) const override {
        const auto convert = std::make_shared<ngraph::opset4::Convert>(node, ngraph::element::i32);
        return std::make_shared<ngraph::opset4::Convert>(convert, ngraph::element::i64);
    }
};

TEST_P(DynamicToStaticTopKPropagationConvert, KPropagation) {
}

INSTANTIATE_TEST_CASE_P(smoke_NGraph, DynamicToStaticTopKPropagationConvert, ::testing::ValuesIn(kVec));

}  // namespace
