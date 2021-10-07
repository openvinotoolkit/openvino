// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "transformations/broadcast_const.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include "legacy/ngraph_ops/eltwise.hpp"

namespace testing {



// ------------------------------------------------------------------------------------------------

namespace {

std::shared_ptr<ngraph::opset8::FakeQuantize> createFakeQuantizeNode(std::shared_ptr<ngraph::op::Op> parent_node) {
    auto input_low = ngraph::opset8::Constant::create(ngraph::element::f32, {}, {-0.5});
    auto input_high = ngraph::opset8::Constant::create(ngraph::element::f32, {}, {0.5});
    auto output_low = ngraph::opset8::Constant::create(ngraph::element::f32, {}, {-0.5});
    auto output_high = ngraph::opset8::Constant::create(ngraph::element::f32, {}, {0.5});
    return std::make_shared<ngraph::opset8::FakeQuantize>(parent_node, input_low,
                                                          input_high, output_low,
                                                          output_high, 0);
}

using Node = std::shared_ptr<ov::op::Op>;

// ------------------------------------------------------------------------------------------------

class IEltwiseFactory {
public:
    virtual ~IEltwiseFactory() = default;
    virtual Node CreateNode(Node left_input, Node right_input) = 0;
};

using EltwiseFactoryPtr = std::shared_ptr<IEltwiseFactory>;

template <typename EltwiseT>
class EltwiseFactory : public IEltwiseFactory {
public:
    Node CreateNode(Node left_input, Node right_input) override { return std::make_shared<EltwiseT>(left_input, right_input); }
};

template <>
class EltwiseFactory<ngraph::op::Eltwise> : public IEltwiseFactory {
public:
    Node CreateNode(Node left_input, Node right_input) override { return std::make_shared<ngraph::op::Eltwise>(left_input, right_input, ELTWISE_TYPE::Sum); }
};

template <typename EltwiseT>
EltwiseFactoryPtr CreateEltwiseFactory() {
    return std::make_shared<EltwiseFactory<EltwiseT>>();
}

// ------------------------------------------------------------------------------------------------

std::shared_ptr<ngraph::Function> CreateFunction(const ngraph::Shape& data_shape,
                                                 const ngraph::Shape& const_shape_dims,
                                                 const ngraph::Shape& const_shape_values,
                                                 bool add_input_fake_quantize,
                                                 bool add_const_fake_quantize,
                                                 bool swap_outputs,
                                                 EltwiseFactoryPtr eltwise_factory) {
    auto input_params = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::Type_t::f32, data_shape);

    auto constant = ngraph::opset8::Constant::create(ngraph::element::Type_t::f32,
                                                     ngraph::Shape{const_shape_dims},
                                                     const_shape_values);
    Node const_last_node = constant;

    if (add_const_fake_quantize) {
        auto fake_quantize = createFakeQuantizeNode(const_last_node);
        const_last_node = fake_quantize;
    }

    Node input_last_node = input_params;
    if (add_input_fake_quantize) {
        auto fake_quantize = createFakeQuantizeNode(input_params);
        input_last_node = fake_quantize;
    }

    Node left_node = input_last_node;
    Node right_node = const_last_node;

    if (swap_outputs)
        left_node.swap(right_node);

    auto add = eltwise_factory->CreateNode(left_node, right_node);

    auto result = std::make_shared<ngraph::opset8::Result>(add);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result},
                                                  ngraph::ParameterVector{input_params});
}

} // namespace

// ------------------------------------------------------------------------------------------------

class BroadcastConstTestFixture: public CommonTestUtils::TestsCommon,
                                        public ::testing::WithParamInterface<std::tuple<EltwiseFactoryPtr,
                                                                                        bool /* add_input_fake_quantize */,
                                                                                        bool /* add_const_fake_quantize */,
                                                                                        bool /* swap_outputs */>> {
public:
    void SetUp() override;
public:
    std::shared_ptr<ngraph::Function> function, reference_function;
};

void BroadcastConstTestFixture::SetUp() {
    // TODO: use auto & [ ... ] = this->GetParam() when C++17
    EltwiseFactoryPtr eltwise_factory;
    bool add_input_fake_quantize;
    bool add_const_fake_quantize;
    bool swap_outputs;
    std::tie(eltwise_factory, add_input_fake_quantize, add_const_fake_quantize, swap_outputs) = this->GetParam();

    function = CreateFunction({3, 2} /* data_shape */,
                              {2}, /* const_shape_dims */
                              {1, 2} /* const_shape_values */,
                              add_input_fake_quantize,
                              add_const_fake_quantize,
                              swap_outputs,
                              eltwise_factory);
    reference_function = CreateFunction({3, 2} /* data_shape */,
                              {3, 2}, /* const_shape_dims */
                              {1, 2, 1, 2, 1, 2} /* const_shape_values */,
                              add_input_fake_quantize,
                              add_const_fake_quantize,
                              swap_outputs,
                              eltwise_factory);
}

void execute_test(std::shared_ptr<ngraph::Function> function,
                  std::shared_ptr<ngraph::Function> reference_function) {
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<GNAPluginNS::BroadcastAddMultiplyConst>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

// ------------------------------------------------------------------------------------------------

namespace {
std::vector<EltwiseFactoryPtr> eltwise_factories = {
    CreateEltwiseFactory<ngraph::opset8::Add>(),
    CreateEltwiseFactory<ngraph::opset8::Subtract>(),
    CreateEltwiseFactory<ngraph::opset8::Multiply>(),
    CreateEltwiseFactory<ngraph::op::Eltwise>()
};
} // namespace

TEST_P(BroadcastConstTestFixture, CompareFunctions) {
    execute_test(function, reference_function);
}

INSTANTIATE_TEST_SUITE_P(BroadcastConstTestSuite, BroadcastConstTestFixture,
                         ::testing::Combine(::testing::ValuesIn(eltwise_factories),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool()));

} // namespace testing
