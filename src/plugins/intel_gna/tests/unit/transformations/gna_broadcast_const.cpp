// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "legacy/ngraph_ops/eltwise.hpp"
#include "legacy/ngraph_ops/scaleshift.hpp"
#include "transformations/broadcast_const.hpp"

namespace testing {

// ------------------------------------------------------------------------------------------------

namespace {

// TODO: use std::make_unique when C++14 will be available
template <typename T, typename... Args>
std::unique_ptr<T> createUnique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

std::shared_ptr<ngraph::opset8::FakeQuantize> createFakeQuantizeNode(std::shared_ptr<ngraph::op::Op> parent_node) {
    auto input_low = ngraph::opset8::Constant::create(ngraph::element::f32, {}, {-0.5});
    auto input_high = ngraph::opset8::Constant::create(ngraph::element::f32, {}, {0.5});
    auto output_low = ngraph::opset8::Constant::create(ngraph::element::f32, {}, {-0.5});
    auto output_high = ngraph::opset8::Constant::create(ngraph::element::f32, {}, {0.5});
    return std::make_shared<ngraph::opset8::FakeQuantize>(parent_node,
                                                          input_low,
                                                          input_high,
                                                          output_low,
                                                          output_high,
                                                          0);
}

using Node = std::shared_ptr<ov::op::Op>;

// ------------------------------------------------------------------------------------------------

struct ShapeInfo {
    ShapeInfo(const ngraph::Shape& a_data_shape,
              const ngraph::Shape& a_const_shape_dims_in,
              const ngraph::Shape& a_const_shape_values_in,
              const ngraph::Shape& a_const_shape_values_out)
        : data_shape(a_data_shape),
          const_shape_dims_in(a_const_shape_dims_in),
          const_shape_values_in(a_const_shape_values_in),
          const_shape_values_out(a_const_shape_values_out) {}

    ngraph::Shape data_shape;
    ngraph::Shape const_shape_dims_in;
    ngraph::Shape const_shape_values_in;
    ngraph::Shape const_shape_values_out;
};

std::unordered_map<ov::op::AutoBroadcastType, ShapeInfo> ShapesRightConst = {
    {ov::op::AutoBroadcastType::NONE,
     ShapeInfo(/* data_shape */ {3, 2},
               /* const_shape_dims_in */ {3, 2},
               /* const_shape_values_in */ {1, 2, 1, 2, 1, 2},
               /* const_shape_values_out */ {1, 2, 1, 2, 1, 2})},
    {ov::op::AutoBroadcastType::EXPLICIT,
     ShapeInfo(/* data_shape */ {3, 2},
               /* const_shape_dims_in */ {3, 2},
               /* const_shape_values_in */ {1, 2, 1, 2, 1, 2},
               /* const_shape_values_out */ {1, 2, 1, 2, 1, 2})},
    {ov::op::AutoBroadcastType::NUMPY,
     ShapeInfo(/* data_shape */ {3, 2},
               /* const_shape_dims_in */ {2},
               /* const_shape_values_in */ {1, 2},
               /* const_shape_values_out */ {1, 2, 1, 2, 1, 2})},
    {ov::op::AutoBroadcastType::PDPD,
     ShapeInfo(/* data_shape */ {3, 2},
               /* const_shape_dims_in */ {3, 1},
               /* const_shape_values_in */ {1, 2, 3},
               /* const_shape_values_out */ {1, 1, 2, 2, 3, 3})}};

// ------------------------------------------------------------------------------------------------

class IEltwiseFactory {
public:
    IEltwiseFactory() = default;
    virtual ~IEltwiseFactory() = default;
    virtual Node CreateNode(Node left_input, Node right_input) const = 0;

    void SetBroadcastType(ov::op::AutoBroadcastType type) {
        m_broadcast_type = type;
    }
    ov::op::AutoBroadcastType GetBroadcastType() const {
        return m_broadcast_type;
    }

private:
    ov::op::AutoBroadcastType m_broadcast_type = ov::op::AutoBroadcastType::NUMPY;
};

using EltwiseFactoryPtr = std::shared_ptr<IEltwiseFactory>;

template <typename EltwiseT>
class EltwiseFactory : public IEltwiseFactory {
public:
    Node CreateNode(Node left_input, Node right_input) const override {
        return std::make_shared<EltwiseT>(left_input, right_input, GetBroadcastType());
    }
};

template <>
class EltwiseFactory<ngraph::op::Eltwise> : public IEltwiseFactory {
public:
    Node CreateNode(Node left_input, Node right_input) const override {
        return std::make_shared<ngraph::op::Eltwise>(left_input, right_input, ELTWISE_TYPE::Sum);
    }
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
                                                 bool add_scaleshift,
                                                 EltwiseFactoryPtr eltwise_factory) {
    const auto input_params_1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::Type_t::f32, data_shape);
    ngraph::ParameterVector params{input_params_1};

    const auto constant_1 = ngraph::opset8::Constant::create(ngraph::element::Type_t::f32,
                                                             ngraph::Shape{const_shape_dims},
                                                             const_shape_values);

    Node const_last_node = constant_1;

    if (add_scaleshift) {
        const auto input_params_2 =
            std::make_shared<ngraph::opset8::Parameter>(ngraph::element::Type_t::f32, data_shape);
        params.push_back(input_params_2);

        const auto constant_2 = ngraph::opset8::Constant::create(ngraph::element::Type_t::f32,
                                                                 ngraph::Shape{const_shape_dims},
                                                                 const_shape_values);

        const_last_node = std::make_shared<ngraph::op::ScaleShiftIE>(input_params_2,
                                                                     constant_1,
                                                                     constant_2,
                                                                     ngraph::element::Type_t::f32);
    }

    if (add_const_fake_quantize) {
        const auto fake_quantize = createFakeQuantizeNode(const_last_node);
        const_last_node = fake_quantize;
    }

    Node input_last_node = input_params_1;

    if (add_input_fake_quantize) {
        const auto fake_quantize = createFakeQuantizeNode(input_last_node);
        input_last_node = fake_quantize;
    }

    Node left_node = input_last_node;
    Node right_node = const_last_node;

    if (swap_outputs)
        left_node.swap(right_node);

    const auto add = eltwise_factory->CreateNode(left_node, right_node);

    const auto result = std::make_shared<ngraph::opset8::Result>(add);
    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, params);
}

}  // namespace

// ------------------------------------------------------------------------------------------------

class BroadcastConstTestFixture : public ov::test::TestsCommon,
                                  public ::testing::WithParamInterface<std::tuple<EltwiseFactoryPtr,
                                                                                  bool /* add_input_fake_quantize */,
                                                                                  bool /* add_const_fake_quantize */,
                                                                                  bool /* swap_outputs */,
                                                                                  bool /* add_scaleshift */,
                                                                                  ov::op::AutoBroadcastType>> {
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
    bool add_scaleshift;
    ov::op::AutoBroadcastType broadcast_type;
    std::tie(eltwise_factory,
             add_input_fake_quantize,
             add_const_fake_quantize,
             swap_outputs,
             add_scaleshift,
             broadcast_type) = this->GetParam();

    eltwise_factory->SetBroadcastType(broadcast_type);

    ShapeInfo shape_info = ShapesRightConst.at(broadcast_type);

    function = CreateFunction(shape_info.data_shape,
                              shape_info.const_shape_dims_in,
                              shape_info.const_shape_values_in,
                              add_input_fake_quantize,
                              add_const_fake_quantize,
                              swap_outputs,
                              add_scaleshift,
                              eltwise_factory);
    reference_function = CreateFunction(shape_info.data_shape,
                                        shape_info.data_shape,
                                        shape_info.const_shape_values_out,
                                        add_input_fake_quantize,
                                        add_const_fake_quantize,
                                        swap_outputs,
                                        add_scaleshift,
                                        eltwise_factory);
}

void execute_test(std::shared_ptr<ngraph::Function> function, std::shared_ptr<ngraph::Function> reference_function) {
    ngraph::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::BroadcastAddMultiplyConst>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

void execute_cloned_test(std::shared_ptr<ngraph::Function> function) {
    auto reference_function = ngraph::clone_function(*function);

    ngraph::pass::Manager manager;
    manager.register_pass<ov::pass::InitNodeInfo>();
    manager.register_pass<ov::intel_gna::pass::BroadcastAddMultiplyConst>();
    manager.run_passes(function);
    ASSERT_NO_THROW(check_rt_info(function));

    FunctionsComparator func_comparator = FunctionsComparator::with_default();
    func_comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    const FunctionsComparator::Result result = func_comparator(function, reference_function);
    ASSERT_TRUE(result.valid);
}

// ------------------------------------------------------------------------------------------------

namespace {

std::vector<EltwiseFactoryPtr> opset8_eltwise_factories = {CreateEltwiseFactory<ngraph::opset8::Add>(),
                                                           CreateEltwiseFactory<ngraph::opset8::Subtract>(),
                                                           CreateEltwiseFactory<ngraph::opset8::Multiply>()};

std::vector<EltwiseFactoryPtr> all_eltwise_factories = {CreateEltwiseFactory<ngraph::opset8::Add>(),
                                                        CreateEltwiseFactory<ngraph::opset8::Subtract>(),
                                                        CreateEltwiseFactory<ngraph::opset8::Multiply>(),
                                                        CreateEltwiseFactory<ngraph::op::Eltwise>()};

std::vector<ov::op::AutoBroadcastType> broadcast_passed_types = {ov::op::AutoBroadcastType::NONE,
                                                                 ov::op::AutoBroadcastType::EXPLICIT};

}  // namespace

TEST_P(BroadcastConstTestFixture, CompareFunctions) {
    execute_test(function, reference_function);
}

INSTANTIATE_TEST_SUITE_P(BroadcastConstTestNumpySuite,
                         BroadcastConstTestFixture,
                         ::testing::Combine(::testing::ValuesIn(all_eltwise_factories),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Values(ov::op::AutoBroadcastType::NUMPY)));

INSTANTIATE_TEST_SUITE_P(BroadcastConstTestPDPDSuite,
                         BroadcastConstTestFixture,
                         ::testing::Combine(::testing::ValuesIn(opset8_eltwise_factories),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Values(false),
                                            ::testing::Bool(),
                                            ::testing::Values(ov::op::AutoBroadcastType::PDPD)));

// ------------------------------------------------------------------------------------------------

class BroadcastConstTestPassedFixture
    : public ov::test::TestsCommon,
      public ::testing::WithParamInterface<std::tuple<EltwiseFactoryPtr,
                                                      bool /* add_input_fake_quantize */,
                                                      bool /* add_const_fake_quantize */,
                                                      bool /* swap_outputs */,
                                                      bool /* add_scaleshift */,
                                                      ov::op::AutoBroadcastType>> {
public:
    void SetUp() override;

public:
    std::shared_ptr<ngraph::Function> function;
};

void BroadcastConstTestPassedFixture::SetUp() {
    // TODO: use auto & [ ... ] = this->GetParam() when C++17
    EltwiseFactoryPtr eltwise_factory;
    bool add_input_fake_quantize;
    bool add_const_fake_quantize;
    bool swap_outputs;
    bool add_scaleshift;
    ov::op::AutoBroadcastType broadcast_type;
    std::tie(eltwise_factory,
             add_input_fake_quantize,
             add_const_fake_quantize,
             swap_outputs,
             add_scaleshift,
             broadcast_type) = this->GetParam();

    eltwise_factory->SetBroadcastType(broadcast_type);

    ShapeInfo shape_info = ShapesRightConst.at(broadcast_type);

    function = CreateFunction(shape_info.data_shape,
                              shape_info.const_shape_dims_in,
                              shape_info.const_shape_values_in,
                              add_input_fake_quantize,
                              add_const_fake_quantize,
                              swap_outputs,
                              add_scaleshift,
                              eltwise_factory);
}

TEST_P(BroadcastConstTestPassedFixture, CompareFunctionsPassedTypes) {
    execute_cloned_test(function);
}

INSTANTIATE_TEST_SUITE_P(BroadcastConstTestPassedSuite,
                         BroadcastConstTestPassedFixture,
                         ::testing::Combine(::testing::ValuesIn(opset8_eltwise_factories),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::Bool(),
                                            ::testing::ValuesIn(broadcast_passed_types)));

}  // namespace testing
