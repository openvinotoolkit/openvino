// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <gmock/gmock-matchers.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <memory>
#include <queue>

#include <ngraph/pass/manager.hpp>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, CompareFunctoinsTIPositive) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto reshape_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 16});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset5::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto res_1 = std::make_shared<opset5::Result>(lstm_cell);
        auto reshape_pattern_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(lstm_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2},
                                               ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1},
                                               ngraph::ParameterVector{X, Y, Z});
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto reshape_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 16});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset5::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto res_1 = std::make_shared<opset5::Result>(lstm_cell);
        res_1->set_friendly_name("res_1");
        auto reshape_pattern_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(lstm_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        res_2->set_friendly_name("res_2");
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2},
                                               ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1},
                                               ngraph::ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    EXPECT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CompareFunctoinsTINegative) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto reshape_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 16});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset5::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto res_1 = std::make_shared<opset5::Result>(lstm_cell);
        auto reshape_pattern_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(lstm_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2},
                                               ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1},
                                               ngraph::ParameterVector{X, Y, Z});
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto reshape_pattern = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 16});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 16}, w_val);
        auto R = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512, 128}, r_val);
        auto B = ngraph::opset5::Constant::create(ngraph::element::f32, ngraph::Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset5::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto relu = std::make_shared<opset5::Relu>(lstm_cell);
        auto res_1 = std::make_shared<opset5::Result>(relu);
        res_1->set_friendly_name("res_1");
        auto reshape_pattern_2 = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(lstm_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        res_2->set_friendly_name("res_2");
        auto body = std::make_shared<Function>(OutputVector{res_1, res_2},
                                               ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{res_ti_1},
                                               ngraph::ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    EXPECT_FALSE(res.first);
    EXPECT_EQ(res.second, "LSTMCell/4 != Relu/0");
}

TEST(TransformationTests, ConstantNegativeDifferentElementType) {
    const auto createConstantFunc = [](ngraph::element::Type t) {
        using namespace ngraph::opset5;
        auto constant = Constant::create(t, ngraph::Shape{1}, {1.1});

        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{constant}, ngraph::ParameterVector{});
    };

    const auto& f1 = createConstantFunc(ngraph::element::f64);
    const auto& f2 = createConstantFunc(ngraph::element::f32);

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr("mismatch in value: 'element_type' : [f64] vs [f32]"));
}

TEST(TransformationTests, ConstantNegativeDifferentValues) {
    const auto createConstantFunc = [](double value) {
        using namespace ngraph::opset5;
        auto constant = Constant::create(ngraph::element::f32, ngraph::Shape{1}, {value});

        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{constant}, ngraph::ParameterVector{});
    };

    const auto& f1 = createConstantFunc(1.0);
    const auto& f2 = createConstantFunc(10.0);

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr("mismatch in value: 'value' : look in to the mem buffer"));
}

TEST(TransformationTests, ConstantNegativeDifferentShapes) {
    const auto createConstantFunc = [](const ngraph::Shape& s) {
        using namespace ngraph::opset5;
        auto constant = Constant::create(ngraph::element::f32, s, {1.1});

        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{constant}, ngraph::ParameterVector{});
    };

    const auto& f1 = createConstantFunc(ngraph::Shape{2});
    const auto& f2 = createConstantFunc(ngraph::Shape{2, 2});

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr("mismatch in value: 'shape' : [2] vs [2, 2]"));
}

TEST(TransformationTests, ClampNegativeDifferentMin) {
    const auto createClampFunc = [](double min) {
        using namespace ngraph::opset5;
        auto constant = Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.0});
        auto clamp = std::make_shared<Clamp>(constant, min, 20.);

        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{clamp}, ngraph::ParameterVector{});
    };

    const auto& f1 = createClampFunc(1.0);
    const auto& f2 = createClampFunc(11.0);

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr("mismatch in value: 'min' "));
}

TEST(TransformationTests, ClampNegativeDifferentMax) {
    const auto createClampFunc = [](double max) {
        using namespace ngraph::opset5;
        auto constant = Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.0});
        auto clamp = std::make_shared<Clamp>(constant, 1., max);

        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{clamp}, ngraph::ParameterVector{});
    };

    const auto& f1 = createClampFunc(10.1);
    const auto& f2 = createClampFunc(101.1);

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr("mismatch in value: 'max' "));
}

TEST(TransformationTests, ConcatNegativeDifferentMax) {
    const auto createConcatFunc = [](int64_t axis) {
        using namespace ngraph::opset5;
        auto constant =
            Constant::create(ngraph::element::f32, ngraph::Shape{10, 10, 2, 2, 3}, {1.0});
        auto clamp = std::make_shared<Concat>(ngraph::OutputVector{constant}, axis);

        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{clamp}, ngraph::ParameterVector{});
    };

    const auto& f1 = createConcatFunc(1);
    const auto& f2 = createConcatFunc(2);

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr("mismatch in value: 'axis' : [1] vs [2]"));
}

TEST(TransformationTests, GreaterNegativeDifferentMax) {
    const auto createGreaterFunc = [](ngraph::op::AutoBroadcastType t) {
        using namespace ngraph::opset5;

        auto input1 = std::make_shared<Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto input2 = std::make_shared<Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto node = std::make_shared<Greater>(input1, input2, t);

        return std::make_shared<Function>(OutputVector{node}, ParameterVector{input1, input2});
    };

    const auto& f1 = createGreaterFunc(ngraph::op::AutoBroadcastType::NUMPY);
    const auto& f2 = createGreaterFunc(ngraph::op::AutoBroadcastType::PDPD);

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr(" mismatch in value: 'auto_broadcast' : [numpy] vs [pdpd]"));
}

TEST(TransformationTests, ReadValueNegativeDifferentMax) {
    const auto createReadValueFunc = [](const std::string& variable_id) {
        using namespace ngraph::opset5;

        auto input1 = std::make_shared<Parameter>(ngraph::element::f16, ngraph::Shape{15, 20, 3});
        auto node = std::make_shared<ReadValue>(input1, variable_id);

        return std::make_shared<Function>(OutputVector{node}, ParameterVector{input1});
    };

    const auto& f1 = createReadValueFunc("10");
    const auto& f2 = createReadValueFunc("20");

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr("mismatch in value: 'variable_id' : [10] vs [20]"));
}

TEST(TransformationTests, ReorgYoloNegativeDifferentMax) {
    const auto createReorgYoloFunc = [](const Strides& stride) {
        using namespace ngraph::opset5;

        auto param =
            std::make_shared<Parameter>(ngraph::element::f32, ngraph::Shape{10, 10, 10, 10});
        auto reorg_yolo = std::make_shared<ReorgYolo>(param, stride);

        return std::make_shared<ngraph::Function>(
            std::make_shared<ngraph::opset1::Result>(reorg_yolo), ngraph::ParameterVector{param});
    };

    const auto& f1 = createReorgYoloFunc({1, 2});
    const auto& f2 = createReorgYoloFunc({2, 2});

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr(" mismatch in value: 'stride' : [1, 2] vs [2, 2]"));
}

namespace {

template <typename Member>
class DummyConstant : public ngraph::op::Op {
public:
    DummyConstant() = default;

    DummyConstant(const Member& member)
        : m_element_type(element::Type_t::u8), m_shape({1, 1}), m_member(member) {
        constructor_validate_and_infer_types();
    }

    DummyConstant(const DummyConstant& o)
        : m_element_type(o.m_element_type), m_shape(o.m_shape), m_member(o.m_member) {
        constructor_validate_and_infer_types();
    }

    DummyConstant& operator=(const DummyConstant&) = delete;

    const NodeTypeInfo& get_type_info() const override {
        static const NodeTypeInfo type_info{typeid(this).name(), 0};
        return type_info;
    }

    void validate_and_infer_types() override {
        set_output_type(0, m_element_type, m_shape);  // !!??
    }

    bool visit_attributes(AttributeVisitor& visitor) override {
        visitor.on_attribute("member", m_member);
        return true;
    }

    bool evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const override {
        return true;
    }

    // Don't constant fold a constant; it would make a copy
    bool constant_fold(OutputVector& outputs, const OutputVector& inputs) override {
        return false;
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        check_new_args_count(this, new_args);
        return std::make_shared<DummyConstant>(*this);
    }

protected:
    element::Type m_element_type{element::Type_t::i64};
    Shape m_shape{1, 1};
    Member m_member{};
};

template <typename Member>
std::shared_ptr<ngraph::Function> createDummyFunc(const Member& m) {
    auto constant = std::make_shared<DummyConstant<Member>>(m);

    return std::make_shared<ngraph::Function>(
        ngraph::NodeVector{constant}, ngraph::ParameterVector{});
}

}  // namespace

TEST(TransformationTests, DummyOpNegativeDifferentElementType) {
    const auto& f1 = createDummyFunc(element::Type_t::i64);
    const auto& f2 = createDummyFunc(element::Type_t::f64);

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr(" mismatch in value: 'member' : [i64] vs [f64]"));
}

TEST(TransformationTests, DummyOpNegativeDifferentIntVector) {
    const auto& f1 = createDummyFunc(std::vector<int>{1, 2, 3});
    const auto& f2 = createDummyFunc(std::vector<int>{3, 2, 1});

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr(" mismatch in value: 'member' : [1, 2, 3] vs [3, 2, 1]"));
}

TEST(TransformationTests, DummyOpNegativeDifferentFloatVector) {
    const auto& f1 = createDummyFunc(std::vector<float>{1., 2., 3.});
    const auto& f2 = createDummyFunc(std::vector<float>{3., 2., 1.});

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr(" mismatch in value: 'member' : [1, 2, 3] vs [3, 2, 1]"));
}

TEST(TransformationTests, DummyOpNegativeDifferentStringVector) {
    const auto& f1 = createDummyFunc(std::vector<std::string>{"a", "ba"});
    const auto& f2 = createDummyFunc(std::vector<std::string>{"b", "ab"});

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr(" mismatch in value: 'member' : [a, ba] vs [b, ab]"));
}

namespace ngraph {

struct TestDummyDataTypeTransformationTests_NO_NGRAPH_NAME_COLISION {};

template <>
class AttributeAdapter<TestDummyDataTypeTransformationTests_NO_NGRAPH_NAME_COLISION>
    : public DirectValueAccessor<TestDummyDataTypeTransformationTests_NO_NGRAPH_NAME_COLISION> {
public:
    AttributeAdapter(TestDummyDataTypeTransformationTests_NO_NGRAPH_NAME_COLISION& value)
        : DirectValueAccessor<TestDummyDataTypeTransformationTests_NO_NGRAPH_NAME_COLISION>(value) {
    }

    static constexpr DiscreteTypeInfo type_info{
        "TestDummyDataTypeTransformationTests_NO_NGRAPH_NAME_COLISION", 0};

    const DiscreteTypeInfo& get_type_info() const override {
        return type_info;
    }
};

constexpr DiscreteTypeInfo
    AttributeAdapter<TestDummyDataTypeTransformationTests_NO_NGRAPH_NAME_COLISION>::type_info;

}  // namespace ngraph

TEST(TransformationTests, DummyOpNegativeNotSupportedType) {
    TestDummyDataTypeTransformationTests_NO_NGRAPH_NAME_COLISION m{};
    const auto& f1 = createDummyFunc(m);
    const auto& f2 = createDummyFunc(m);

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr(" [drop `void` comparison which is '"));
}

TEST(TransformationTests, DifferentPrecisionVersusAttributes) {
    const auto createReadValueFunc = [](ngraph::element::Type t) {
        using namespace ngraph::opset5;

        auto input1 = std::make_shared<Parameter>(t, ngraph::Shape{15, 20, 3});
        auto node = std::make_shared<ReadValue>(input1, "1");

        return std::make_shared<Function>(OutputVector{node}, ParameterVector{input1});
    };

    const auto& f1 = createReadValueFunc(ngraph::element::f16);
    const auto& f2 = createReadValueFunc(ngraph::element::i16);

    ///
    /// if FunctionComparator::ATTRIBUTES is select error from Attribute comparator override error
    /// found when FunctionComparator::PRECISION is enabled
    ///

    {  // check precision only
        const auto fc = FunctionsComparator::no_default().enable(FunctionsComparator::PRECISIONS);
        const auto res = fc.compare(f1, f2);
        EXPECT_FALSE(res.valid);
        EXPECT_THAT(res.message, HasSubstr("Different element type detected"));
        EXPECT_THAT(res.message, HasSubstr("f16"));
        EXPECT_THAT(res.message, HasSubstr("i16"));
    }

    {  // check precision and attributes
        const auto fc = FunctionsComparator::no_default()
                            .enable(FunctionsComparator::PRECISIONS)
                            .enable(FunctionsComparator::ATTRIBUTES);
        const auto res = fc.compare(f1, f2);
        EXPECT_FALSE(res.valid);
        EXPECT_THAT(res.message, HasSubstr("Comparison of attributes failed for nodes "));
        EXPECT_THAT(res.message, HasSubstr("f16"));
        EXPECT_THAT(res.message, HasSubstr("i16"));
    }
}
