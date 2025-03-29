// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

TEST(TransformationTests, CompareFunctoinsTIPositive) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto reshape_pattern = opset5::Constant::create(element::i64, Shape{2}, {1, 16});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = opset5::Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset5::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto res_1 = std::make_shared<opset5::Result>(lstm_cell);
        auto reshape_pattern_2 = opset5::Constant::create(element::i64, Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(lstm_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y, Z});
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto reshape_pattern = opset5::Constant::create(element::i64, Shape{2}, {1, 16});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = opset5::Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset5::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto res_1 = std::make_shared<opset5::Result>(lstm_cell);
        res_1->set_friendly_name("res_1");
        auto reshape_pattern_2 = opset5::Constant::create(element::i64, Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(lstm_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        res_2->set_friendly_name("res_2");
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y, Z});
    }

    auto res = compare_functions(f, f_ref);
    EXPECT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, CompareFunctoinsTINegative) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto reshape_pattern = opset5::Constant::create(element::i64, Shape{2}, {1, 16});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = opset5::Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset5::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto res_1 = std::make_shared<opset5::Result>(lstm_cell);
        auto reshape_pattern_2 = opset5::Constant::create(element::i64, Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(lstm_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y, Z});
    }

    {
        auto X = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 2, 16});
        auto Y = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Z = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        auto Xi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 1, 16});
        auto Yi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});
        auto Zi = std::make_shared<opset5::Parameter>(element::f32, Shape{1, 128});

        // Body
        auto reshape_pattern = opset5::Constant::create(element::i64, Shape{2}, {1, 16});
        auto squeeze = std::make_shared<opset5::Reshape>(Xi, reshape_pattern, false);

        auto w_val = std::vector<float>(512 * 16, 0);
        auto r_val = std::vector<float>(512 * 128, 0);
        auto b_val = std::vector<float>(512, 0);
        auto W = opset5::Constant::create(element::f32, Shape{512, 16}, w_val);
        auto R = opset5::Constant::create(element::f32, Shape{512, 128}, r_val);
        auto B = opset5::Constant::create(element::f32, Shape{512}, b_val);

        auto lstm_cell = std::make_shared<opset5::LSTMCell>(squeeze, Yi, Zi, W, R, B, 128);
        auto relu = std::make_shared<opset5::Relu>(lstm_cell);
        auto res_1 = std::make_shared<opset5::Result>(relu);
        res_1->set_friendly_name("res_1");
        auto reshape_pattern_2 = opset5::Constant::create(element::i64, Shape{3}, {1, 1, 128});
        auto unsqueeze = std::make_shared<opset5::Reshape>(lstm_cell, reshape_pattern_2, false);
        auto res_2 = std::make_shared<opset5::Result>(unsqueeze);
        res_2->set_friendly_name("res_2");
        auto body = std::make_shared<Model>(OutputVector{res_1, res_2}, ParameterVector{Xi, Yi, Zi});

        auto tensor_iterator = std::make_shared<opset5::TensorIterator>();
        tensor_iterator->set_body(body);

        tensor_iterator->set_invariant_input(Zi, Z);
        tensor_iterator->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        tensor_iterator->set_merged_input(Yi, Y, res_1);

        auto out0 = tensor_iterator->get_iter_value(res_1, -1);
        auto out1 = tensor_iterator->get_concatenated_slices(res_2, 0, 1, 1, -1, 1);

        auto res_ti_1 = std::make_shared<opset5::Result>(tensor_iterator->output(1));
        f_ref = std::make_shared<ov::Model>(NodeVector{res_ti_1}, ParameterVector{X, Y, Z});
    }

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    auto res = fc(f, f_ref);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr("LSTMCell/opset4 != Relu/opset1"));
}

TEST(TransformationTests, CompareFunctoinsTINegativeDifferentElementTypeBetweenSubGraphsInputs) {
    const auto createFunc = [](element::Type e_type) {
        using namespace opset6;

        auto X = std::make_shared<Parameter>(e_type, Shape{1, 2});
        auto Y = std::make_shared<Parameter>(e_type, Shape{1, 2});

        auto Xi = std::make_shared<Parameter>(e_type, Shape{1, 2});
        auto Yi = std::make_shared<Parameter>(e_type, Shape{1, 2});

        // Body
        auto add = std::make_shared<Add>(Xi, Yi);
        auto result = std::make_shared<Result>(add);

        auto ti_body = std::make_shared<Model>(OutputVector{result}, ParameterVector{Xi, Yi});

        auto ti = std::make_shared<TensorIterator>();
        ti->set_body(ti_body);
        ti->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        ti->set_sliced_input(Yi, Y, 0, 1, 1, -1, 1);

        auto out = ti->get_concatenated_slices(result, 0, 1, 1, -1, 1);

        return std::make_shared<Model>(NodeVector{out.get_node_shared_ptr()}, ParameterVector{X, Y});
    };
    const auto f1 = createFunc(element::f32);
    const auto f2 = createFunc(element::f16);

    using FnCmp = FunctionsComparator;

    const auto result = FnCmp::with_default().compare(f1, f2);

    EXPECT_FALSE(result.valid);
    EXPECT_THAT(result.message, HasSubstr("different SubGraph InputDescription"));
}

TEST(TransformationTests, CompareFunctoinsTINegativeDifferentElementTypeBetweenInputAndParameter) {
    const auto createFunc = [](element::Type e_type) {
        using namespace opset6;

        auto X = std::make_shared<Parameter>(element::f64, Shape{1, 2});  // <<
        auto Y = std::make_shared<Parameter>(e_type, Shape{1, 2});

        auto Xi = std::make_shared<Parameter>(e_type, Shape{1, 2});
        auto Yi = std::make_shared<Parameter>(e_type, Shape{1, 2});

        // Body
        auto add = std::make_shared<Add>(Xi, Yi);
        auto result = std::make_shared<Result>(add);

        auto ti_body = std::make_shared<Model>(OutputVector{result}, ParameterVector{Xi, Yi});

        auto ti = std::make_shared<TensorIterator>();
        ti->set_body(ti_body);
        ti->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        ti->set_sliced_input(Yi, Y, 0, 1, 1, -1, 1);

        auto out = ti->get_concatenated_slices(result, 0, 1, 1, -1, 1);

        return std::make_shared<Model>(NodeVector{out.get_node_shared_ptr()}, ParameterVector{X, Y});
    };
    const auto f1 = createFunc(element::f32);

    using FnCmp = FunctionsComparator;

    const auto result = FnCmp::with_default().compare(f1, f1);

    EXPECT_FALSE(result.valid);
    EXPECT_THAT(result.message, HasSubstr("inputs and parameters mismatch"));
}

TEST(TransformationTests, CompareFunctoinsTINegativeDifferentElementTypeBetweentResultAndOutput) {
    const auto createFunc = [](element::Type result_element_type, const Shape& result_shape) {
        using namespace opset6;

        auto X = std::make_shared<Parameter>(element::f32, Shape{1, 2});
        auto Y = std::make_shared<Parameter>(element::f32, Shape{1, 2});

        auto Xi = std::make_shared<Parameter>(element::f32, Shape{1, 2});
        auto Yi = std::make_shared<Parameter>(element::f32, Shape{1, 2});

        // Body
        auto add = std::make_shared<Add>(Xi, Yi);
        auto result = std::make_shared<Result>(add);

        auto ti_body = std::make_shared<Model>(OutputVector{result}, ParameterVector{Xi, Yi});

        auto ti = std::make_shared<TensorIterator>();
        ti->set_body(ti_body);
        ti->set_sliced_input(Xi, X, 0, 1, 1, -1, 1);
        ti->set_sliced_input(Yi, Y, 0, 1, 1, -1, 1);

        auto out = ti->get_concatenated_slices(result, 0, 1, 1, -1, 1);

        auto fn = std::make_shared<Model>(NodeVector{out.get_node_shared_ptr()}, ParameterVector{X, Y});

        /// <<
        auto&& result_out = result->output(0);
        Node* result_out_node = result_out.get_node();
        result_out_node->set_output_type(0, result_element_type, result_shape);
        /// <<

        return fn;
    };
    {  // check element type difference
        const auto f1 = createFunc(element::u16, Shape{10, 20});
        const auto f2 = createFunc(element::u64, Shape{10, 20});

        using FnCmp = FunctionsComparator;

        const auto result = FnCmp::with_default().compare(f1, f2);

        EXPECT_FALSE(result.valid);
        EXPECT_THAT(result.message, HasSubstr("outputs and results mismatch"));
    }
    {  // check Shape difference
        const auto f1 = createFunc(element::u16, Shape{11, 20});
        const auto f2 = createFunc(element::u16, Shape{12, 20});

        using FnCmp = FunctionsComparator;

        const auto result = FnCmp::with_default().compare(f1, f2);

        EXPECT_FALSE(result.valid);
        EXPECT_THAT(result.message, HasSubstr("outputs and results mismatch"));
    }
}

TEST(TransformationTests, ConstantNegativeDifferentElementType) {
    const auto createConstantFunc = [](element::Type t) {
        using namespace opset5;
        auto constant = Constant::create(t, Shape{1}, {1.1});

        return std::make_shared<ov::Model>(NodeVector{constant}, ParameterVector{});
    };

    const auto& f1 = createConstantFunc(element::f64);
    const auto& f2 = createConstantFunc(element::f32);

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr("mismatch in value: 'element_type' : [f64] vs [f32]"));
}

TEST(TransformationTests, ConstantNegativeDifferentValues) {
    const auto createConstantFunc = [](double value) {
        using namespace opset5;
        auto constant = Constant::create(element::f32, Shape{1}, {value});

        return std::make_shared<ov::Model>(NodeVector{constant}, ParameterVector{});
    };

    const auto& f1 = createConstantFunc(1.0);
    const auto& f2 = createConstantFunc(10.0);

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr("mismatch in value: 'value' : look in to the mem buffer"));
}

TEST(TransformationTests, ConstantNegativeDifferentShapes) {
    const auto createConstantFunc = [](const Shape& s) {
        using namespace opset5;
        auto constant = Constant::create(element::f32, s, {1.1});

        return std::make_shared<ov::Model>(NodeVector{constant}, ParameterVector{});
    };

    const auto& f1 = createConstantFunc(Shape{2});
    const auto& f2 = createConstantFunc(Shape{2, 2});

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr("mismatch in value: 'shape' : [2] vs [2, 2]"));
}

TEST(TransformationTests, ClampNegativeDifferentMin) {
    const auto createClampFunc = [](double min) {
        using namespace opset5;
        auto constant = Constant::create(element::f32, Shape{1}, {1.0});
        auto clamp = std::make_shared<Clamp>(constant, min, 20.);

        return std::make_shared<ov::Model>(NodeVector{clamp}, ParameterVector{});
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
        using namespace opset5;
        auto constant = Constant::create(element::f32, Shape{1}, {1.0});
        auto clamp = std::make_shared<Clamp>(constant, 1., max);

        return std::make_shared<ov::Model>(NodeVector{clamp}, ParameterVector{});
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
        using namespace opset5;
        auto constant = Constant::create(element::f32, Shape{10, 10, 2, 2, 3}, {1.0});
        auto clamp = std::make_shared<Concat>(OutputVector{constant}, axis);

        return std::make_shared<ov::Model>(NodeVector{clamp}, ParameterVector{});
    };

    const auto& f1 = createConcatFunc(1);
    const auto& f2 = createConcatFunc(2);

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr("mismatch in value: 'axis' : [1] vs [2]"));
}

TEST(TransformationTests, GreaterNegativeDifferentMax) {
    const auto createGreaterFunc = [](op::AutoBroadcastType t) {
        using namespace opset5;

        auto input1 = std::make_shared<Parameter>(element::f16, Shape{15, 20, 3});
        auto input2 = std::make_shared<Parameter>(element::f16, Shape{15, 20, 3});
        auto node = std::make_shared<Greater>(input1, input2, t);

        return std::make_shared<Model>(OutputVector{node}, ParameterVector{input1, input2});
    };

    const auto& f1 = createGreaterFunc(op::AutoBroadcastType::NUMPY);
    const auto& f2 = createGreaterFunc(op::AutoBroadcastType::PDPD);

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr(" mismatch in value: 'auto_broadcast' : [numpy] vs [pdpd]"));
}

TEST(TransformationTests, ReadValueNegativeDifferentMax) {
    const auto createReadValueFunc = [](const std::string& variable_id) {
        using namespace opset5;

        auto input1 = std::make_shared<Parameter>(element::f16, Shape{15, 20, 3});
        auto node = std::make_shared<ReadValue>(input1, variable_id);

        return std::make_shared<Model>(OutputVector{node}, ParameterVector{input1});
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
        using namespace opset5;

        auto param = std::make_shared<Parameter>(element::f32, Shape{10, 10, 10, 10});
        auto reorg_yolo = std::make_shared<ReorgYolo>(param, stride);

        return std::make_shared<ov::Model>(std::make_shared<ov::op::v0::Result>(reorg_yolo), ParameterVector{param});
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
class DummyConstant : public op::Op {
public:
    DummyConstant() = default;

    DummyConstant(const Member& member) : m_element_type(element::Type_t::u8), m_shape({1, 1}), m_member(member) {
        constructor_validate_and_infer_types();
    }

    DummyConstant(const DummyConstant& o) : m_element_type(o.m_element_type), m_shape(o.m_shape), m_member(o.m_member) {
        constructor_validate_and_infer_types();
    }

    DummyConstant& operator=(const DummyConstant&) = delete;

    const NodeTypeInfo& get_type_info() const override {
        static const NodeTypeInfo type_info{typeid(this).name(), "0"};
        return type_info;
    }

    void validate_and_infer_types() override {
        set_output_type(0, m_element_type, m_shape);  // !!??
    }

    bool visit_attributes(AttributeVisitor& visitor) override {
        visitor.on_attribute("member", m_member);
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
std::shared_ptr<ov::Model> createDummyFunc(const Member& m) {
    auto constant = std::make_shared<DummyConstant<Member>>(m);

    return std::make_shared<ov::Model>(NodeVector{constant}, ParameterVector{});
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

namespace ov {

struct TestDummyDataTypeTransformationTests_NO_OV_NAME_COLISION {};

template <>
class AttributeAdapter<TestDummyDataTypeTransformationTests_NO_OV_NAME_COLISION>
    : public DirectValueAccessor<TestDummyDataTypeTransformationTests_NO_OV_NAME_COLISION> {
public:
    AttributeAdapter(TestDummyDataTypeTransformationTests_NO_OV_NAME_COLISION& value)
        : DirectValueAccessor<TestDummyDataTypeTransformationTests_NO_OV_NAME_COLISION>(value) {}

    OPENVINO_RTTI("TestDummyDataTypeTransformationTests_NO_OV_NAME_COLISION");
};
}  // namespace ov

TEST(TransformationTests, DummyOpNegativeNotSupportedType) {
    ov::TestDummyDataTypeTransformationTests_NO_OV_NAME_COLISION m{};
    const auto& f1 = createDummyFunc(m);
    const auto& f2 = createDummyFunc(m);

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr(" [drop `void` comparison which is '"));
}

TEST(TransformationTests, DifferentPrecisionVersusAttributes) {
    const auto createReadValueFunc = [](element::Type t) {
        using namespace opset5;

        auto input1 = std::make_shared<Parameter>(t, Shape{15, 20, 3});
        auto node = std::make_shared<ReadValue>(input1, "1");

        return std::make_shared<Model>(OutputVector{node}, ParameterVector{input1});
    };

    const auto& f1 = createReadValueFunc(element::f16);
    const auto& f2 = createReadValueFunc(element::i16);

    ///
    /// if FunctionComparator::ATTRIBUTES is select error from Attribute comparator override error
    /// found when FunctionComparator::PRECISION is enabled
    ///

    {  // check precision only
        const auto fc = FunctionsComparator::no_default()
                            .enable(FunctionsComparator::NODES)
                            .enable(FunctionsComparator::PRECISIONS);
        const auto res = fc.compare(f1, f2);
        EXPECT_FALSE(res.valid);
        EXPECT_THAT(res.message, HasSubstr("Different element type detected"));
        EXPECT_THAT(res.message, HasSubstr("f16"));
        EXPECT_THAT(res.message, HasSubstr("i16"));
    }

    {  // check precision and attributes
        const auto fc = FunctionsComparator::no_default()
                            .enable(FunctionsComparator::NODES)
                            .enable(FunctionsComparator::PRECISIONS)
                            .enable(FunctionsComparator::ATTRIBUTES);
        const auto res = fc.compare(f1, f2);
        EXPECT_FALSE(res.valid);
        EXPECT_THAT(res.message, HasSubstr("Comparison of attributes failed for nodes "));
        EXPECT_THAT(res.message, HasSubstr("f16"));
        EXPECT_THAT(res.message, HasSubstr("i16"));
    }
}

namespace {
const auto createU1ConstantFunc = [](const Shape& s, const uint8_t* data) {
    using namespace opset5;
    auto c = std::make_shared<Constant>(element::u1, s, data);

    return std::make_shared<ov::Model>(NodeVector{c}, ParameterVector{});
};
}

TEST(TransformationTests, ConstantComparison_ElementTypeU1_Positive_1stbit) {
    const Shape shape{1};
    const uint8_t data[1] = {0x80};  // 1000'0000

    const auto& f1 = createU1ConstantFunc(shape, static_cast<const uint8_t*>(data));
    const auto& f2 = createU1ConstantFunc(shape, static_cast<const uint8_t*>(data));

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f1, f2);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, ConstantComparison_ElementTypeU1_Positive_9thbit) {
    const Shape shape{9};
    const uint8_t data[2] = {0x00, 0x80};  // 0000'0000 1000'0000

    const auto& f1 = createU1ConstantFunc(shape, static_cast<const uint8_t*>(data));
    const auto& f2 = createU1ConstantFunc(shape, static_cast<const uint8_t*>(data));

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f1, f2);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, ConstantComparison_ElementTypeU1_Positive_garbage) {
    // unused mem (after 9th bit) in bit stream should not be compared
    const Shape shape{9};
    const uint8_t data1[2] = {0xAA, 0x8F};  // 1010'1010 1000'1111
    const uint8_t data2[2] = {0xAA, 0xF0};  // 1010'1010 1111'0000

    const auto& f1 = createU1ConstantFunc(shape, static_cast<const uint8_t*>(data1));
    const auto& f2 = createU1ConstantFunc(shape, static_cast<const uint8_t*>(data2));

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f1, f2);
    EXPECT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, ConstantComparison_ElementTypeU1_Negative) {
    const Shape shape{1};
    const uint8_t data1[1] = {0x80};  // 1000 0000
    const uint8_t data2[1] = {0x01};  // 0000 0001

    const auto& f1 = createU1ConstantFunc(shape, static_cast<const uint8_t*>(data1));
    const auto& f2 = createU1ConstantFunc(shape, static_cast<const uint8_t*>(data2));

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr("Different Constant values detected"));
}

TEST(TransformationTests, ConstantComparison_ElementTypeU1_Negative_9thbit) {
    const Shape shape{9};
    const uint8_t data1[2] = {0x00, 0x80};  // 0000 0000 1000 0000
    const uint8_t data2[2] = {0x00, 0x00};  // 0000 0000 0000 0000

    const auto& f1 = createU1ConstantFunc(shape, static_cast<const uint8_t*>(data1));
    const auto& f2 = createU1ConstantFunc(shape, static_cast<const uint8_t*>(data2));

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::CONST_VALUES);
    const auto res = fc.compare(f1, f2);
    EXPECT_FALSE(res.valid);
    EXPECT_THAT(res.message, HasSubstr("Different Constant values detected"));
}
