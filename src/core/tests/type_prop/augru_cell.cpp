// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/augru_cell.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/type_prop.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/opsets/opset9.hpp"

using namespace std;
using namespace ov;

TEST(type_prop, augru_cell) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    const auto X = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto H_t = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto B = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size});
    const auto A = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, 1});

    const auto augru_cell = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_EQ(augru_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(augru_cell->get_output_shape(0), (Shape{batch_size, hidden_size}));
}

TEST(type_prop, augru_cell_invalid_input) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;
    auto B = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size});
    auto A = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, 1});

    const auto X = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, input_size});
    auto R = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, hidden_size});

    // Invalid W tensor shape.
    auto W = make_shared<opset9::Parameter>(element::f32, Shape{hidden_size, input_size});
    try {
        const auto gru_cell = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
        FAIL() << "AUGRUCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("First dimension of W input shape is required to be compatible"));
    }

    // Invalid R tensor shape.
    W = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    R = make_shared<opset9::Parameter>(element::f32, Shape{hidden_size, 1});
    try {
        const auto gru_cell = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
        FAIL() << "AUGRUCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Dimension `hidden_size` is not matched between inputs"));
    }

    // Invalid H_t tensor shape.
    R = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    H_t = make_shared<opset9::Parameter>(element::f32, Shape{4, hidden_size});
    try {
        const auto gru_cell = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
        FAIL() << "AUGRUCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("Dimension `batch_size` is not matched between inputs"));
    }

    // Invalid B tensor shape.
    H_t = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, hidden_size});
    B = make_shared<opset9::Parameter>(element::f32, Shape{hidden_size});
    try {
        const auto gru_cell = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
        FAIL() << "GRUCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("First dimension of B input shape is required to be compatible"));
    }

    // Invalid A tensor shape.
    A = make_shared<opset9::Parameter>(element::f32, Shape{hidden_size});
    B = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size});
    try {
        const auto gru_cell = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
        FAIL() << "GRUCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("'A' input must be a 2D tensor."));
    }
}

TEST(type_prop, augru_cell_dynamic_batch_size) {
    const auto batch_size = Dimension::dynamic();
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    const auto X = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, input_size});
    const auto W = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    const auto R = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    const auto B = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size});
    const auto A = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, 1});

    const auto augru_cell = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_EQ(augru_cell->get_output_element_type(0), element::f32);
    EXPECT_EQ(augru_cell->get_output_partial_shape(0), (PartialShape{batch_size, hidden_size}));
}

TEST(type_prop, augru_cell_dynamic_batch_and_input_size) {
    const auto batch_size = Dimension::dynamic();
    const auto input_size = Dimension::dynamic();
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    const auto X = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, input_size});
    const auto W = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    const auto R = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    const auto H_t = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    const auto B = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size});
    const auto A = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, 1});

    const auto augru_cell = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_EQ(augru_cell->get_output_partial_shape(0), (PartialShape{batch_size, hidden_size}));
    EXPECT_EQ(augru_cell->get_output_element_type(0), element::f32);
}

TEST(type_prop, augru_cell_invalid_input_rank) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    auto X = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, input_size});
    auto R = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    auto B = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size});
    auto A = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, 1});

    // Invalid rank for W tensor.
    auto W = make_shared<opset9::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size),
                 ov::NodeValidationFailure)
        << "AUGRUCell node was created with invalid data.";

    // Invalid rank for X tensor.
    W = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    X = make_shared<opset9::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size),
                 ov::NodeValidationFailure)
        << "AUGRUCell node was created with invalid data.";

    // Invalid rank for H_t tensor.
    X = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, input_size});
    H_t = make_shared<opset9::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size),
                 ov::NodeValidationFailure)
        << "AUGRUCell node was created with invalid data.";

    // Invalid rank for R tensor.
    H_t = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    R = make_shared<opset9::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size),
                 ov::NodeValidationFailure)
        << "AUGRUCell node was created with invalid data.";

    // Invalid rank for B tensor.
    R = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    B = make_shared<opset9::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size),
                 ov::NodeValidationFailure)
        << "AUGRUCell node was created with invalid data.";

    // Invalid rank for A tensor.
    B = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size});
    A = make_shared<opset9::Parameter>(element::f32, PartialShape{});
    ASSERT_THROW(const auto unused = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size),
                 ov::NodeValidationFailure)
        << "AUGRUCell node was created with invalid data.";
}

TEST(type_prop, augru_cell_input_dynamic_rank) {
    int64_t batch_size = 2;
    int64_t input_size = 3;
    int64_t hidden_size = 3;
    int64_t gates_count = 3;

    auto X = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, input_size});
    auto R = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    auto H_t = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    auto B = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size});
    auto A = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, 1});

    auto check_dynamic_gru = [&](const shared_ptr<op::internal::AUGRUCell>& augru) -> bool {
        return augru->output(0).get_partial_shape() == PartialShape{batch_size, hidden_size} &&
               augru->output(0).get_element_type() == augru->input(0).get_element_type();
    };

    // Dynamic rank for W tensor.
    auto W = make_shared<opset9::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto augru_w = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_TRUE(check_dynamic_gru(augru_w));

    // Dynamic rank for X tensor.
    W = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, input_size});
    X = make_shared<opset9::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto augru_x = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_TRUE(check_dynamic_gru(augru_x));

    // Dynamic rank for H_t tensor.
    X = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, input_size});
    H_t = make_shared<opset9::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto augru_h = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_TRUE(check_dynamic_gru(augru_h));

    // Dynamic rank for R tensor.
    H_t = make_shared<opset9::Parameter>(element::f32, PartialShape{batch_size, hidden_size});
    R = make_shared<opset9::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto augru_r = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_TRUE(check_dynamic_gru(augru_r));

    // Dynamic rank for B tensor.
    R = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size, hidden_size});
    B = make_shared<opset9::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto augru_b = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_TRUE(check_dynamic_gru(augru_b));

    // Dynamic rank for A tensor.
    B = make_shared<opset9::Parameter>(element::f32, PartialShape{gates_count * hidden_size});
    A = make_shared<opset9::Parameter>(element::f32, PartialShape::dynamic(Rank::dynamic()));
    auto augru_a = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);
    EXPECT_TRUE(check_dynamic_gru(augru_a));
}

namespace {
struct NotSupportedArguments {
    float clip = 0;
    vector<string> activations = {"sigmoid", "tanh"};
    vector<float> activations_alpha, activations_beta;
    bool linear_before_reset = false;
};

class AttributeVisitorMock : public AttributeVisitor {
public:
    AttributeVisitorMock(const NotSupportedArguments& args) : m_args{args} {}
    void on_adapter(const std::string& name, ValueAccessor<void>& adapter) override {}

    void on_adapter(const std::string& name, ValueAccessor<double>& adapter) override {
        if ("clip" == name) {
            adapter.set(m_args.clip);
        }
    }

    void on_adapter(const std::string& name, ValueAccessor<vector<string>>& adapter) override {
        if ("activations" == name) {
            adapter.set(m_args.activations);
        }
    }

    void on_adapter(const std::string& name, ValueAccessor<vector<float>>& adapter) override {
        if ("activations_alpha" == name) {
            adapter.set(m_args.activations_alpha);
        }
        if ("activations_beta" == name) {
            adapter.set(m_args.activations_beta);
        }
    }

    void on_adapter(const std::string& name, ValueAccessor<bool>& adapter) override {
        if ("linear_before_reset" == name) {
            adapter.set(m_args.linear_before_reset);
        }
    }

private:
    NotSupportedArguments m_args;
};
}  // namespace

TEST(type_prop, augru_not_supported_attributes) {
    const size_t batch_size = 2;
    const size_t input_size = 3;
    const size_t hidden_size = 3;
    const size_t gates_count = 3;

    const auto X = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, input_size});
    const auto H_t = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, hidden_size});
    const auto W = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size, input_size});
    const auto R = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size, hidden_size});
    const auto B = make_shared<opset9::Parameter>(element::f32, Shape{gates_count * hidden_size});
    const auto A = make_shared<opset9::Parameter>(element::f32, Shape{batch_size, 1});

    const auto augru_cell = make_shared<op::internal::AUGRUCell>(X, H_t, W, R, B, A, hidden_size);

    NotSupportedArguments args;
    args.clip = 2.f;
    AttributeVisitorMock visitor(args);
    augru_cell->visit_attributes(visitor);

    try {
        augru_cell->validate_and_infer_types();
        FAIL() << "AUGRUCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("AUGRUCell doesn't support clip other than 0."));
    }

    args = NotSupportedArguments();
    args.activations = {"relu", "tanh"};
    visitor = AttributeVisitorMock(args);
    augru_cell->visit_attributes(visitor);

    try {
        augru_cell->validate_and_infer_types();
        FAIL() << "AUGRUCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("AUGRUCell supports only sigmoid for f and tanh for g activation functions."));
    }

    args = NotSupportedArguments();
    args.activations_beta = {1, 2};
    visitor = AttributeVisitorMock(args);
    augru_cell->visit_attributes(visitor);

    try {
        augru_cell->validate_and_infer_types();
        FAIL() << "AUGRUCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(),
                             std::string("AUGRUCell doesn't support activations_alpha and activations_beta."));
    }

    args = NotSupportedArguments();
    args.linear_before_reset = true;
    visitor = AttributeVisitorMock(args);
    augru_cell->visit_attributes(visitor);

    try {
        augru_cell->validate_and_infer_types();
        FAIL() << "AUGRUCell node was created with invalid data.";
    } catch (const NodeValidationFailure& error) {
        EXPECT_HAS_SUBSTRING(error.what(), std::string("AUGRUCell supports only linear_before_reset equals false."));
    }
}
