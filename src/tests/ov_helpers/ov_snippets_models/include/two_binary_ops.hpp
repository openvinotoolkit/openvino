// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "snippets_helpers.hpp"

namespace ov {
namespace test {
namespace snippets {

/*
 * @class BaseDummyOperation
 * @brief Base not type relaxed operation with absent validation and parameterized output.
 */
class BaseDummyOperation : public ov::op::Op {
public:
    BaseDummyOperation(const Output<Node>& arg0,
                       const Output<Node>& arg1,
                       const element::Type& output_type = element::dynamic)
        : Op({arg0, arg1}),
          output_type(output_type) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_type(0,
                        output_type == element::dynamic ? get_input_element_type(0) : output_type,
                        get_input_partial_shape(0));
    }

    element::Type get_output_type() const { return output_type; }

protected:
    element::Type output_type;
};

/*
 * @class DummyOperation1
 * @brief Not type relaxed operation with absent validation and parameterized output.
 */
class DummyOperation1 : public BaseDummyOperation {
public:
    OPENVINO_OP("DummyOperation1", "test::snippets");

    DummyOperation1(const Output<Node>& arg0,
                    const Output<Node>& arg1,
                    const element::Type& output_type = element::dynamic)
        : BaseDummyOperation(arg0, arg1, output_type) {
        constructor_validate_and_infer_types();
    }

    DummyOperation1(const BaseDummyOperation& op) : DummyOperation1(op.get_input_source_output(0), op.get_input_source_output(1), op.get_output_type()) {
        constructor_validate_and_infer_types();
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        return std::make_shared<DummyOperation1>(new_args.at(0), new_args.at(1), this->get_output_type());
    }
};

/*
 * @class DummyOperation2
 * @brief Not type relaxed operation with absent validation and parameterized output.
 */
class DummyOperation2 : public BaseDummyOperation {
public:
    OPENVINO_OP("DummyOperation2", "test::snippets");

    DummyOperation2(const Output<Node>& arg0,
                    const Output<Node>& arg1,
                    const element::Type& output_type = element::dynamic)
        : BaseDummyOperation(arg0, arg1, output_type) {
        constructor_validate_and_infer_types();
    }

    DummyOperation2(const BaseDummyOperation& op) : DummyOperation2(op.get_input_source_output(0), op.get_input_source_output(1), op.get_output_type()) {
        constructor_validate_and_infer_types();
    }

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override {
        return std::make_shared<DummyOperation2>(new_args.at(0), new_args.at(1), this->get_output_type());
    }
};

/*
 * @class TwoBinaryOpsFunction
 * @brief The function for unit tests only, includes dummy DummyOperation1 & DummyOperation2 operations.
 *
 * Input arguments are used to create function in getOriginal or getReference methods only.
 * Dont use getLowered method, it is not implemented and throw std::runtime_error exception.
 *
 *    Parameter    Parameter
 *        \         /
 *         \       /
 *       DummyOperation1   Constant
 *              \          /
 *               \        /
 *              DummyOperation2
 *                    |
 *                    |
 *                 Result
 *
 */
class TwoBinaryOpsFunction : public SnippetsFunctionBase {
public:
    class Actual {
    public:
        std::pair<element::Type, element::Type> convertion_before_op1;
        element::Type convertion_before_op2_1;
        std::pair<element::Type, element::Type> convertion_before_op2_2;
        element::Type convertion_after_op2;
    };

    class Expected {
    public:
        std::pair<element::Type, element::Type> convertion_before_op1;
        element::Type convertion_before_op2_1;
        std::pair<element::Type, element::Type> convertion_before_op2_2;
        element::Type convertion_after_op2;
        element::Type convertion_before_result;
    };

    explicit TwoBinaryOpsFunction(
        const std::vector<PartialShape> input_shapes,
        const ov::element::Type precision1,
        const ov::element::Type precision2,
        const ov::element::Type constant_precision,
        Actual actual,
        Expected expected) :
        SnippetsFunctionBase(input_shapes),
        precision1(precision1),
        precision2(precision2),
        constant_precision(constant_precision),
        actual(actual),
        expected(expected) {
        OPENVINO_ASSERT(input_shapes.size() == 2ull, "input_shapes size has to be equal to 2");
    }

protected:
    std::shared_ptr<ov::Model> initOriginal() const override;
    std::shared_ptr<ov::Model> initReference() const override;

    const ov::element::Type precision1;
    const ov::element::Type precision2;
    const ov::element::Type constant_precision;
    const Actual actual;
    const Expected expected;

private:
    /*
     * Returns model implicitly via getOriginal call in initOriginal or getReference in initReference.
     */
    static std::shared_ptr<ov::Model> get(
        const ov::element::Type& precision1,
        const ov::PartialShape& inputShape1,
        const ov::element::Type& precision2,
        const ov::PartialShape& inputShape2,
        const ov::element::Type& constant_precision,
        const std::pair<element::Type, element::Type>& convertion_before_op1,
        const element::Type& convertion_before_op2_1,
        const std::pair<element::Type, element::Type>& convertion_before_op2_2,
        const element::Type& convertion_after_op2,
        const element::Type& convertion_before_result = {});
};

}  // namespace snippets
}  // namespace test
}  // namespace ov
