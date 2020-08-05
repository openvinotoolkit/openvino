//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************
#include <memory>
#include <numeric>

#include "matmul.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/matmul_factory.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/runtime/reference/matmul.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::MatMul, "MatMul", 0);

op::MatMul::MatMul(const Output<Node>& A,
                   const Output<Node>& B,
                   const bool& transpose_a,
                   const bool& transpose_b)
    : FusedOp(OutputVector{A, B})
    , m_transpose_a{transpose_a}
    , m_transpose_b{transpose_b}
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::MatMul::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("transpose_a", m_transpose_a);
    visitor.on_attribute("transpose_b", m_transpose_b);
    return true;
}

void op::MatMul::pre_validate_and_infer_types()
{
    element::Type result_et;
    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, get_input_element_type(0), get_input_element_type(1)),
        "Arguments do not have the same element type (arg0 element type: ",
        get_input_element_type(0),
        ", arg1 element type: ",
        get_input_element_type(1),
        ").");

    const Rank& A_rank = get_input_partial_shape(0).rank();
    const Rank& B_rank = get_input_partial_shape(1).rank();

    if (A_rank.is_static() && B_rank.is_static())
    {
        Rank max_rank = A_rank.get_length() > B_rank.get_length() ? A_rank : B_rank;
        set_output_type(0, result_et, PartialShape::dynamic(max_rank));
    }
}

OutputVector op::MatMul::decompose_op() const
{
    auto A = input_value(0);
    auto B = input_value(1);

    const auto a_rank = A.get_shape().size();
    const auto b_rank = B.get_shape().size();

    if (m_transpose_a && a_rank >= 2)
    {
        vector<size_t> axes_order(a_rank);
        // generate default axes_order.
        iota(axes_order.begin(), axes_order.end(), 0);
        // transpose the last 2 spatial dims
        swap(axes_order[a_rank - 1], axes_order[a_rank - 2]);
        A = builder::reorder_axes(A, axes_order);
    }

    if (m_transpose_b && b_rank >= 2)
    {
        vector<size_t> axes_order(b_rank);
        iota(axes_order.begin(), axes_order.end(), 0);
        swap(axes_order[b_rank - 1], axes_order[b_rank - 2]);
        B = builder::reorder_axes(B, axes_order);
    }

    builder::MatmulFactory factory({A, B});
    return factory.make_matmul_op();
}

shared_ptr<Node> op::MatMul::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<MatMul>(new_args.at(0), new_args.at(1), m_transpose_a, m_transpose_b);
}

namespace
{
    Shape evaluate_matmul_output_shape(const Shape& arg0_shape,
                                       const Shape& arg1_shape,
                                       bool transpose_a,
                                       bool transpose_b)
    {
        Shape output_shape;
        Shape arg0_shape_update = arg0_shape;
        Shape arg1_shape_update = arg1_shape;

        size_t arg0_rank = arg0_shape.size();
        size_t arg1_rank = arg1_shape.size();

        if (transpose_a && arg0_rank > 1)
        {
            swap(arg0_shape_update[arg0_rank - 2], arg0_shape_update[arg0_rank - 1]);
        }
        if (transpose_b && arg1_rank > 1)
        {
            swap(arg1_shape_update[arg1_rank - 2], arg1_shape_update[arg1_rank - 1]);
        }

        if (arg0_rank == 1 && arg1_rank == 1)
        {
            NGRAPH_CHECK(arg0_shape_update == arg1_shape_update, "Incompatible arg shapes");
            output_shape = Shape{};
        }
        else if (arg0_rank == 1)
        {
            // i.e., arg0 shape {3}, arg1 shape{2, 3, 2}, output shape {2, 2}
            NGRAPH_CHECK(arg0_shape_update[0] == arg1_shape_update[arg1_rank - 2],
                         "Incompatible arg shapes");
            arg1_shape_update.erase(arg1_shape_update.begin() + arg1_rank - 2);
            output_shape = arg1_shape_update;
        }
        else if (arg1_rank == 1)
        {
            // i.e., arg0 shape {2, 2, 3}, arg1 shape{3}, output shape {2, 2}
            NGRAPH_CHECK(arg1_shape_update[0] == arg0_shape_update[arg0_rank - 1],
                         "Incompatible arg shapes");
            arg0_shape_update.erase(arg0_shape_update.begin() + arg0_rank - 1);
            output_shape = arg0_shape_update;
        }
        else if (arg0_rank == 2 && arg1_rank == 2)
        {
            NGRAPH_CHECK(arg0_shape_update[1] == arg1_shape_update[0], "Incompatible arg shapes");
            output_shape = Shape{arg0_shape_update[0], arg1_shape_update[1]};
        }
        else
        {
            NGRAPH_CHECK(arg0_shape_update[arg0_rank - 1] == arg1_shape_update[arg1_rank - 2],
                         "Incompatible arg shapes");

            const auto& broadcast_shapes = builder::get_numpy_broadcast_shapes(
                {Shape{begin(arg0_shape_update), next(end(arg0_shape_update), -2)},
                 Shape{begin(arg1_shape_update), next(end(arg1_shape_update), -2)}});

            output_shape = broadcast_shapes.first;
            output_shape.insert(output_shape.end(), arg0_shape_update[arg0_rank - 2]);
            output_shape.insert(output_shape.end(), arg1_shape_update[arg1_rank - 1]);
        }

        return output_shape;
    }

    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& output,
                  bool transpose_a,
                  bool transpose_b)
    {
        using T = typename element_type_traits<ET>::value_type;

        Shape arg0_shape = arg0->get_shape();
        Shape arg1_shape = arg1->get_shape();

        Shape output_shape =
            evaluate_matmul_output_shape(arg0_shape, arg1_shape, transpose_a, transpose_b);
        output->set_element_type(arg0->get_element_type());
        output->set_shape(output_shape);

        runtime::reference::matmul<T>(arg0->get_data_ptr<ET>(),
                                      arg1->get_data_ptr<ET>(),
                                      output->get_data_ptr<ET>(),
                                      arg0_shape,
                                      arg1_shape,
                                      output_shape,
                                      transpose_a,
                                      transpose_b);
        return true;
    }

    bool evaluate_matmul(const HostTensorPtr& arg0,
                         const HostTensorPtr& arg1,
                         const HostTensorPtr& output,
                         bool transpose_a,
                         bool transpose_b)
    {
        bool rc = true;

        switch (arg0->get_element_type())
        {
            TYPE_CASE(i32)(arg0, arg1, output, transpose_a, transpose_b);
            break;
            TYPE_CASE(i64)(arg0, arg1, output, transpose_a, transpose_b);
            break;
            TYPE_CASE(u32)(arg0, arg1, output, transpose_a, transpose_b);
            break;
            TYPE_CASE(u64)(arg0, arg1, output, transpose_a, transpose_b);
            break;
            TYPE_CASE(f16)(arg0, arg1, output, transpose_a, transpose_b);
            break;
            TYPE_CASE(f32)(arg0, arg1, output, transpose_a, transpose_b);
            break;
        default: rc = false; break;
        }
        return rc;
    }
}

bool op::MatMul::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    return evaluate_matmul(inputs[0], inputs[1], outputs[0], get_transpose_a(), get_transpose_b());
}
