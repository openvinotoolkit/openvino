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

#include "itt.hpp"
#include "matmul.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/runtime/reference/matmul.hpp"

using namespace std;
using namespace ngraph;

NGRAPH_RTTI_DEFINITION(op::MatMul, "MatMul", 0);

op::MatMul::MatMul(const Output<Node>& A,
                   const Output<Node>& B,
                   const bool& transpose_a,
                   const bool& transpose_b)
    : Op(OutputVector{A, B})
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

shared_ptr<Node> op::MatMul::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<MatMul>(new_args.at(0), new_args.at(1), m_transpose_a, m_transpose_b);
}

namespace
{
    PartialShape evaluate_matmul_output_shape(const PartialShape& arg0_shape,
                                              const PartialShape& arg1_shape,
                                              bool transpose_a,
                                              bool transpose_b)
    {
        auto arg0_rank = arg0_shape.rank().get_length();
        auto arg1_rank = arg1_shape.rank().get_length();
        auto max_rank = std::max(arg0_rank, arg1_rank);
        auto merged_dimension = Dimension::dynamic();

        std::vector<Dimension> arg0_shape_update(arg0_rank);
        std::vector<Dimension> arg1_shape_update(arg1_rank);
        std::vector<Dimension> output_shape(max_rank);

        for (size_t i = 0; i < arg0_rank; i++)
        {
            arg0_shape_update[i] = arg0_shape[i];
        }

        for (size_t i = 0; i < arg1_rank; i++)
        {
            arg1_shape_update[i] = arg1_shape[i];
        }

        if (transpose_a)
        {
            // 1D vector of shape {S} is reshaped to {S, 1}
            if (arg0_rank == 1)
            {
                arg0_shape_update.insert(arg0_shape_update.begin(), 1);
                arg0_rank = arg0_shape_update.size();
            }

            swap(arg0_shape_update[arg0_rank - 2], arg0_shape_update[arg0_rank - 1]);
        }
        if (transpose_b)
        {
            // 1D vector of shape {S} is reshaped to {1, S}
            if (arg1_rank == 1)
            {
                arg1_shape_update.insert(arg1_shape_update.begin(), 1);
                arg1_rank = arg1_shape_update.size();
            }

            swap(arg1_shape_update[arg1_rank - 2], arg1_shape_update[arg1_rank - 1]);
        }

        if (arg0_rank == 1 && arg1_rank == 1)
        {
            NGRAPH_CHECK(
                Dimension::merge(merged_dimension, arg0_shape_update[0], arg1_shape_update[0]),
                "Incompatible matrix dimensions");
            output_shape.erase(output_shape.begin(), output_shape.end());
        }
        else if (arg0_rank == 1)
        {
            // i.e., arg0 shape {3}, arg1 shape{2, 3, 2}, output shape {2, 2}
            NGRAPH_CHECK(Dimension::merge(merged_dimension,
                                          arg0_shape_update[0],
                                          arg1_shape_update[arg1_rank - 2]),
                         "Incompatible matrix dimensions");
            arg1_shape_update.erase(arg1_shape_update.begin() + arg1_rank - 2);
            output_shape = arg1_shape_update;
        }
        else if (arg1_rank == 1)
        {
            // i.e., arg0 shape {2, 2, 3}, arg1 shape{3}, output shape {2, 2}
            NGRAPH_CHECK(Dimension::merge(merged_dimension,
                                          arg1_shape_update[0],
                                          arg0_shape_update[arg0_rank - 1]),
                         "Incompatible matrix dimensions.");
            arg0_shape_update.erase(arg0_shape_update.begin() + arg0_rank - 1);
            output_shape = arg0_shape_update;
        }
        else
        {
            NGRAPH_CHECK(Dimension::merge(merged_dimension,
                                          arg0_shape_update[arg0_rank - 1],
                                          arg1_shape_update[arg1_rank - 2]),
                         "Incompatible matrix dimensions.");

            // handle batch size
            if (max_rank > 2)
            {
                std::vector<Dimension> low_size_matrix =
                    arg0_rank > arg1_rank ? arg1_shape_update : arg0_shape_update;
                std::vector<Dimension> big_size_matrix =
                    arg0_rank > arg1_rank ? arg0_shape_update : arg1_shape_update;

                if (arg0_rank != arg1_rank)
                {
                    size_t delta_rank = big_size_matrix.size() - low_size_matrix.size();

                    // resize low_size_matrix (with 1) to have the same dimension as
                    // big_size_matrix
                    low_size_matrix.resize(low_size_matrix.size() + delta_rank, 1);
                    // move added 1 values to the beginning
                    std::rotate(low_size_matrix.begin(),
                                low_size_matrix.end() - delta_rank,
                                low_size_matrix.end());
                }

                // get max value for all batches, COL_INDEX_DIM and ROW_INDEX_DIM are updated at the
                // end
                for (auto i = 0; i < max_rank - 2; i++)
                {
                    if ((low_size_matrix[i].is_dynamic() || big_size_matrix[i].is_dynamic()))
                    {
                        // non-dynamic value is asssigned to output when it is > 1, otherwise
                        // dynamic dimension is forwarded to output
                        auto output_value = low_size_matrix[i].is_dynamic()
                                                ? big_size_matrix[i].get_length()
                                                : low_size_matrix[i].get_length();

                        if (output_value != 1)
                        {
                            output_shape[i] = output_value;
                        }
                        else
                        {
                            output_shape[i] = Dimension::dynamic();
                        }
                    }
                    else
                    {
                        output_shape[i] = std::max(low_size_matrix[i].get_length(),
                                                   big_size_matrix[i].get_length());
                    }
                }
            }

            // in output_shape replace 2 last axes with ROW_INDEX_DIM from arg0 matrix
            // and COL_INDEX_DIM from arg1 matrix
            output_shape.at(output_shape.size() - 2) = arg0_shape_update.at(arg0_rank - 2);
            output_shape.at(output_shape.size() - 1) = arg1_shape_update.at(arg1_rank - 1);
        }

        return PartialShape{output_shape};
    }

    template <element::Type_t ET>
    bool evaluate(const HostTensorPtr& arg0,
                  const HostTensorPtr& arg1,
                  const HostTensorPtr& output,
                  bool transpose_a,
                  bool transpose_b)
    {
        using T = typename element_type_traits<ET>::value_type;

        Shape arg0_shape = Shape{arg0->get_shape()};
        Shape arg1_shape = Shape{arg1->get_shape()};

        PartialShape output_partial_shape = evaluate_matmul_output_shape(
            PartialShape{arg0_shape}, PartialShape{arg1_shape}, transpose_a, transpose_b);
        Shape output_shape = output_partial_shape.to_shape();
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
} // namespace

bool op::MatMul::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const
{
    OV_ITT_SCOPED_TASK(itt::domains::nGraphOp, "op::MatMul::evaluate");
    return evaluate_matmul(inputs[0], inputs[1], outputs[0], get_transpose_a(), get_transpose_b());
}

void ngraph::op::v0::MatMul::validate_and_infer_types()
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

    const auto& A_matrix = get_input_partial_shape(0);
    const auto& B_matrix = get_input_partial_shape(1);

    if (A_matrix.rank().is_static() && B_matrix.rank().is_static())
    {
        PartialShape output_shape;

        const bool transpose_a = get_transpose_a();
        const bool transpose_b = get_transpose_b();

        output_shape = evaluate_matmul_output_shape(A_matrix, B_matrix, transpose_a, transpose_b);

        set_output_type(0, result_et, output_shape);
    }
    else
    {
        set_output_type(0, result_et, PartialShape::dynamic());
    }
}
