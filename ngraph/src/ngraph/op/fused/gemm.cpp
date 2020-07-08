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
#include "ngraph/op/fused/gemm.hpp"

#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/builder/reshape.hpp"
#include "ngraph/op/add.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/fused/matmul.hpp"
#include "ngraph/op/multiply.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Gemm::type_info;

op::Gemm::Gemm(const Output<Node>& A,
               const Output<Node>& B,
               const Output<Node>& C,
               double alpha,
               double beta,
               bool transA,
               bool transB)
    : FusedOp({A, B, C})
    , m_alpha{alpha}
    , m_beta{beta}
    , m_transA{transA}
    , m_transB{transB}
{
    constructor_validate_and_infer_types();
}

OutputVector op::Gemm::decompose_op() const
{
    auto A = input_value(0);
    auto B = input_value(1);
    auto C = input_value(2);

    if (m_transA)
    {
        A = builder::transpose(A);
    }
    if (m_transB)
    {
        B = builder::transpose(B);
    }

    A = builder::flatten(A, 1);
    B = builder::flatten(B, 1);

    // A' * B'
    std::shared_ptr<Node> a_dot_b = std::make_shared<op::Dot>(A, B);

    // alpha
    std::shared_ptr<Node> alpha_node = std::make_shared<op::Constant>(
        a_dot_b->get_element_type(), a_dot_b->get_shape(), std::vector<double>{m_alpha});
    // alpha * A' * B'
    a_dot_b = std::make_shared<op::Multiply>(alpha_node, a_dot_b);

    // beta * C
    std::shared_ptr<Node> beta_node = std::make_shared<op::Constant>(
        C.get_element_type(), C.get_shape(), std::vector<double>{m_beta});
    C = std::make_shared<op::Multiply>(beta_node, C);

    // alpha * A' * B' + beta * C
    // The input tensor `C` should be "unidirectionally broadcastable" to the `a_dot_b` tensor.
    auto broadcasted_c = builder::numpy_broadcast(C, a_dot_b->get_shape());
    return {std::make_shared<op::Add>(a_dot_b, broadcasted_c)};
}

shared_ptr<Node> op::Gemm::clone_with_new_inputs(const OutputVector& new_args) const
{
    if (new_args.size() != 3)
    {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<Gemm>(
        new_args.at(0), new_args.at(1), new_args.at(2), m_alpha, m_beta, m_transA, m_transB);
}
