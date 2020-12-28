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

#include "ngraph/op/embeddingbag_packedsum.hpp"
#include "itt.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v3::EmbeddingBagPackedSum::type_info;

op::v3::EmbeddingBagPackedSum::EmbeddingBagPackedSum(const Output<Node>& emb_table,
                                                     const Output<Node>& indices,
                                                     const Output<Node>& per_sample_weights)
    : util::EmbeddingBagPackedBase(emb_table, indices, per_sample_weights)
{
}

op::v3::EmbeddingBagPackedSum::EmbeddingBagPackedSum(const Output<Node>& emb_table,
                                                     const Output<Node>& indices)
    : util::EmbeddingBagPackedBase(emb_table, indices)
{
}

shared_ptr<Node>
    op::v3::EmbeddingBagPackedSum::clone_with_new_inputs(const OutputVector& new_args) const
{
    NGRAPH_OP_SCOPE(v3_EmbeddingBagPackedSum_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 2)
    {
        return make_shared<op::v3::EmbeddingBagPackedSum>(new_args.at(0), new_args.at(1));
    }
    else if (new_args.size() == 3)
    {
        return make_shared<op::v3::EmbeddingBagPackedSum>(
            new_args.at(0), new_args.at(1), new_args.at(2));
    }
    else
    {
        throw ngraph_error("Incorrect number of arguments");
    }
}
