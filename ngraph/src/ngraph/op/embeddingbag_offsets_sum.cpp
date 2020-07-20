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

#include "ngraph/op/embeddingbag_offsets_sum.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/runtime/reference/embedding_bag_offsets_sum.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v3::EmbeddingBagOffsetsSum::type_info;

op::v3::EmbeddingBagOffsetsSum::EmbeddingBagOffsetsSum(const Output<Node>& emb_table,
                                                       const Output<Node>& indices,
                                                       const Output<Node>& offsets,
                                                       const Output<Node>& default_index,
                                                       const Output<Node>& per_sample_weights)
    : util::EmbeddingBagOffsetsBase(emb_table, indices, offsets, default_index, per_sample_weights)
{
}

op::v3::EmbeddingBagOffsetsSum::EmbeddingBagOffsetsSum(const Output<Node>& emb_table,
                                                       const Output<Node>& indices,
                                                       const Output<Node>& offsets,
                                                       const Output<Node>& default_index)
    : util::EmbeddingBagOffsetsBase(emb_table, indices, offsets, default_index)
{
}

op::v3::EmbeddingBagOffsetsSum::EmbeddingBagOffsetsSum(const Output<Node>& emb_table,
                                                       const Output<Node>& indices,
                                                       const Output<Node>& offsets)
    : util::EmbeddingBagOffsetsBase(emb_table, indices, offsets)
{
}

shared_ptr<Node>
    op::v3::EmbeddingBagOffsetsSum::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    if (new_args.size() == 3)
    {
        return make_shared<op::v3::EmbeddingBagOffsetsSum>(
            new_args.at(0), new_args.at(1), new_args.at(2));
    }
    else if (new_args.size() == 4)
    {
        return make_shared<op::v3::EmbeddingBagOffsetsSum>(
            new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3));
    }
    else if (new_args.size() == 5)
    {
        return make_shared<op::v3::EmbeddingBagOffsetsSum>(
            new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), new_args.at(4));
    }
    else
    {
        throw ngraph_error("Incorrect number of arguments");
    }
}


namespace {
    template<element::Type_t ET>
    inline bool
    evaluate(const HostTensorVector &args, const HostTensorPtr &out) {
        using T = typename element_type_traits<ET>::value_type;
#define REF_CALL(elType) \
        runtime::reference::embeddingBagOffsetsSum<T, typename element_type_traits<elType>::value_type>( \
                                                       args[0]->get_data_ptr<ET>(), \
                                                       args[1]->get_data_ptr<elType>(), \
                                                       args[2]->get_data_ptr<elType>(), \
                                                       args.size() > 3 ? args[3]->get_data_ptr<elType>() : nullptr, \
                                                       args.size() > 4 ? args[4]->get_data_ptr<ET>() : nullptr, \
                                                       out->get_data_ptr<ET>(), \
                                                       shape_size(args[1]->get_shape()), \
                                                       out->get_shape()); \
        break;

        switch (args[1]->get_element_type()) {
            case element::Type_t::i32:
                REF_CALL(element::Type_t::i32);
            case element::Type_t::i64:
                REF_CALL(element::Type_t::i64);
            default:
                return false;
        }
#undef REF_CALL
        return true;
    }


    bool evaluate_ebos(const HostTensorVector &args, const HostTensorPtr &out) {
        bool rc = true;

        switch (out->get_element_type()) {
            TYPE_CASE(u8)(args, out);
                break;
            TYPE_CASE(i32)(args, out);
                break;
            TYPE_CASE(f16)(args, out);
                break;
            TYPE_CASE(f32)(args, out);
                break;
            default:
                rc = false;
                break;
        }
        return rc;
    }
}

bool op::EmbeddingBagOffsetsSum::evaluate(const HostTensorVector &outputs, const HostTensorVector &inputs) {
    return evaluate_ebos(inputs, outputs[0]);
}
