// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/embeddingbag_offsets_sum.hpp"

#include "itt.hpp"
#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v3::EmbeddingBagOffsetsSum);

op::v3::EmbeddingBagOffsetsSum::EmbeddingBagOffsetsSum(const Output<Node>& emb_table,
                                                       const Output<Node>& indices,
                                                       const Output<Node>& offsets,
                                                       const Output<Node>& default_index,
                                                       const Output<Node>& per_sample_weights)
    : util::EmbeddingBagOffsetsBase(emb_table, indices, offsets, default_index, per_sample_weights) {}

op::v3::EmbeddingBagOffsetsSum::EmbeddingBagOffsetsSum(const Output<Node>& emb_table,
                                                       const Output<Node>& indices,
                                                       const Output<Node>& offsets,
                                                       const Output<Node>& default_index)
    : util::EmbeddingBagOffsetsBase(emb_table, indices, offsets, default_index) {}

op::v3::EmbeddingBagOffsetsSum::EmbeddingBagOffsetsSum(const Output<Node>& emb_table,
                                                       const Output<Node>& indices,
                                                       const Output<Node>& offsets)
    : util::EmbeddingBagOffsetsBase(emb_table, indices, offsets) {}

shared_ptr<Node> op::v3::EmbeddingBagOffsetsSum::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_EmbeddingBagOffsetsSum_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 3) {
        return make_shared<op::v3::EmbeddingBagOffsetsSum>(new_args.at(0), new_args.at(1), new_args.at(2));
    } else if (new_args.size() == 4) {
        return make_shared<op::v3::EmbeddingBagOffsetsSum>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           new_args.at(3));
    } else if (new_args.size() == 5) {
        return make_shared<op::v3::EmbeddingBagOffsetsSum>(new_args.at(0),
                                                           new_args.at(1),
                                                           new_args.at(2),
                                                           new_args.at(3),
                                                           new_args.at(4));
    } else {
        throw ngraph_error("Incorrect number of arguments");
    }
}
