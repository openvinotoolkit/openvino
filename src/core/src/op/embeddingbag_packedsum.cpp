// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/embeddingbag_packedsum.hpp"

#include "itt.hpp"

namespace ov {

op::v3::EmbeddingBagPackedSum::EmbeddingBagPackedSum(const Output<Node>& emb_table,
                                                     const Output<Node>& indices,
                                                     const Output<Node>& per_sample_weights)
    : util::EmbeddingBagPackedBase(emb_table, indices, per_sample_weights) {}

op::v3::EmbeddingBagPackedSum::EmbeddingBagPackedSum(const Output<Node>& emb_table, const Output<Node>& indices)
    : util::EmbeddingBagPackedBase(emb_table, indices) {}

std::shared_ptr<Node> op::v3::EmbeddingBagPackedSum::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v3_EmbeddingBagPackedSum_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 2) {
        return std::make_shared<op::v3::EmbeddingBagPackedSum>(new_args.at(0), new_args.at(1));
    } else if (new_args.size() == 3) {
        return std::make_shared<op::v3::EmbeddingBagPackedSum>(new_args.at(0), new_args.at(1), new_args.at(2));
    } else {
        OPENVINO_THROW("Incorrect number of arguments");
    }
}
bool op::v3::EmbeddingBagPackedSum::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v3_EmbeddingBagPackedSum_visit_attributes);
    return true;
}
}  // namespace ov
