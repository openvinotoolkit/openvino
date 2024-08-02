// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/embeddingbag_packed.hpp"

#include "itt.hpp"

namespace ov {

op::v15::EmbeddingBagPacked::EmbeddingBagPacked(const Output<Node>& emb_table,
                                                const Output<Node>& indices,
                                                const Output<Node>& per_sample_weights,
                                                const Reduction& reduction)
    : util::EmbeddingBagPackedBase(emb_table, indices, per_sample_weights, reduction) {}

op::v15::EmbeddingBagPacked::EmbeddingBagPacked(const Output<Node>& emb_table,
                                                const Output<Node>& indices,
                                                const Reduction& reduction)
    : util::EmbeddingBagPackedBase(emb_table, indices, reduction) {}

std::shared_ptr<Node> op::v15::EmbeddingBagPacked::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v15_EmbeddingBagPacked_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 2) {
        return std::make_shared<op::v15::EmbeddingBagPacked>(new_args.at(0), new_args.at(1), m_reduction);
    } else if (new_args.size() == 3) {
        return std::make_shared<op::v15::EmbeddingBagPacked>(new_args.at(0),
                                                             new_args.at(1),
                                                             new_args.at(2),
                                                             m_reduction);
    } else {
        OPENVINO_THROW("Incorrect number of arguments");
    }
}
}  // namespace ov
