// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/embeddingbag_offsets.hpp"

#include "itt.hpp"

namespace ov {

op::v15::EmbeddingBagOffsets::EmbeddingBagOffsets(const Output<Node>& emb_table,
                                                  const Output<Node>& indices,
                                                  const Output<Node>& offsets,
                                                  const Output<Node>& default_index,
                                                  const Output<Node>& per_sample_weights,
                                                  const Reduction& reduction)
    : util::EmbeddingBagOffsetsBase(emb_table, indices, offsets, default_index, per_sample_weights, reduction) {}

op::v15::EmbeddingBagOffsets::EmbeddingBagOffsets(const Output<Node>& emb_table,
                                                  const Output<Node>& indices,
                                                  const Output<Node>& offsets,
                                                  const Output<Node>& default_index,
                                                  const Reduction& reduction)
    : util::EmbeddingBagOffsetsBase(emb_table, indices, offsets, default_index, reduction) {}

op::v15::EmbeddingBagOffsets::EmbeddingBagOffsets(const Output<Node>& emb_table,
                                                  const Output<Node>& indices,
                                                  const Output<Node>& offsets,
                                                  const Reduction& reduction)
    : util::EmbeddingBagOffsetsBase(emb_table, indices, offsets, reduction) {}

std::shared_ptr<Node> op::v15::EmbeddingBagOffsets::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v15_EmbeddingBagOffsets_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 3) {
        return std::make_shared<op::v15::EmbeddingBagOffsets>(new_args.at(0),
                                                              new_args.at(1),
                                                              new_args.at(2),
                                                              m_reduction);
    } else if (new_args.size() == 4) {
        return std::make_shared<op::v15::EmbeddingBagOffsets>(new_args.at(0),
                                                              new_args.at(1),
                                                              new_args.at(2),
                                                              new_args.at(3),
                                                              m_reduction);
    } else if (new_args.size() == 5) {
        return std::make_shared<op::v15::EmbeddingBagOffsets>(new_args.at(0),
                                                              new_args.at(1),
                                                              new_args.at(2),
                                                              new_args.at(3),
                                                              new_args.at(4),
                                                              m_reduction);
    } else {
        OPENVINO_THROW("Incorrect number of arguments");
    }
}
}  // namespace ov
