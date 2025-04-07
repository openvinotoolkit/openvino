// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_embedding_bag_offsets15_downgrade.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/embeddingbag_offsets.hpp"
#include "openvino/op/embeddingbag_offsets_sum.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3::
    ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3() {
    MATCHER_SCOPE(ConvertEmbeddingBagOffsets15ToEmbeddingBagOffsetsSum3);

    const auto emb_v15_pattern = pattern::wrap_type<ov::op::v15::EmbeddingBagOffsets>();

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto emb_v15 = ov::as_type_ptr<ov::op::v15::EmbeddingBagOffsets>(m.get_match_root());
        if (!emb_v15 || transformation_callback(emb_v15) ||
            emb_v15->get_reduction() != ov::op::v15::EmbeddingBagOffsets::Reduction::SUM) {
            return false;
        }
        std::shared_ptr<ov::op::v3::EmbeddingBagOffsetsSum> emb_v3;
        if (emb_v15->get_input_size() == 3) {
            emb_v3 = std::make_shared<op::v3::EmbeddingBagOffsetsSum>(emb_v15->input_value(0),
                                                                      emb_v15->input_value(1),
                                                                      emb_v15->input_value(2));
        } else if (emb_v15->get_input_size() == 4) {
            emb_v3 = std::make_shared<op::v3::EmbeddingBagOffsetsSum>(emb_v15->input_value(0),
                                                                      emb_v15->input_value(1),
                                                                      emb_v15->input_value(2),
                                                                      emb_v15->input_value(3));
        } else if (emb_v15->get_input_size() == 5) {
            emb_v3 = std::make_shared<op::v3::EmbeddingBagOffsetsSum>(emb_v15->input_value(0),
                                                                      emb_v15->input_value(1),
                                                                      emb_v15->input_value(2),
                                                                      emb_v15->input_value(3),
                                                                      emb_v15->input_value(4));
        } else {
            return false;
        }

        emb_v3->set_friendly_name(emb_v15->get_friendly_name());
        copy_runtime_info(emb_v15, emb_v3);
        replace_node(emb_v15, emb_v3);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(emb_v15_pattern, matcher_name);
    register_matcher(m, callback);
}
