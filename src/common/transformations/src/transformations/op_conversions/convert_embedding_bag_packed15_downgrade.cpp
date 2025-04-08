// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_embedding_bag_packed15_downgrade.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/embeddingbag_packed.hpp"
#include "openvino/op/embeddingbag_packedsum.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::ConvertEmbeddingBagPacked15ToEmbeddingBagPackedSum3::ConvertEmbeddingBagPacked15ToEmbeddingBagPackedSum3() {
    MATCHER_SCOPE(ConvertEmbeddingBagPacked15ToEmbeddingBagPackedSum3);

    const auto emb_v15_pattern = pattern::wrap_type<ov::op::v15::EmbeddingBagPacked>();

    const matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto emb_v15 = ov::as_type_ptr<ov::op::v15::EmbeddingBagPacked>(m.get_match_root());
        if (!emb_v15 || transformation_callback(emb_v15) ||
            emb_v15->get_reduction() != ov::op::v15::EmbeddingBagPacked::Reduction::SUM) {
            return false;
        }
        std::shared_ptr<ov::op::v3::EmbeddingBagPackedSum> emb_v3;
        if (emb_v15->get_input_size() == 2) {
            emb_v3 = std::make_shared<op::v3::EmbeddingBagPackedSum>(emb_v15->input_value(0), emb_v15->input_value(1));
        } else if (emb_v15->get_input_size() == 3) {
            emb_v3 = std::make_shared<op::v3::EmbeddingBagPackedSum>(emb_v15->input_value(0),
                                                                     emb_v15->input_value(1),
                                                                     emb_v15->input_value(2));
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
