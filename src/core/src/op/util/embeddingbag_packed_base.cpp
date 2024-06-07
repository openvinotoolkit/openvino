// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/embeddingbag_packed_base.hpp"

#include "embeddingbag_packed_shape_inference.hpp"
#include "itt.hpp"

ov::op::util::EmbeddingBagPackedBase::EmbeddingBagPackedBase(const Output<Node>& emb_table,
                                                             const Output<Node>& indices,
                                                             const Output<Node>& per_sample_weights)
    : Op({emb_table, indices, per_sample_weights}),
      m_reduction{Reduction::SUM} {
    constructor_validate_and_infer_types();
}

ov::op::util::EmbeddingBagPackedBase::EmbeddingBagPackedBase(const Output<Node>& emb_table, const Output<Node>& indices)
    : Op({emb_table, indices}),
      m_reduction{Reduction::SUM} {
    constructor_validate_and_infer_types();
}
ov::op::util::EmbeddingBagPackedBase::EmbeddingBagPackedBase(
    const Output<Node>& emb_table,
    const Output<Node>& indices,
    const Output<Node>& per_sample_weights,
    const ov::op::util::EmbeddingBagPackedBase::Reduction& reduction)
    : Op({emb_table, indices, per_sample_weights}),
      m_reduction{reduction} {
    constructor_validate_and_infer_types();
}

ov::op::util::EmbeddingBagPackedBase::EmbeddingBagPackedBase(
    const Output<Node>& emb_table,
    const Output<Node>& indices,
    const ov::op::util::EmbeddingBagPackedBase::Reduction& reduction)
    : Op({emb_table, indices}),
      m_reduction{reduction} {
    constructor_validate_and_infer_types();
}

void ov::op::util::EmbeddingBagPackedBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_EmbeddingBagPackedBase_validate_and_infer_types);
    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(INDICES) == element::i64 || get_input_element_type(INDICES) == element::i32,
        "INDICES type must be i32 or i64");

    if (get_input_size() == 3) {
        NODE_VALIDATION_CHECK(this,
                              m_reduction == Reduction::SUM,
                              "Per sample weights can only be used in Reduction::SUM mode.");
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(EMB_TABLE).compatible(get_input_element_type(PER_SAMPLE_WEIGHTS)),
                              "Per sample weight element type (",
                              get_input_element_type(PER_SAMPLE_WEIGHTS),
                              ") must match embedding table element type (",
                              get_input_element_type(EMB_TABLE),
                              ")");
    }

    const auto& emb_et = get_input_element_type(EMB_TABLE);
    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    set_output_type(0, emb_et, shape_infer(this, input_shapes)[0]);
}

bool ov::op::util::EmbeddingBagPackedBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_EmbeddingBagPackedBase_visit_attributes);
    visitor.on_attribute("reduction", m_reduction);
    return true;
}

namespace ov {
std::ostream& operator<<(std::ostream& s, const op::util::EmbeddingBagPackedBase::Reduction& reduction) {
    return s << as_string(reduction);
}
template <>
OPENVINO_API EnumNames<op::util::EmbeddingBagPackedBase::Reduction>&
EnumNames<op::util::EmbeddingBagPackedBase::Reduction>::get() {
    static auto enum_names = EnumNames<op::util::EmbeddingBagPackedBase::Reduction>(
        "op::util::EmbeddingBagPackedBase::Reduction",
        {{"sum", op::util::EmbeddingBagPackedBase::Reduction::SUM},
         {"mean", op::util::EmbeddingBagPackedBase::Reduction::MEAN}});
    return enum_names;
}

}  // namespace ov
