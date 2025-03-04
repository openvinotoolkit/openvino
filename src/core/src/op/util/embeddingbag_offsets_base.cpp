// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/embeddingbag_offsets_base.hpp"

#include "embeddingbag_offsets_shape_inference.hpp"
#include "itt.hpp"

ov::op::util::EmbeddingBagOffsetsBase::EmbeddingBagOffsetsBase(const Output<Node>& emb_table,
                                                               const Output<Node>& indices,
                                                               const Output<Node>& offsets,
                                                               const Output<Node>& default_index,
                                                               const Output<Node>& per_sample_weights)
    : Op({emb_table, indices, offsets, default_index, per_sample_weights}),
      m_reduction{Reduction::SUM} {
    constructor_validate_and_infer_types();
}

ov::op::util::EmbeddingBagOffsetsBase::EmbeddingBagOffsetsBase(const Output<Node>& emb_table,
                                                               const Output<Node>& indices,
                                                               const Output<Node>& offsets,
                                                               const Output<Node>& default_index)
    : Op({emb_table, indices, offsets, default_index}),
      m_reduction{Reduction::SUM} {
    constructor_validate_and_infer_types();
}

ov::op::util::EmbeddingBagOffsetsBase::EmbeddingBagOffsetsBase(const Output<Node>& emb_table,
                                                               const Output<Node>& indices,
                                                               const Output<Node>& offsets)
    : Op({emb_table, indices, offsets}),
      m_reduction{Reduction::SUM} {
    constructor_validate_and_infer_types();
}

ov::op::util::EmbeddingBagOffsetsBase::EmbeddingBagOffsetsBase(const Output<Node>& emb_table,
                                                               const Output<Node>& indices,
                                                               const Output<Node>& offsets,
                                                               const Output<Node>& default_index,
                                                               const Output<Node>& per_sample_weights,
                                                               const Reduction& reduction)
    : Op({emb_table, indices, offsets, default_index, per_sample_weights}),
      m_reduction{reduction} {
    constructor_validate_and_infer_types();
}

ov::op::util::EmbeddingBagOffsetsBase::EmbeddingBagOffsetsBase(const Output<Node>& emb_table,
                                                               const Output<Node>& indices,
                                                               const Output<Node>& offsets,
                                                               const Output<Node>& default_index,
                                                               const Reduction& reduction)
    : Op({emb_table, indices, offsets, default_index}),
      m_reduction{reduction} {
    constructor_validate_and_infer_types();
}

ov::op::util::EmbeddingBagOffsetsBase::EmbeddingBagOffsetsBase(const Output<Node>& emb_table,
                                                               const Output<Node>& indices,
                                                               const Output<Node>& offsets,
                                                               const Reduction& reduction)
    : Op({emb_table, indices, offsets}),
      m_reduction{reduction} {
    constructor_validate_and_infer_types();
}

void ov::op::util::EmbeddingBagOffsetsBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_EmbeddingBagOffsetsBase_validate_and_infer_types);
    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(OFFSETS) == element::i64 || get_input_element_type(OFFSETS) == element::i32,
        "OFFSETS type must be i32 or i64");

    NODE_VALIDATION_CHECK(
        this,
        get_input_element_type(INDICES) == element::i64 || get_input_element_type(INDICES) == element::i32,
        "INDICES type must be i32 or i64");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(INDICES).compatible(get_input_element_type(OFFSETS)),
                          "Offsets element type (",
                          get_input_element_type(OFFSETS),
                          ") must match indices element type (",
                          get_input_element_type(INDICES),
                          ")");

    if (get_input_size() >= 4) {
        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(DEFAULT_INDEX) == element::i64 ||
                                  get_input_element_type(DEFAULT_INDEX) == element::i32,
                              "DEFAULT_INDEX type must be i32 or i64");

        NODE_VALIDATION_CHECK(this,
                              get_input_element_type(INDICES).compatible(get_input_element_type(DEFAULT_INDEX)),
                              "Default_index element type (",
                              get_input_element_type(DEFAULT_INDEX),
                              ") must match indices element type (",
                              get_input_element_type(INDICES),
                              ")");
    }

    if (get_input_size() == 5) {
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

    const auto& result_et = get_input_element_type(EMB_TABLE);
    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    set_output_type(0, result_et, shape_infer(this, input_shapes)[0]);
}

bool ov::op::util::EmbeddingBagOffsetsBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_EmbeddingBagOffsetsBase_visit_attributes);
    visitor.on_attribute("reduction", m_reduction);
    return true;
}
namespace ov {
template <>
OPENVINO_API EnumNames<op::util::EmbeddingBagOffsetsBase::Reduction>&
EnumNames<op::util::EmbeddingBagOffsetsBase::Reduction>::get() {
    static auto enum_names = EnumNames<op::util::EmbeddingBagOffsetsBase::Reduction>(
        "op::util::EmbeddingBagOffsetsBase::Reduction",
        {{"sum", op::util::EmbeddingBagOffsetsBase::Reduction::SUM},
         {"mean", op::util::EmbeddingBagOffsetsBase::Reduction::MEAN}});
    return enum_names;
}
std::ostream& operator<<(std::ostream& s, const op::util::EmbeddingBagOffsetsBase::Reduction& reduction) {
    return s << as_string(reduction);
}

AttributeAdapter<op::util::EmbeddingBagOffsetsBase::Reduction>::~AttributeAdapter() = default;
}  // namespace ov
