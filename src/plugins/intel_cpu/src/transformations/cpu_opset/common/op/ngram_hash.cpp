// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngram_hash.hpp"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/op.hpp"
#include "transformations/itt.hpp"

ov::intel_cpu::NgramHashNode::NgramHashNode(const ov::OutputVector& tokens,
                                            std::vector<int64_t> multipliers,
                                            std::vector<int64_t> moduli,
                                            std::vector<int64_t> offsets)
    : Op(tokens),
      m_multipliers(std::move(multipliers)),
      m_moduli(std::move(moduli)),
      m_offsets(std::move(offsets)) {
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::intel_cpu::NgramHashNode::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(NgramHashNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::NgramHashNode>(new_args, m_multipliers, m_moduli, m_offsets);
}

bool ov::intel_cpu::NgramHashNode::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(NgramHashNode_visit_attributes);
    visitor.on_attribute("multipliers", m_multipliers);
    visitor.on_attribute("moduli", m_moduli);
    visitor.on_attribute("offsets", m_offsets);
    return true;
}

void ov::intel_cpu::NgramHashNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(NgramHashNode_validate_and_infer_types);
    OPENVINO_ASSERT(get_input_size() > 0 && get_input_size() == m_multipliers.size(),
                    "NgramHash expects one multiplier per token input");
    OPENVINO_ASSERT(!m_moduli.empty() && m_moduli.size() == m_offsets.size(),
                    "NgramHash expects one offset per modulus");
    OPENVINO_ASSERT(std::all_of(m_moduli.begin(),
                                m_moduli.end(),
                                [](int64_t m) {
                                    return m > 0;
                                }),
                    "NgramHash moduli must be positive");

    const auto& et = get_input_element_type(0);
    OPENVINO_ASSERT(et.is_integral_number(), "NgramHash token inputs must be integer, got ", et);
    auto shape = get_input_partial_shape(0);
    for (size_t i = 1; i < get_input_size(); ++i) {
        OPENVINO_ASSERT(get_input_element_type(i) == et, "NgramHash token inputs must share one element type");
        OPENVINO_ASSERT(ov::PartialShape::merge_into(shape, get_input_partial_shape(i)),
                        "NgramHash token inputs must share one shape");
    }
    if (shape.rank().is_static()) {
        shape.push_back(static_cast<int64_t>(m_moduli.size()));
    }
    set_output_type(0, et, shape);
}
