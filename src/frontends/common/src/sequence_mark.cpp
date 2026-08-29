// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/sequence_mark.hpp"

#include <optional>

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/sequence_erase.hpp"
#include "openvino/frontend/sequence_insert.hpp"
#include "openvino/op/identity.hpp"
#include "openvino/util/log.hpp"

namespace ov {
namespace frontend {

namespace {

ov::Output<ov::Node> unwrap_identity(const ov::Output<ov::Node>& value) {
    ov::Output<ov::Node> cur = value;
    while (auto identity = ov::as_type_ptr<ov::op::v16::Identity>(cur.get_node_shared_ptr())) {
        cur = identity->input_value(0);
    }
    return cur;
}

// Resolve a statically known sequence position into an index. A negative
// position counts from the back (Python list semantics): index == size + pos.
// `size_bias` is 1 for insert (append at index == size is valid) and 0 for
// erase. Returns nullopt when the position is dynamic or out of range.
std::optional<size_t> static_position(const ov::Output<ov::Node>& position, size_t size, int64_t size_bias) {
    auto pc = ov::util::get_constant_from_source(position);
    if (!pc) {
        return std::nullopt;
    }
    const auto values = pc->cast_vector<int64_t>();
    if (values.size() != 1) {
        return std::nullopt;
    }
    int64_t pos = values[0];
    if (pos < 0) {
        pos += static_cast<int64_t>(size);
    }
    if (pos < 0 || pos >= static_cast<int64_t>(size) + size_bias) {
        return std::nullopt;
    }
    return static_cast<size_t>(pos);
}

void enumerate_sequence(const ov::Output<ov::Node>& value, ov::OutputVector& out, int depth) {
    if (depth > 256) {
        // Recursion guard against pathological chains. Collapsing to one opaque
        // slot here undercounts a genuinely long Insert/Erase chain, so surface
        // it: a >256-deep sequence chain points at an unhandled pattern rather
        // than a real stack-depth risk.
        OPENVINO_DEBUG("SequenceMark::get_sequence: sequence chain exceeds depth 256; element count may be "
                       "undercounted.");
        out.push_back(value);
        return;
    }
    const auto v = unwrap_identity(value);
    const auto node = v.get_node_shared_ptr();
    if (auto mark = ov::as_type_ptr<SequenceMark>(node)) {
        for (size_t i = 0; i < mark->get_input_size(); ++i) {
            const auto in = mark->input_value(i);
            const auto in_node = unwrap_identity(in).get_node_shared_ptr();
            // Flatten only SequenceInsert/SequenceErase chains. A directly
            // nested SequenceMark represents an inline nested list/tuple
            // construct (e.g. PyTorch builds nested SequenceMarks); it must stay
            // a single opaque element so the returned element count and
            // index->element mapping keep matching context.get_output_size().
            if (ov::is_type<SequenceInsert>(in_node) || ov::is_type<SequenceErase>(in_node)) {
                enumerate_sequence(in, out, depth + 1);
            } else {
                out.push_back(in);
            }
        }
        return;
    }
    if (auto ins = ov::as_type_ptr<SequenceInsert>(node)) {
        ov::OutputVector base;
        enumerate_sequence(ins->get_input_sequence(), base, depth + 1);
        size_t pos = base.size();  // default (no position): append at end
        if (ins->has_position()) {
            auto resolved = static_position(ins->get_position(), base.size(), /*size_bias=*/1);
            if (!resolved) {
                // Non-constant (or out-of-range) position: the spliced index is
                // not statically known, so keep the whole SequenceInsert as a
                // single opaque element (documented contract). Downstream passes
                // resolve such dynamic positions structurally (Select chains) or
                // leave the reader unconverted.
                out.push_back(v);
                return;
            }
            pos = *resolved;
        }
        base.insert(base.begin() + static_cast<long>(pos), ins->get_tensor());
        out.insert(out.end(), base.begin(), base.end());
        return;
    }
    if (auto era = ov::as_type_ptr<SequenceErase>(node)) {
        ov::OutputVector base;
        enumerate_sequence(era->get_input_sequence(), base, depth + 1);
        if (!base.empty()) {
            size_t pos = base.size() - 1;  // default (no position): erase last
            if (era->has_position()) {
                auto resolved = static_position(era->get_position(), base.size(), /*size_bias=*/0);
                if (!resolved) {
                    // Non-constant (or out-of-range) position: keep the whole
                    // SequenceErase as a single opaque element (documented
                    // contract) rather than guessing erase-last.
                    out.push_back(v);
                    return;
                }
                pos = *resolved;
            }
            base.erase(base.begin() + static_cast<long>(pos));
        }
        out.insert(out.end(), base.begin(), base.end());
        return;
    }
    out.push_back(v);
}

}  // namespace

SequenceMark::SequenceMark(const ov::OutputVector& inputs) : ov::op::util::FrameworkNode(inputs, 1) {
    validate_and_infer_types();
}

void SequenceMark::validate_and_infer_types() {
    // Infer element type: if all inputs have the same type, use that type
    element::Type output_type = element::dynamic;
    if (get_input_size() > 0) {
        output_type = get_input_element_type(0);
        for (size_t i = 1; i < get_input_size(); ++i) {
            if (get_input_element_type(i) != output_type) {
                output_type = element::dynamic;
                break;
            }
        }
    }

    // Infer shape: if all inputs are 0D (scalars), output shape is 1D with size = number of elements
    // If any input is more than 0D, we cannot determine the shape (depends on concat axis)
    PartialShape output_shape = PartialShape::dynamic();
    if (get_input_size() > 0) {
        bool all_scalars = true;
        for (size_t i = 0; i < get_input_size(); ++i) {
            const auto& input_shape = get_input_partial_shape(i);
            if (input_shape.rank().is_dynamic() || input_shape.rank().get_length() != 0) {
                all_scalars = false;
                break;
            }
        }
        if (all_scalars) {
            output_shape = PartialShape{static_cast<int64_t>(get_input_size())};
        }
    }

    set_output_type(0, output_type, output_shape);
}

std::shared_ptr<Node> SequenceMark::clone_with_new_inputs(const OutputVector& inputs) const {
    return std::make_shared<SequenceMark>(inputs);
}

size_t SequenceMark::size() const {
    return get_input_size();
}

bool SequenceMark::empty() const {
    return get_input_size() == 0;
}

ov::Output<ov::Node> SequenceMark::get_element(size_t index) const {
    return input_value(index);
}

ov::OutputVector SequenceMark::get_sequence() const {
    ov::OutputVector result;
    // Enumeration is read-only; const_pointer_cast just adapts shared_from_this()
    // to the Output<Node> the enumerator expects.
    enumerate_sequence({std::const_pointer_cast<ov::Node>(shared_from_this()), 0}, result, 0);
    return result;
}

}  // namespace frontend
}  // namespace ov
