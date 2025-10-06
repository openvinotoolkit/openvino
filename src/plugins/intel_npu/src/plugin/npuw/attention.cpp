// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "attention.hpp"

#include "openvino/op/broadcast.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/util/op_types.hpp"  // is_parameter
#include "util.hpp"

std::optional<ov::npuw::function::Attention> ov::npuw::function::Attention::from(
    const std::shared_ptr<ov::Model>& model) {
    ov::npuw::function::Attention dyn;

    // Find the mask input (also sizeable). FIXME: We know too much at this point
    auto ops = model->get_ordered_ops();
    auto sdpa_iter = std::find_if(ops.begin(), ops.end(), [](auto&& node_ptr) {
        return ov::is_type<ov::op::v13::ScaledDotProductAttention>(node_ptr);
    });
    if (sdpa_iter == ops.end()) {
        LOG_WARN("SDPA is not found in the attn subgraph!");
        return std::nullopt;
    }

    // Traverse the SDPA's mask input upwards to find the proper Parameter.
    // Only unary ops are allowed along the way
    auto sdpa_node = *sdpa_iter;
    NPUW_ASSERT(sdpa_node->inputs().size() >= 4);

    auto mask_in_node = sdpa_node->inputs()[3].get_source_output().get_node_shared_ptr();
    while (mask_in_node && !ov::op::util::is_parameter(mask_in_node)) {
        if (mask_in_node->inputs().size() != 1) {
            LOG_WARN("Non-unary or disconnected op on the way from SDPA to input mask");
            return std::nullopt;
        }
        mask_in_node = mask_in_node->inputs()[0].get_source_output().get_node_shared_ptr();
    }
    NPUW_ASSERT(ov::op::util::is_parameter(mask_in_node));
    dyn._mask = std::static_pointer_cast<ov::op::v0::Parameter>(mask_in_node);
    dyn._mask_shape = dyn._mask->get_shape();

    // Find the attention inputs with dynamic range
    const auto& f_params = model->get_parameters();
    NPUW_ASSERT(f_params.size() > 0);

    auto find_context_dim = [&](const auto& param, auto&& f) {
        const auto& param_shape = param->get_shape();
        // Look for the dynamic parameter size - past size in this case
        // With our approach it is context_size - query_size
        auto past_len = dyn.context_len() - dyn.query_len();
        auto dim_iter = std::find(param_shape.begin(), param_shape.end(), past_len);
        if (dim_iter == param_shape.end()) {
            // No such dim found
            return false;
        }
        if (std::find(dim_iter + 1, param_shape.end(), past_len) != param_shape.end()) {
            // There must be no other such dim
            return false;
        }
        f(*dim_iter);
        return true;
    };

    for (auto&& param : f_params) {
        // A bad test but it is what it is
        if (ov::npuw::util::starts_with(param->get_friendly_name(), "past")) {
            if (!find_context_dim(param, [&](std::size_t dim) {
                    dyn._inputs.push_back(ov::npuw::function::Attention::Param{param, 2});
                })) {
                LOG_WARN("Couldn't identify SDPA parameter's dynamic dimension");
                return std::nullopt;
            }
        }
    }  // for(f_params)

    if (dyn._inputs.empty() || !dyn._mask) {
        return std::nullopt;
    }

    // Apply transformation to the model. Note: only function body is modified
    // Accumulate the reshape map
    std::map<ov::Output<ov::Node>, ov::PartialShape> new_shapes;
    for (auto&& p : dyn._inputs) {
        ov::PartialShape dyn_shape = p.param->get_shape();  // Here it is yet static
        dyn_shape[p.dim] = ov::Dimension();                 // ..and now is dynamic
        new_shapes[p.param->output(0)] = std::move(dyn_shape);
    }
    // Mask
    {
        ov::PartialShape dyn_shape = dyn._mask_shape;
        // Put the mask's innermost dimension dynamic
        *dyn_shape.rbegin() = ov::Dimension();
        new_shapes[dyn._mask->output(0)] = std::move(dyn_shape);
    }
    model->reshape(new_shapes);

    // Patch Broadcast constants if there's any. If there's broadcast in the attention
    // block, its shape argument is normally a precomputed Const (which would be
    // an expression/a subgraph in the original dynamic IR). Since we retrofit
    // dynamism into a static shape environment here, we need to patch it back.
    for (auto&& op : model->get_ordered_ops()) {
        if (!ov::is_type<ov::op::v3::Broadcast>(op)) {
            continue;
        }
        // Inspect the constant
        auto shape_source = op->input(1).get_source_output().get_node_shared_ptr();
        if (!ov::is_type<ov::op::v0::Constant>(shape_source)) {
            LOG_WARN("SDPA Broadcast's 2nd input is not Const: " << shape_source << ", skipping");
            continue;
        }

        auto shape_const = std::dynamic_pointer_cast<ov::op::v0::Constant>(shape_source);
        auto shape_values = shape_const->cast_vector<int32_t>();
        for (auto&& d : shape_values) {
            //  Assume the context length is the mask's innermost dimension
            if (static_cast<std::size_t>(d) == dyn.context_len()) {
                d = 1;
            }
        }
        auto new_const = std::make_shared<ov::op::v0::Constant>(shape_const->get_element_type(),
                                                                shape_const->get_shape(),
                                                                shape_values);
        op->input(1).replace_source_output(new_const);
    }
    model->validate_nodes_and_infer_types();

    return {std::move(dyn)};
}

ov::npuw::runtime::attention::PositionIDs::PositionIDs(std::size_t param_idx,
                                                       const ov::npuw::compiled::Attention& d,
                                                       const ov::ISyncInferRequest& rq)
    : m_position_ids_idx(param_idx),
      m_d(d),
      m_rq(rq) {
    // FIXME: speculative decode is indistinguishable at this point!
    m_case = m_d.query_size == 1 ? Case::GENERATE : Case::PREFILL;
}

ov::npuw::runtime::attention::Selector::Ptr ov::npuw::runtime::attention::PositionIDs::find(
    const ov::npuw::compiled::Attention& d,
    const ov::ISyncInferRequest& rq) {
    auto is_position_ids = [](const ov::Output<const ov::Node>& p) {
        const auto& shape = p.get_shape();
        // FIXME: 2D/3D position IDs are not supported here YET
        return p.get_node()->get_friendly_name() == "position_ids" &&
               (shape.size() == 1 || (shape.size() == 2 && shape[0] == 1));
    };

    const auto& inputs = rq.get_inputs();
    auto pos_ids_iter = std::find_if(inputs.begin(), inputs.end(), is_position_ids);
    if (pos_ids_iter != inputs.end()) {
        const auto param_idx = std::distance(inputs.begin(), pos_ids_iter);
        return Selector::Ptr{new PositionIDs(param_idx, d, rq)};
    }
    return Selector::Ptr{};
}

void ov::npuw::runtime::attention::PositionIDs::prepare() {
    const auto& iport = m_rq.get_compiled_model()->inputs()[m_position_ids_idx];
    const auto in_tensor = m_rq.get_tensor(iport);
    const auto in_dims = in_tensor->get_shape();

    // There's several cases possible:
    // a. Prefill input_ids, including chunk
    // b. Generate input_ids, 1
    // c. Generate input_ids, N (speculative)
    // Prefill (even chunked) is left-padded, so for (a) it's enough to take the last element.
    // Same works for b (there's no choise).
    // c may require traversing the tensor backwards as Generate with N>1 is right_padded (?)

    auto* pos_data_ptr = in_tensor->data<int64_t>();
    for (auto idx = in_dims.back() - 1; idx >= 0; idx--) {
        if (pos_data_ptr[idx] > 0) {
            // Initialize fields
            m_current_length = pos_data_ptr[idx];
            switch (m_case) {
            case Case::GENERATE:
                // decode case, we have pos_id-1 past elements to take from kvcache
                m_past_length = m_current_length;
                break;
            case Case::PREFILL:
                // chunked prefill case. calculate the past_length in full chunks
                // FIXME: We know too much about chunking here
                m_past_length = (m_current_length / m_d.query_size) * m_d.query_size;
                break;
            default:
                NPUW_ASSERT(false && "Reached the unreachable code");
            }
            return;
        }
    }
    LOG_WARN("Dynamic selector - no data found in the feature?");
    m_current_length = -1;
}

int64_t ov::npuw::runtime::attention::PositionIDs::length() const {
    return m_current_length;
}

int64_t ov::npuw::runtime::attention::PositionIDs::past_length() const {
    return m_past_length;
}
