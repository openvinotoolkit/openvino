// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unfold_sync_infer_request.hpp"

#include <algorithm>

#include "attn/attn_subgraph.hpp"
#include "compiled_model.hpp"
#include "host_flash_attention.hpp"
#include "infer_request_utils.hpp"
#include "logging.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "util.hpp"

ov::npuw::UnfoldInferRequest::UnfoldInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model)
    : ov::npuw::IBaseInferRequest(compiled_model) {
    // Create infer requests
    // Preallocate funcall tensors & substitute function call requests
    for (std::size_t i = 0; i < m_num_submodels; i++) {
        LOG_INFO("Creating infer request for Subgraph[" << i << "]...");
        LOG_BLOCK();
        auto& comp_model_desc = m_npuw_model->m_compiled_submodels[i];

        if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
            // no model & no funcall - optimized out, do nothing
            LOG_INFO("OPTIMIZED OUT");
            continue;
        }

        if (comp_model_desc.replaced_by) {
            // Pre-allocate output tensors for this function call
            const auto real_idx = comp_model_desc.replaced_by.value();
            auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
            if (proto_comp_model_desc.spatial) {
                NPUW_ASSERT(false && "Spatial is not supported in unfold");
            }
            if (ov::npuw::attn::get_compiled_dynamic(proto_comp_model_desc.pipeline.context) != nullptr) {
                NPUW_ASSERT(false && "Dynamic is not supported in unfold");
            }
        }  // if(replaced_by)

        const auto real_idx = comp_model_desc.replaced_by.value_or(i);
        auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
        // NB: UnfoldInferRequest is _NOT_ fail-safe! Fail means fail here
        m_subrequests[i] = proto_comp_model_desc.compiled_model->create_infer_request();
        LOG_EXEC(m_npuw_model->m_name << " Subgraph[" << i << "] on " << m_npuw_model->submodel_device(i)
                                      << ": create_infer_request() -> ireq@" << m_subrequests[i]._ptr.get()
                                      << (comp_model_desc.replaced_by ? " (from func Subgraph[" +
                                                                            std::to_string(real_idx) + "])"
                                                                      : std::string{}));
        if (const auto* hfa = ov::npuw::attn::get_compiled_hfa(proto_comp_model_desc.pipeline.context)) {
            // The request created above runs the final tile, see bind_hfa_call()
            HFACall call;
            call.idx = i;
            call.real_idx = real_idx;
            call.hfa = hfa;
            call.params.resize(proto_comp_model_desc.param_base);
            m_hfa_call_of[i] = m_hfa_calls.size();
            m_hfa_calls.push_back(std::move(call));
        }
        LOG_INFO("DONE");
    }  // for(submodels)

    alloc_quant_gather();

    LOG_INFO("Connecting subrequests...");
    LOG_BLOCK();
    for (const auto& kvp : m_npuw_model->m_submodels_input_to_prev_output) {
        const auto& subm_idx_to = kvp.first.first;
        const auto& port_idx_to = kvp.first.second;
        const auto& subm_idx_from = kvp.second.first;
        const auto& port_idx_from = kvp.second.second;

        LOG_DEBUG("Subgraph[" << subm_idx_from << "]/" << port_idx_from << " --> " << "Subgraph[" << subm_idx_to << "]/"
                              << port_idx_to);

        const auto is_optimized_out = [&](std::size_t idx) {
            const auto& desc = m_npuw_model->m_compiled_submodels[idx];
            return !desc.compiled_model && !desc.replaced_by;
        };
        if (is_optimized_out(subm_idx_from) || is_optimized_out(subm_idx_to)) {
            LOG_DEBUG("Skip: connection involves an optimized-out subgraph");
            continue;
        }

        NPUW_ASSERT(m_subrequests[subm_idx_from]);  // prod request is created
        NPUW_ASSERT(m_subrequests[subm_idx_to]);    // cons request is created
        NPUW_ASSERT(m_subrequests[subm_idx_from]._ptr != m_subrequests[subm_idx_to]._ptr);

        if (const auto it = m_hfa_call_of.find(subm_idx_to); it != m_hfa_call_of.end()) {
            // The parameters of an HFA call are not the inputs of its requests, see bind_hfa_call()
            const auto& oport = m_subrequests[subm_idx_from]->get_compiled_model()->outputs()[port_idx_from];
            auto& call = m_hfa_calls[it->second];
            call.params.at(port_idx_to) = m_subrequests[subm_idx_from]->get_tensor(oport);
            if (port_idx_to == call.hfa->_sdpa_attention_info._sdpa_indices.attention_mask) {
                call.mask_producer = subm_idx_from;
            }
            continue;
        }

        const auto& iport = m_subrequests[subm_idx_to]->get_compiled_model()->inputs()[port_idx_to];
        const auto& oport = m_subrequests[subm_idx_from]->get_compiled_model()->outputs()[port_idx_from];
        const auto& tensor = m_subrequests[subm_idx_from]->get_tensor(oport);
        LOG_DEBUG("Set Subgraph[" << subm_idx_to << "]/" << iport << " to Subgraph[" << subm_idx_from << "]/" << oport);
        m_subrequests[subm_idx_to]->set_tensor(iport, tensor);
    }  // for(map)
    LOG_INFO("Done");

    init_gio();

    for (size_t i = 0; i < m_num_submodels; i++) {
        LOG_VERB("Trying to preemptively set tensors for Subgraph[" << i << "]...");
        LOG_BLOCK();
        auto& comp_model_desc = m_npuw_model->m_compiled_submodels[i];
        if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
            continue;  // Optimized out
        }
        if (comp_model_desc.replaced_by) {
            unpack_closure(i, m_subrequests[i]);
        }
        LOG_VERB("Done");
    }

    if (!m_hfa_calls.empty()) {
        // Created by CompiledModel only with execution lists enabled, see create_base_infer_request(). The first tile
        // request of every call is created right away, to check that the tile model is on the NPU as well.
        try {
            for (auto& call : m_hfa_calls) {
                call.tile_requests.push_back(call.hfa->_compiled_tile_model->create_infer_request());
            }
            RqPtrs requests;
            for (const auto& request : m_subrequests) {
                if (request) {
                    requests.push_back(request);
                }
            }
            for (const auto& call : m_hfa_calls) {
                requests.push_back(call.tile_requests.front());
            }
            for (const auto& request : requests) {
                const auto context = request->get_compiled_model()->get_context();
                auto factory = std::dynamic_pointer_cast<::intel_npu::IExecutionListFactory>(context._ptr);
                OPENVINO_ASSERT(factory != nullptr, "a request is not on an NPU supporting execution lists");
                m_hfa_list_factory = std::move(factory);
            }
        } catch (const std::exception& ex) {
            m_hfa_list_factory.reset();
            LOG_EXEC(m_npuw_model->m_name << ": execution list not used: " << ex.what());
        }
        LOG_EXEC(m_npuw_model->m_name << ": " << m_hfa_calls.size()
                                      << " HFA calls run by UnfoldInferRequest with tile requests of their own, "
                                      << (m_hfa_list_factory ? "one execution list per segment of a chunk"
                                                             : "requests run one by one"));
    } else if (ov::npuw::util::exec_list_enabled_for(m_npuw_model->m_name)) {
        create_execution_list();
    }
}

bool ov::npuw::UnfoldInferRequest::bind_behavior_input(std::size_t idx,
                                                       std::size_t,
                                                       std::size_t input_idx,
                                                       const ov::SoPtr<ov::ITensor>& tensor,
                                                       RqPtr) {
    const auto it = m_hfa_call_of.find(idx);
    if (it == m_hfa_call_of.end()) {
        return false;
    }
    m_hfa_calls[it->second].params.at(input_idx) = tensor;
    return true;
}

ov::npuw::UnfoldInferRequest::HFAState& ov::npuw::UnfoldInferRequest::hfa_state(const HFACall& call) {
    auto [it, inserted] = m_hfa_states.try_emplace(call.real_idx);
    auto& state = it->second;
    if (!inserted) {
        return state;
    }

    const auto& tile_in = call.hfa->_sdpa_attention_info._tile_input_indices;
    const auto& inputs = call.hfa->_compiled_tile_model->inputs();
    const auto& final_inputs = call.hfa->_compiled_final_tile_model->inputs();
    const auto device = m_npuw_model->submodel_device(call.real_idx);
    const auto alloc = [&](std::size_t port) {
        const auto& iport = inputs.at(port);
        const auto& final_iport = final_inputs.at(port);
        OPENVINO_ASSERT(iport.get_element_type() == final_iport.get_element_type() &&
                            iport.get_partial_shape() == final_iport.get_partial_shape(),
                        "HFA: the regular and the final tile models must have the same state inputs");
        return allocMem(iport.get_element_type(), iport.get_shape(), device);
    };
    state.init_acc = alloc(tile_in.acc);
    state.init_max = alloc(tile_in.max);
    state.init_d = alloc(tile_in.d);
    state.acc = alloc(tile_in.acc);
    state.max = alloc(tile_in.max);
    state.d = alloc(tile_in.d);
    // Only read by the first tile of every call, so they keep their values
    runtime::host_flash_attention::HFARuntimeContext::initialize_state_tensors(state.init_acc,
                                                                               state.init_max,
                                                                               state.init_d);
    return state;
}

ov::npuw::UnfoldInferRequest::HFAMask& ov::npuw::UnfoldInferRequest::hfa_mask(const HFACall& call) {
    const auto& info = call.hfa->_sdpa_attention_info;
    const auto& source = call.params.at(info._sdpa_indices.attention_mask);
    OPENVINO_ASSERT(source, "HFA Subgraph[", call.idx, "]: the attention mask is not bound");
    auto [it, inserted] = m_hfa_masks.try_emplace(HFAMaskKey{call.real_idx, source->data()});
    auto& mask = it->second;
    if (inserted) {
        mask.source = source;
        const auto& port = call.hfa->_compiled_final_tile_model->inputs().at(info._tile_input_indices.mask);
        mask.final_tile =
            allocMem(port.get_element_type(), port.get_shape(), m_npuw_model->submodel_device(call.real_idx));
    }
    return mask;
}

void ov::npuw::UnfoldInferRequest::cut_hfa_mask(const HFACall& call, HFAMask& mask) {
    // Same regions as the attention runtime behavior: the regular tiles read the leading part of the mask, the final
    // tile its trailing part
    constexpr uint32_t MASK_KV_SEQ_DIM = 3;
    const auto& source = mask.source;
    OPENVINO_ASSERT(MASK_KV_SEQ_DIM < source->get_shape().size(), "HFA: mask sequence dimension is out of range");
    const int64_t mask_length = static_cast<int64_t>(source->get_shape()[MASK_KV_SEQ_DIM]);
    const int64_t past_tile_size = call.hfa->_past_tile_size;
    const int64_t final_tile_size = call.hfa->_final_tile_size;

    const auto copy_region = [&](const ov::SoPtr<ov::ITensor>& dst, int64_t offset, int64_t length) {
        OPENVINO_ASSERT(offset >= 0 && offset + length <= mask_length, "HFA: mask tile is out of the mask");
        OPENVINO_ASSERT(dst->get_element_type() == source->get_element_type(), "HFA: mask tile type mismatch");
        ov::npuw::util::copy_tensor_by_dim(ov::npuw::util::view(source, MASK_KV_SEQ_DIM, offset, length),
                                           dst,
                                           MASK_KV_SEQ_DIM,
                                           MASK_KV_SEQ_DIM);
    };
    for (std::size_t tile = 0; tile < mask.needed_regular; tile++) {
        copy_region(mask.regular.at(tile), static_cast<int64_t>(tile) * past_tile_size, past_tile_size);
    }
    copy_region(mask.final_tile, mask_length - final_tile_size, final_tile_size);
    mask.cut = true;
}

void ov::npuw::UnfoldInferRequest::bind_hfa_call(HFACall& call) {
    const auto* hfa = call.hfa;
    const auto& info = hfa->_sdpa_attention_info;
    const auto& sdpa_in = info._sdpa_indices;
    const auto& tile_in = info._tile_input_indices;
    const auto K_SEQ_DIM = static_cast<uint32_t>(info._k_seq_dim);
    const int64_t past_tile_size = hfa->_past_tile_size;
    const int64_t present_tile_size = static_cast<int64_t>(info._query_size);
    OPENVINO_ASSERT(hfa->_final_tile_size == present_tile_size,
                    "HFA: final tile size must equal the query size (PREFILL-only)");

    const auto param = [&](std::size_t param_idx) {
        const auto& tensor = call.params.at(param_idx);
        OPENVINO_ASSERT(tensor, "HFA Subgraph[", call.idx, "]: function parameter ", param_idx, " is not bound");
        return tensor;
    };

    const auto& present_key = param(sdpa_in.present_key);
    OPENVINO_ASSERT(static_cast<int64_t>(present_key->get_shape().at(K_SEQ_DIM)) == present_tile_size,
                    "HFA: present KV length must equal the compiled query/present tile size");
    call.tiles.clear();
    call.layer_kv_length = present_tile_size;
    for (std::size_t block = 0; block < sdpa_in.past_key_blocks.size(); block++) {
        const int64_t block_length =
            static_cast<int64_t>(param(sdpa_in.past_key_blocks[block])->get_shape().at(K_SEQ_DIM));
        OPENVINO_ASSERT(past_tile_size > 0 && block_length % past_tile_size == 0,
                        "HFA: KV block length must be a multiple of the past tile size");
        for (int64_t offset = 0; offset < block_length; offset += past_tile_size) {
            call.tiles.push_back({block, offset});
        }
        call.layer_kv_length += block_length;
    }

    // The final tile takes Q and the present K/V as they are: a host copy would read them before they are produced
    auto& final_request = m_subrequests[call.idx];
    const auto& final_inputs = hfa->_compiled_final_tile_model->inputs();
    const auto bind_whole = [&](std::size_t port, const ov::SoPtr<ov::ITensor>& tensor, const char* name) {
        const auto& iport = final_inputs.at(port);
        OPENVINO_ASSERT(tensor->get_element_type() == iport.get_element_type() &&
                            ov::PartialShape(tensor->get_shape()) == iport.get_partial_shape(),
                        "HFA Subgraph[",
                        call.idx,
                        "]: the final tile can't take the ",
                        name,
                        " tensor as it is, and a host copy is not possible with execution lists");
        final_request->set_tensor(iport, tensor);
    };
    bind_whole(tile_in.q, param(sdpa_in.query), "query");
    bind_whole(tile_in.k, present_key, "present key");
    bind_whole(tile_in.v, param(sdpa_in.present_value), "present value");
    final_request->set_tensor(final_inputs.at(tile_in.mask), hfa_mask(call).final_tile);

    call.bound_params.clear();
    for (const auto& tensor : call.params) {
        call.bound_params.push_back(tensor ? tensor->data() : nullptr);
    }
    call.bound_tile_requests = 0u;
    call.final_reads_init.reset();
}

void ov::npuw::UnfoldInferRequest::bind_hfa_tile(HFACall& call, std::size_t tile) {
    const auto* hfa = call.hfa;
    const auto& info = hfa->_sdpa_attention_info;
    const auto& sdpa_in = info._sdpa_indices;
    const auto& tile_in = info._tile_input_indices;
    const auto& tile_out = info._tile_output_indices;
    const auto& inputs = hfa->_compiled_tile_model->inputs();
    const auto& outputs = hfa->_compiled_tile_model->outputs();
    const int64_t past_tile_size = hfa->_past_tile_size;
    const auto& where = call.tiles.at(tile);
    auto& request = call.tile_requests.at(tile);

    request->set_tensor(inputs.at(tile_in.q), call.params.at(sdpa_in.query));

    const auto bind_kv = [&](std::size_t port, const ov::SoPtr<ov::ITensor>& block, uint32_t dim, const char* name) {
        const auto& iport = inputs.at(port);
        OPENVINO_ASSERT(block->get_element_type() == iport.get_element_type(),
                        "HFA Subgraph[",
                        call.idx,
                        "]: ",
                        name,
                        " tile type mismatch, a host conversion is not possible with execution lists");
        if (where.offset == 0 && static_cast<int64_t>(block->get_shape().at(dim)) == past_tile_size) {
            request->set_tensor(iport, block);  // The whole block
            return;
        }
        OPENVINO_ASSERT(hfa->_can_use_tensor_view,
                        "HFA Subgraph[",
                        call.idx,
                        "]: ",
                        name,
                        " tiles need strided views, which the device does not support");
        request->set_tensor(iport, ov::npuw::util::view(block, dim, where.offset, past_tile_size));
    };
    bind_kv(tile_in.k,
            call.params.at(sdpa_in.past_key_blocks.at(where.block)),
            static_cast<uint32_t>(info._k_seq_dim),
            "K");
    bind_kv(tile_in.v,
            call.params.at(sdpa_in.past_value_blocks.at(where.block)),
            static_cast<uint32_t>(info._v_seq_dim),
            "V");

    // With mask skipping (fused HFA), the regular tile model has no mask input
    if (tile_in.mask < inputs.size()) {
        auto& mask = hfa_mask(call);
        while (mask.regular.size() <= tile) {
            const auto& iport = inputs.at(tile_in.mask);
            mask.regular.push_back(
                allocMem(iport.get_element_type(), iport.get_shape(), m_npuw_model->submodel_device(call.real_idx)));
        }
        request->set_tensor(inputs.at(tile_in.mask), mask.regular.at(tile));
    }

    // The first tile starts from the initial state, the others continue the state in place
    auto& state = hfa_state(call);
    request->set_tensor(inputs.at(tile_in.acc), tile == 0 ? state.init_acc : state.acc);
    request->set_tensor(inputs.at(tile_in.max), tile == 0 ? state.init_max : state.max);
    request->set_tensor(inputs.at(tile_in.d), tile == 0 ? state.init_d : state.d);
    request->set_tensor(outputs.at(tile_out.acc), state.acc);
    request->set_tensor(outputs.at(tile_out.max), state.max);
    request->set_tensor(outputs.at(tile_out.d), state.d);
}

void ov::npuw::UnfoldInferRequest::prepare_hfa_calls() {
    using namespace ov::npuw::runtime::host_flash_attention;
    if (!m_hfa_selector) {
        const auto query_size = m_hfa_calls.front().hfa->_sdpa_attention_info._query_size;
        m_hfa_selector = PositionIDs::find(query_size, *this);
        if (!m_hfa_selector) {
            m_hfa_selector = HistorySize::find(query_size, *this);
        }
        OPENVINO_ASSERT(m_hfa_selector,
                        "HFA: the model has no position_ids input and the npuw_history_size state was not set "
                        "before the first infer()");
    }
    m_hfa_selector->prepare(get_history_size());
    OPENVINO_ASSERT(m_hfa_selector->this_case() == Selector::Case::PREFILL, "HFA supports only PREFILL");
    const int64_t context_length = m_hfa_selector->context_length();

    for (auto& entry : m_hfa_masks) {
        entry.second.needed_regular = 0u;
        entry.second.cut = false;
    }

    for (auto& call : m_hfa_calls) {
        bool rebind = call.bound_params.size() != call.params.size();
        for (std::size_t i = 0; !rebind && i < call.params.size(); i++) {
            rebind = (call.params[i] ? call.params[i]->data() : nullptr) != call.bound_params[i];
        }
        if (rebind) {
            bind_hfa_call(call);
        }

        const int64_t present_tile_size = static_cast<int64_t>(call.hfa->_sdpa_attention_info._query_size);
        const int64_t past_tile_size = call.hfa->_past_tile_size;
        const int64_t past_length = std::min(context_length, call.layer_kv_length) - present_tile_size;
        OPENVINO_ASSERT(past_length >= 0 && past_length % past_tile_size == 0,
                        "HFA: past length must be a multiple of the past tile size");
        call.past_tiles = static_cast<std::size_t>(past_length / past_tile_size);
        OPENVINO_ASSERT(call.past_tiles <= call.tiles.size(), "HFA: more past tiles than the KV blocks hold");

        while (call.tile_requests.size() < call.past_tiles) {
            call.tile_requests.push_back(call.hfa->_compiled_tile_model->create_infer_request());
        }
        for (; call.bound_tile_requests < call.past_tiles; call.bound_tile_requests++) {
            bind_hfa_tile(call, call.bound_tile_requests);
        }

        // Without past tiles, the final tile starts from the initial state
        const auto& tile_in = call.hfa->_sdpa_attention_info._tile_input_indices;
        const bool reads_init = call.past_tiles == 0;
        if (call.final_reads_init != reads_init) {
            const auto& final_inputs = call.hfa->_compiled_final_tile_model->inputs();
            auto& state = hfa_state(call);
            auto& final_request = m_subrequests[call.idx];
            final_request->set_tensor(final_inputs.at(tile_in.acc), reads_init ? state.init_acc : state.acc);
            final_request->set_tensor(final_inputs.at(tile_in.max), reads_init ? state.init_max : state.max);
            final_request->set_tensor(final_inputs.at(tile_in.d), reads_init ? state.init_d : state.d);
            call.final_reads_init = reads_init;
        }

        auto& mask = hfa_mask(call);
        if (tile_in.mask < call.hfa->_compiled_tile_model->inputs().size()) {
            mask.needed_regular = std::max(mask.needed_regular, call.past_tiles);
        }
    }
}

void ov::npuw::UnfoldInferRequest::run_hfa_segment(RqPtrs& requests, std::size_t segment) {
    if (requests.empty()) {
        return;
    }
    const std::string tag = " [segment " + std::to_string(segment) + ", " + std::to_string(requests.size()) +
                            (m_hfa_list_factory ? " graph executes in one execution list]" : " requests]");
    if (m_hfa_list_factory) {
        if (m_hfa_lists.size() <= segment) {
            m_hfa_lists.resize(segment + 1);
        }
        auto& list = m_hfa_lists[segment];
        if (!list) {
            list = m_hfa_list_factory->create_execution_list();
        }
        // Recorded again for every chunk: the number of tiles changes from chunk to chunk
        list->reset();
        for (const auto& request : requests) {
            list->append(request._ptr);
        }
        exec_trace(0, requests.front(), "start_async()", tag);
        list->start_async();
        list->wait();
        exec_trace(0, requests.back(), "wait() returned", tag);
    } else {
        for (const auto& request : requests) {
            exec_trace(0, request, "start_async()", tag);
            request->start_async();
            request->wait();
            exec_trace(0, request, "wait() returned", tag);
        }
    }
    requests.clear();
}

void ov::npuw::UnfoldInferRequest::infer_with_hfa() {
    for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
        if (m_subrequests[idx]) {
            bind_global_params(idx, m_subrequests[idx]);
            bind_global_results(idx, m_subrequests[idx]);
        }
    }
    prepare_hfa_calls();

    // A mask that is a global parameter is known now. A mask produced by a subgraph is cut after that subgraph has
    // run, so the requests before the first call using it go into a segment of their own.
    for (const auto& call : m_hfa_calls) {
        auto& mask = hfa_mask(call);
        if (!call.mask_producer && !mask.cut) {
            cut_hfa_mask(call, mask);
        }
    }

    RqPtrs pending;
    std::size_t segment = 0u;
    for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
        if (!m_subrequests[idx]) {
            continue;
        }
        if (const auto it = m_hfa_call_of.find(idx); it != m_hfa_call_of.end()) {
            const auto& call = m_hfa_calls[it->second];
            auto& mask = hfa_mask(call);
            if (!mask.cut) {
                run_hfa_segment(pending, segment++);
                cut_hfa_mask(call, mask);
            }
            const auto past_tiles = static_cast<std::ptrdiff_t>(call.past_tiles);
            pending.insert(pending.end(), call.tile_requests.begin(), call.tile_requests.begin() + past_tiles);
        }
        pending.push_back(m_subrequests[idx]);
    }
    run_hfa_segment(pending, segment);
}

void ov::npuw::UnfoldInferRequest::create_execution_list() {
    // Check every subgraph first: falling back is only possible before the first append(), since an
    // appended request is owned by its list and can't run on its own anymore.
    std::vector<std::pair<std::size_t, std::shared_ptr<::intel_npu::IExecutionListFactory>>> members;
    try {
        for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
            if (!m_subrequests[idx]) {
                continue;
            }
            const auto context = m_subrequests[idx]->get_compiled_model()->get_context();
            auto factory = std::dynamic_pointer_cast<::intel_npu::IExecutionListFactory>(context._ptr);
            OPENVINO_ASSERT(factory != nullptr, "Subgraph[", idx, "] is not on an NPU supporting execution lists");
            members.emplace_back(idx, std::move(factory));
        }
    } catch (const std::exception& ex) {
        LOG_EXEC(m_npuw_model->m_name << ": execution list not used: " << ex.what());
        return;
    }
    if (members.empty()) {
        return;
    }

    // No prepare() here: it would allocate device-visible buffers for every global parameter that is
    // not bound yet (e.g. a full KV cache per attention call). The list records on its first
    // start_async(), after infer() has bound everything.
    m_exec_list = {members.front().first, members.front().first, 0u, members.front().second->create_execution_list()};
    for (const auto& member : members) {
        m_exec_list.list->append(m_subrequests[member.first]._ptr);
        m_exec_list.last = member.first;
        m_exec_list.executes++;
    }
    LOG_EXEC(m_npuw_model->m_name << ": execution list active, Subgraph[" << m_exec_list.first << ".."
                                  << m_exec_list.last << "], " << m_exec_list.executes
                                  << " graph executes (recorded on first infer)");
}

bool ov::npuw::UnfoldInferRequest::valid_subrequest(std::size_t idx) const {
    return m_subrequests.at(idx) != nullptr;
}

void ov::npuw::UnfoldInferRequest::infer() {
    exec_trace_begin("UnfoldInferRequest");
    if (!m_hfa_calls.empty()) {
        infer_with_hfa();
        exec_trace_end();
        return;
    }
    const bool do_async = m_npuw_model->m_cfg.get<::intel_npu::NPUW_FUNCALL_ASYNC>();

    auto prepare = [&](std::size_t idx) {
        if (idx >= m_subrequests.size()) {
            return;
        }
        bind_global_params(idx, m_subrequests[idx]);
        bind_global_results(idx, m_subrequests[idx]);
    };
    // Subgraph index of each request passed to wait_and_clear(), for exec tracing only
    std::vector<std::size_t> previous_idx;
    auto wait_and_clear = [&](RqPtrs& rqs) {
        for (std::size_t i = 0; i < rqs.size(); i++) {
            rqs[i]->wait();
            exec_trace(previous_idx[i], rqs[i], "wait() returned");
        }
        rqs.clear();
        previous_idx.clear();
    };

    if (m_exec_list.list) {
        // PROTOTYPE: one submission on the list's own queue; barriers inside it order the executes
        const auto& el = m_exec_list;
        for (std::size_t idx = el.first; idx <= el.last; idx++) {
            if (m_subrequests[idx]) {
                prepare(idx);
            }
        }
        const std::string tag = " [execution list Subgraph[" + std::to_string(el.first) + ".." +
                                std::to_string(el.last) + "], " + std::to_string(el.executes) + " graph executes]";
        exec_trace(el.first, m_subrequests[el.first], "start_async()", tag);
        el.list->start_async();
        el.list->wait();
        exec_trace(el.last, m_subrequests[el.last], "wait() returned", tag);
    } else if (do_async) {
        std::size_t past_repl_id = 0u;
        RqPtrs previous_requests;

        prepare(0);
        for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
            auto& subr = m_subrequests[idx];
            if (!subr) {
                prepare(idx + 1);
                continue;
            }
            auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
            const auto this_repl_id = comp_model_desc.replaced_by.value_or(idx);
            if (this_repl_id != past_repl_id) {
                // For non-repeating blocks, the above value_or returns idx
                // For repeating blocks, it returns the function group id
                // If either is not equal to the past_repl_id, make a barrier here
                wait_and_clear(previous_requests);
                past_repl_id = this_repl_id;
            }
            exec_trace(idx, subr, "start_async()");
            subr->start_async();
            previous_requests.push_back(subr);
            previous_idx.push_back(idx);
            prepare(idx + 1);
        }
        wait_and_clear(previous_requests);
    } else {
        prepare(0);
        for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
            auto& subr = m_subrequests[idx];
            if (!subr) {
                prepare(idx + 1);
                continue;
            }
            exec_trace(idx, subr, "start_async()");
            subr->start_async();
            prepare(idx + 1);
            subr->wait();
            exec_trace(idx, subr, "wait() returned");
        }
    }  // (async)
    exec_trace_end();
}
