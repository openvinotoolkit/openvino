// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "just_sync_infer_request.hpp"

#include <algorithm>
#include <memory>
#include <string>

#include "compiled_model.hpp"
#include "host_flash_attention.hpp"
#include "infer_request_utils.hpp"  // to utilize copy_tensor_by_dim
#include "logging.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "plugin.hpp"
#include "pyramid_attention.hpp"
#include "weights_bank.hpp"

ov::npuw::MemAccessSim::MemAccessSim(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model) {
    LOG_VERB("Running memory access simulation...");
    LOG_BLOCK();

    // Initialize the read list
    m_read_list.resize(compiled_model->m_compiled_submodels.size());

    // Initialize read counters for tensors in the graph:
    // 1. Interconnect
    for (const auto& kvp : compiled_model->m_submodels_input_to_prev_output) {
        const auto& read_to = kvp.first;     // who reads
        const auto& read_from = kvp.second;  // reads what

        if (read_to == CompiledModel::NO_LINK || read_from == CompiledModel::NO_LINK) {
            continue;
        }

        // Record # of reads for this particular Source
        m_remaining_reads[read_from]++;

        // Record a read request for this particular Subgraph (who reads the Source)
        m_read_list[read_to.first].push_back(read_from);
    }
    // 2. Global model's outputs
    for (auto&& read_from : compiled_model->m_outputs_to_submodels_outputs) {
        m_remaining_reads[read_from]++;
    }

    LOG_VERB("Done");
}

const ov::npuw::MemAccessSim::ReadList& ov::npuw::MemAccessSim::read_list(std::size_t idx) const {
    return m_read_list.at(idx);
}

std::size_t ov::npuw::MemAccessSim::remaining_reads(const LinkFrom& from) {
    return m_remaining_reads.at(from);
}

void ov::npuw::MemAccessSim::register_read(const LinkFrom& from) {
    m_remaining_reads.at(from)--;
}

ov::npuw::FuncMemMgr::FuncMemMgr(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model)
    : m_sim(compiled_model),
      m_model(compiled_model) {}

void ov::npuw::FuncMemMgr::set_alloc(AllocFcn&& fcn) {
    m_alloc = std::move(fcn);
}

void ov::npuw::FuncMemMgr::assign_memory() {
    LOG_VERB("Assigning function memory...");
    LOG_BLOCK();

    const auto num_submodels = m_model->m_compiled_submodels.size();

    // Walk over the subgraphs, pre-allocate and pre-assign tensors to the subgraphs
    // outputs.
    for (std::size_t idx = 0u; idx < num_submodels; idx++) {
        LOG_VERB("Process Subgraph[" << idx << "]");
        LOG_BLOCK();
        const auto& comp_model_desc = m_model->m_compiled_submodels[idx];
        if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
            // no model & no funcall - optimized out, do nothing
            continue;
        }

        // Simulate subgraph execution: poll its input list first
        const auto& read_list = m_sim.read_list(idx);

        // Now, get the outputs for the subgraph. If it is "regular", there's
        // nothing to do - this subgraph owns its outputs on its own.
        // If it is a function, though - look up in the function's memory storage.
        if (comp_model_desc.replaced_by) {
            const auto real_idx = comp_model_desc.replaced_by.value();
            const auto& proto_comp_model_desc = m_model->m_compiled_submodels[real_idx];

            const auto num_outs = proto_comp_model_desc.compiled_model->outputs().size();
            for (std::size_t out_idx = 0u; out_idx < num_outs; out_idx++) {
                const LinkFrom this_out = LinkFrom{idx, out_idx};
                assign(this_out);
            }
        }

        // Here happens the imaginary execution... Hocus pocus, done - that's a
        // simulation after all
        // After the execution, mark that the read_list was read.
        for (auto&& from : read_list) {
            m_sim.register_read(from);
        }
        LOG_VERB("Done");
    }

    // Report memory residency
    for (auto&& m : m_memory) {
        LOG_VERB("Function " << m.first.first << "/out port " << m.first.second << " : maximum memory residency "
                             << m.second.size() << " tensor(s)");
    }

    LOG_VERB("Done");
}

void ov::npuw::FuncMemMgr::assign(const LinkFrom& from) {
    // This method is the center of the function memory management.
    // The logic is simple:
    // - Look for an output tensor to reuse
    //   - If there's one, assign it to this allocation
    //   - If there's none, allocate a new tensor
    // - How a tensor to reuse is piced:
    //   1. It should exist
    //   2. It's "remaining reads" count should be 0 (all planned reads
    //      happened at this point).
    // The tensor storage is organized like this:
    // - Function: Here we use .replaced_by as a function identifier; taken from `from`
    //   - Output index: taken from `from`
    //     - A vector of resident tensors

    LOG_VERB("Assinging tensor for Subgraph[" << from.first << "]/" << from.second << "...");
    LOG_BLOCK();

    const auto& comp_model_desc = m_model->m_compiled_submodels[from.first];
    NPUW_ASSERT(comp_model_desc.replaced_by.has_value());

    const auto real_idx = comp_model_desc.replaced_by.value();

    FO func_output = {real_idx, from.second};
    auto& assigned_memory = m_memory[func_output];
    auto asgn_iter = std::find_if(assigned_memory.begin(), assigned_memory.end(), [&](Assignment& a) {
        return m_sim.remaining_reads(a.from) == 0u;
    });
    if (asgn_iter != assigned_memory.end()) {
        // Reassign this memory slot to the new "from"
        asgn_iter->from = from;
        m_table[from] = asgn_iter->ptr;
    } else {
        // No free space at this point - allocate a new tensor
        const auto& proto_comp_model_desc = m_model->m_compiled_submodels[real_idx];
        const auto& proto_comp_model = proto_comp_model_desc.compiled_model;

        const auto& oport = proto_comp_model->outputs()[from.second];
        ov::Shape oshape = oport.get_shape();

        if (proto_comp_model_desc.spatial) {
            oshape[proto_comp_model_desc.spatial->out_dim] = proto_comp_model_desc.spatial->range;
        }
        const auto& device = m_model->funcall_mem_device(real_idx);
        // FIXME: handle the lazy way (see BaseSyncInferRequest::get_tensor())
        // and share between submodels to reduce memory consumption
        TensorPtr new_tensor = m_alloc(oport.get_element_type(), oshape, device);
        NPUW_ASSERT(new_tensor);

        assigned_memory.push_back(Assignment{new_tensor, from});
        m_table[from] = new_tensor;
    }
    LOG_VERB("Done");
}

ov::npuw::TensorPtr ov::npuw::FuncMemMgr::get_tensor(const LinkFrom& from) {
    return m_table.at(from);
}

ov::npuw::JustInferRequest::JustInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model)
    : IBaseInferRequest(compiled_model),
      m_func_mem_mgr(compiled_model) {
    using namespace std::placeholders;
    m_func_mem_mgr.set_alloc(std::bind(&JustInferRequest::allocMem, this, _1, _2, _3));
    m_func_mem_mgr.assign_memory();

    m_closure_update_required = m_npuw_model->m_cfg.get<::intel_npu::NPUW_FOLD>();
    m_use_function_pipelining = m_npuw_model->m_cfg.get<::intel_npu::NPUW_FUNCALL_ASYNC>();
    if (m_use_function_pipelining) {
        LOG_WARN("Function call pipelining is enabled for " << m_npuw_model->m_name
                                                            << ", expect a higher memory consumption");
        m_funcall_pipeline.resize(m_num_submodels);
    }

    m_spatial_io.resize(m_num_submodels);
    m_attention_io.resize(m_num_submodels);
    m_hfa_io.resize(m_num_submodels);

    // Create infer requests
    // Preallocate funcall tensors & substitute function call requests
    bool failover_happened = false;
    bool has_spatial = false;
    bool has_dynamic = false;
    bool has_pyramid = false;
    bool has_hfa = false;
    std::size_t dynamic_sub_idx = -1;
    std::size_t pyramid_sub_idx = -1;
    std::size_t hfa_sub_idx = -1;
    for (size_t i = 0; i < m_num_submodels; i++) {
        LOG_INFO("Creating infer request for Subgraph[" << i << "]...");
        LOG_BLOCK();
        auto& comp_model_desc = m_npuw_model->m_compiled_submodels[i];

        if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
            // no model & no funcall - optimized out, do nothing
            LOG_INFO("OPTIMIZED OUT");
            continue;
        }

        // FIXME: Shouldn't this be handled by the base class? (in create_tensor)
        // A special case for function calls
        if (comp_model_desc.replaced_by) {
            // Pre-allocate output tensors for this function call
            const auto real_idx = comp_model_desc.replaced_by.value();
            auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
            auto& proto_comp_model = proto_comp_model_desc.compiled_model;
            const auto num_outputs = proto_comp_model->outputs().size();

            // Initialize the spatial IO placeholders, if required
            if (proto_comp_model_desc.spatial) {
                has_spatial = true;

                m_spatial_io[real_idx].inputs.resize(proto_comp_model_desc.param_base);
                m_spatial_io[real_idx].input_tails.resize(proto_comp_model_desc.param_base);
                m_spatial_io[real_idx].outputs.resize(num_outputs);
                m_spatial_io[real_idx].output_tails.resize(num_outputs);

                if (proto_comp_model_desc.spatial->tail_size) {
                    // Preallocate extra buffers for tail processing
                    // Note: these buffers are allocated to the entire NWAY (> tail_size)
                    for (auto&& p : proto_comp_model_desc.spatial->params) {
                        const auto& iport = proto_comp_model_desc.compiled_model->inputs()[p.idx];
                        m_spatial_io[real_idx].input_tails[p.idx] =
                            allocOut(iport, m_npuw_model->funcall_mem_device(real_idx));
                    }
                    const auto num_outs = proto_comp_model_desc.compiled_model->outputs().size();
                    for (std::size_t out_idx = 0u; out_idx < num_outs; out_idx++) {
                        const auto& oport = proto_comp_model_desc.compiled_model->outputs()[out_idx];
                        m_spatial_io[real_idx].output_tails[out_idx] =
                            allocOut(oport, m_npuw_model->funcall_mem_device(real_idx));
                    }
                }
            }  // if(spatial)

            // Initialize the dynamic IO placeholders, if required
            if (proto_comp_model_desc.attention) {
                // Sanity check first
                if (has_dynamic && dynamic_sub_idx != real_idx) {
                    OPENVINO_THROW("Only single attention type is permitted for model");
                }
                has_dynamic = true;
                dynamic_sub_idx = real_idx;
                m_attention_io[i].inputs.resize(proto_comp_model_desc.param_base);
            }  // if(dynamic)

            if (proto_comp_model_desc.pyramid_attention) {
                // Sanity check first
                if (has_pyramid && pyramid_sub_idx != real_idx) {
                    OPENVINO_THROW("Only single pyramid attention type is permitted for model");
                }
                has_pyramid = true;
                pyramid_sub_idx = real_idx;
                m_attention_io[i].inputs.resize(proto_comp_model_desc.param_base);
            }  // if(pyramid)

            if (proto_comp_model_desc.host_flash_attention) {
                // Sanity check first
                if (has_hfa && hfa_sub_idx != real_idx) {
                    OPENVINO_THROW("Only single flash attention type is permitted for model");
                }
                has_hfa = true;
                hfa_sub_idx = real_idx;
                m_hfa_io[i].inputs.resize(proto_comp_model_desc.param_base);
                const auto num_outputs =
                    proto_comp_model_desc.host_flash_attention.value()._compiled_final_tile_model->outputs().size();
                m_hfa_io[i].outputs.resize(num_outputs);
            }  // if(hfa)

            for (size_t out_idx = 0; out_idx < num_outputs; out_idx++) {
                const auto from = LinkFrom{i, out_idx};
                m_funcall_result[from] = m_func_mem_mgr.get_tensor(from);
            }
            if (real_idx != i) {
                // If this function call is NOT the function body, do nothing here - the original
                // request will be used.
                LOG_INFO("REUSE " << real_idx);
                continue;
            }
        }  // if(replaced_by)

        // Special cases are handled -- so nothing to do here
        const bool is_piped = is_pipelined(i);
        bool recompiled = false;
        auto rqs = create_infer_requests(i, is_piped ? 2 : 1, &recompiled);
        failover_happened |= recompiled;
        m_subrequests[i] = rqs.at(0);
        m_subrequest_devices[i] = *comp_model_desc.device_it;
        if (is_piped) {
            m_funcall_pipeline[i].subrequest = rqs.at(1);
        }

        // Create pyramid attention infer requests if this function has pyramid attention
        // IMPORTANT: Must be created AFTER main infer requests to enable direct reuse:
        // The last pyramid infer request directly reuses the main infer request object
        if (comp_model_desc.replaced_by) {
            const auto real_idx = comp_model_desc.replaced_by.value();
            auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
            if (proto_comp_model_desc.pyramid_attention) {
                setup_pyramid_infer_requests(real_idx, is_piped, false);
            }
            // Create HFA tile infer requests if this function has host flash attention
            if (proto_comp_model_desc.host_flash_attention) {
                setup_hfa_infer_requests(real_idx,
                                         is_piped,
                                         /* is_recreate */ false,
                                         /* enable_hfa_optimizations */ true);
            }
        }

        LOG_INFO("DONE");
    }  // for(submodels)

    if (failover_happened) {
        LOG_INFO("Refined device distribution:");
        LOG_BLOCK();
        m_npuw_model->log_device_dist();
    }

    // Identify connections for the funcall pipeline, if needed
    if (m_use_function_pipelining) {
        LOG_INFO("Setting up the funcall pipeline...");
        LOG_BLOCK();
        std::vector<std::optional<std::size_t>> prevs(m_num_submodels);
        for (std::size_t i = 0; i < m_num_submodels; i++) {
            auto& comp_model_desc = m_npuw_model->m_compiled_submodels[i];
            if (comp_model_desc.replaced_by) {  // a function call..
                if (!is_pipelined(i)) {
                    LOG_INFO("Skip subgraph[" << i << "] as it is a single-call function");
                    continue;
                }
                // Use real_id to accumulate information about
                // different functions
                const auto real_id = comp_model_desc.replaced_by.value();
                if (!prevs[real_id]) {  // ..met for a first time
                    LOG_INFO("Mark subgraph[" << i << "] as a head of pipeline...");
                    m_funcall_heads.push_back(i);
                } else {  // ..seen before
                    // Record that _this_ id follows the last known
                    // _prev_ if in the funcall pipeline
                    const auto prev_id = prevs[real_id].value();
                    LOG_INFO("Mark subgraph[" << i << "] as a successor of subraph[" << prev_id
                                              << "] in the function pipeline");
                    m_funcall_pipeline[prev_id].next = {i};
                }
                prevs[real_id] = {i};
            }  // if (replaced_by)
        }
    }  // if(function_pipelining)

    alloc_quant_gather();
    connect_subrequests();
    init_gio();

    for (size_t i = 0; i < m_num_submodels; i++) {
        LOG_VERB("Trying to preemptively set tensors for Subgraph[" << i << "]...");
        LOG_BLOCK();
        auto& comp_model_desc = m_npuw_model->m_compiled_submodels[i];
        // FIXME: figure out our cases and if this should be replaced with &&
        // Note: replaced_by is utilized below unconditionally
        if (!comp_model_desc.compiled_model || !comp_model_desc.replaced_by) {
            continue;
        }
        const auto real_idx = comp_model_desc.replaced_by.value();
        auto& func_desc = m_npuw_model->m_compiled_submodels[real_idx];

        // So - closure update is NOT required, OR the function is SINGLE -
        // just handle it's closure here and don't do it in runtime
        if (!m_closure_update_required || func_desc.forced_to_fcall) {
            unpack_closure(i, m_subrequests[real_idx]);
        }
        LOG_VERB("Done");
    }

    // Handle spatial dynamic submission
    if (has_spatial) {
        if (m_npuw_model->m_cfg.get<::intel_npu::NPUW_SPATIAL_DYN>()) {
            LOG_VERB("Finding spatial features...");
            LOG_BLOCK();
            m_spatial_selector = runtime::spatial::AttentionMask::find(*this);
            if (!m_spatial_selector) {
                LOG_WARN("Spatial capability is enabled, but no run-time features were found.");
                // Fallback selector to ALL
                m_spatial_selector.reset(new runtime::spatial::All());
            }
        } else {
            // Just force selector to ALL
            m_spatial_selector.reset(new runtime::spatial::All());
        }
        LOG_VERB("Done");
    }

    // Handle dynamic submission
    if (has_dynamic) {
        if (!m_npuw_model->m_cfg.get<::intel_npu::NPUW_ATTN_DYN>()) {
            // Even if the attention is detected and ready to go dynamic,
            // force it on the full range
            LOG_WARN("Dynamic capability is enabled, but won't be used due to user preference");
            m_attention_selector.reset(new runtime::attention::All());
        } else {
            const auto& dyn = m_npuw_model->m_compiled_submodels.at(dynamic_sub_idx).attention.value();
            m_attention_selector = runtime::attention::PositionIDs::find(dyn, *this);
            if (!m_attention_selector) {
                LOG_WARN("Dynamic capability is enabled, but no run-time features were found.");
                m_attention_selector.reset(new runtime::attention::All());
            }
        }
        LOG_VERB("Done");
    }

    // Handle pyramid attention
    if (has_pyramid) {
        const auto& pyramid_dyn = m_npuw_model->m_compiled_submodels.at(pyramid_sub_idx).pyramid_attention.value();
        const auto pyramid_count = pyramid_dyn._compiled_models.size();
        if (!m_npuw_model->m_cfg.get<::intel_npu::NPUW_ATTN_DYN>()) {
            // Even if the attention is detected and ready to go pyramid,
            // force it on the full range
            m_pyramid_selector.reset(new runtime::pyramid_attention::All(pyramid_count));
        } else {
            m_pyramid_selector = runtime::pyramid_attention::PositionIDs::find(pyramid_dyn, *this);
            if (!m_pyramid_selector) {
                LOG_WARN("Pyramid dynamic capability is enabled, but no run-time features were found.");
                // Create All selector with the number of pyramid models
                m_pyramid_selector.reset(new runtime::pyramid_attention::All(pyramid_count));
            }
        }
    }

    if (has_hfa) {
        const auto& hfa_desc = m_npuw_model->m_compiled_submodels.at(hfa_sub_idx).host_flash_attention.value();
        const size_t query_size = hfa_desc._sdpa_attention_info._query_size;
        m_hfa_selector = runtime::host_flash_attention::PositionIDs::find(query_size, *this);
        if (!m_hfa_selector) {
            // HFA requires PositionIDs selector - cannot fallback to 'All' like Dynamic/Pyramid
            // because HFA uses tile-based execution without a full-range infer request
            OPENVINO_THROW("HFA dynamic capability is enabled, but no run-time features were found.");
        }
        LOG_VERB("Done");
    }
}

void ov::npuw::JustInferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                            const ov::SoPtr<ov::ITensor>& tensor) {
    m_port_to_tensor[port] = TensorStorage{tensor, true};

    // Check if setting output tensor
    for (std::size_t i = 0; i < m_npuw_model->outputs().size(); ++i) {
        if (m_npuw_model->outputs()[i] == port) {
            const auto& from_submodel = m_npuw_model->m_outputs_to_submodels_outputs.at(i);
            auto funcall_result_iter = m_funcall_result.find(from_submodel);
            // This is a tricky case:
            // 1) We already allocated an output tensor in m_funcall_result via FMM
            // 2) We got an output tensor from outside
            // m_funcall_result and m_port_to_tensor aren't connected, thus we will only write
            // to m_funcall_result, but get_tensor() would return an empty tensor from m_port_to_tensor.
            // Here we have to set the tensor to function's output, so the function will write to the correct tensor.
            if (funcall_result_iter != m_funcall_result.end()) {
                funcall_result_iter->second = tensor;
            }
        }
    }

    // Process setting input tensor
    handle_set_remote_input(port, tensor);
}

ov::npuw::TensorPtr ov::npuw::JustInferRequest::alloc_global_out(std::size_t out_idx) const {
    const auto& from_submodel = m_npuw_model->m_outputs_to_submodels_outputs.at(out_idx);
    auto funcall_result_iter = m_funcall_result.find(from_submodel);
    if (funcall_result_iter != m_funcall_result.end()) {
        return funcall_result_iter->second;
    }
    return IBaseInferRequest::alloc_global_out(out_idx);
}

void ov::npuw::JustInferRequest::connect_subrequests() {
    LOG_INFO("Connecting subrequests...");
    LOG_BLOCK();
    auto& subm = m_npuw_model->m_compiled_submodels;
    auto& subreqs = m_subrequests;
    for (const auto& kvp : m_npuw_model->m_submodels_input_to_prev_output) {
        const auto& subm_idx_to = kvp.first.first;
        const auto& port_idx_to = kvp.first.second;
        const auto& subm_idx_from = kvp.second.first;
        const auto& port_idx_from = kvp.second.second;

        LOG_DEBUG("Subgraph[" << subm_idx_from << "]/" << port_idx_from << " --> " << "Subgraph[" << subm_idx_to << "]/"
                              << port_idx_to);
        LOG_BLOCK();

        if (subm[subm_idx_from].replaced_by && subm[subm_idx_to].replaced_by) {
            // A function call to function call connection:
            // - Skip it here, setting in/out tensors will be handled in runtime
            LOG_DEBUG("Skip: both are function calls");
            continue;
        } else if (subm[subm_idx_from].replaced_by && !subm[subm_idx_to].replaced_by) {
            // A function call to normal subgraph connection:
            // - Take a tensor from the storage & assign it to the reader
            const auto& iport = m_subrequests[subm_idx_to]->get_compiled_model()->inputs()[port_idx_to];
            const auto& tensor = m_funcall_result.at(LinkFrom{subm_idx_from, port_idx_from});
            subreqs[subm_idx_to]->set_tensor(iport, tensor);
            LOG_DEBUG("Set Subgraph[" << subm_idx_to << "]/" << iport << " to internal tensor");
        } else if (!subm[subm_idx_from].replaced_by && subm[subm_idx_to].replaced_by) {
            LOG_DEBUG("Skip: reader is a function call");
            continue;
        } else if (!subreqs[subm_idx_from] && subreqs[subm_idx_to]) {
            // Subrequests may be optimized out, but in this case there should be
            // no connection between them and their consumers in the input_to_prev
            // map (links are erased & Parameters are replaced with Const) at earlier
            // stages.
            OPENVINO_THROW("FATAL: \"Prev. Output\" Request ",
                           subm_idx_from,
                           " in input_to_prev_output mapping was optimized out,"
                           " but it consumer request ",
                           subm_idx_to,
                           " wasn't!");
        } else if (!subreqs[subm_idx_to]) {
            // FIXME: Links like this probbaly shouldn't exist in the map, too.
            // Need to research why this is not THROW
            LOG_WARN("\"Input\" Request in input_to_prev_output mapping was optimized out");
            continue;
        } else {
            // Final case:
            NPUW_ASSERT(!subm[subm_idx_from].replaced_by);  // prod subgraph isn't a funcall
            NPUW_ASSERT(!subm[subm_idx_to].replaced_by);    // cons subgraph isn't a funcall
            NPUW_ASSERT(subreqs[subm_idx_from]);            // prod request is created
            NPUW_ASSERT(subreqs[subm_idx_to]);              // cons request is created

            // Just set one's output tensor to another's input
            const auto& iport = subreqs[subm_idx_to]->get_compiled_model()->inputs()[port_idx_to];
            const auto& oport = subreqs[subm_idx_from]->get_compiled_model()->outputs()[port_idx_from];
            const auto& tensor = subreqs[subm_idx_from]->get_tensor(oport);
            LOG_DEBUG("Set Subgraph[" << subm_idx_to << "]/" << iport << " to Subgraph[" << subm_idx_from << "]/"
                                      << oport);
            subreqs[subm_idx_to]->set_tensor(iport, tensor);
        }
    }  // for(map)
    LOG_INFO("Done");
}

void ov::npuw::JustInferRequest::prepare_for_infer() {
    LOG_DEBUG("Preparing to infer...");
    LOG_BLOCK();

    if (m_pyramid_selector) {
        m_pyramid_selector->prepare(get_history_size());

        // Get the pyramid model ID based on current sequence length (updated in prepare())
        auto pyramid_id = m_pyramid_selector->pyramid_id();

        for (auto&& id : m_funcall_heads) {
            auto& comp_model_desc = m_npuw_model->m_compiled_submodels[id];
            if (comp_model_desc.pyramid_attention.has_value()) {
                m_subrequests[id] = comp_model_desc.pyramid_infer_requests[pyramid_id];
                if (is_pipelined(id)) {
                    m_funcall_pipeline[id].subrequest = comp_model_desc.pyramid_pipeline_requests[pyramid_id];
                }
            }
        }
    }

    // Submit global parameters (if needed) for the first subgraph
    bind_global_parameters(next(0));

    // If funcall pipelining is enabled, prefill the function "heads"
    // with constant arguments. The list of heads is empty otherwise.
    for (auto&& id : m_funcall_heads) {
        LOG_DEBUG("Pre-initializing weights for subgraph[" << id << "]");
        unpack_closure(id, m_subrequests[id]);
    }

    // Adjust spatial input range, if supported
    if (m_spatial_selector) {
        m_spatial_selector->prepare();
    }

    // So do the dynamic range
    if (m_attention_selector) {
        m_attention_selector->prepare(get_history_size());
    }

    // HFA selector
    if (m_hfa_selector) {
        m_hfa_selector->prepare(get_history_size());
        if (m_hfa_runtime_ctx) {
            m_hfa_runtime_ctx->clear_mask_cache();
        }
    }

    // FIXME: attention-specific, needs to be moved out after refactoring
    m_cached_attention_mask = {};

    LOG_DEBUG("Done");
}

ov::npuw::IBaseInferRequest::RqPtr ov::npuw::JustInferRequest::get_real_subrequest(std::size_t idx) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    const auto real_idx = comp_model_desc.replaced_by.value_or(idx);
    return m_subrequests[real_idx];
}

bool ov::npuw::JustInferRequest::valid_subrequest(std::size_t idx) const {
    auto* ncthis = const_cast<ov::npuw::JustInferRequest*>(this);
    return ncthis->get_real_subrequest(idx) != nullptr;
}

void ov::npuw::JustInferRequest::start_subrequest(std::size_t idx) {
    m_subrequests[idx]->start_async();
}

void ov::npuw::JustInferRequest::bind_global_parameters(std::size_t idx) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    const auto real_idx = comp_model_desc.replaced_by.value_or(idx);

    // pick which subrequest we actually work on here
    if (now_idx() && real_idx == real(now_idx().value()) && is_pipelined(now_idx().value())) {
        LOG_DEBUG("Accessing the pipeline subrequest");
        // The real index of request we need to prepare IS
        // the same request which executes now AND
        // function_pipelining enabled - select the reserve request.
        NPUW_ASSERT(m_funcall_pipeline[real_idx].subrequest);
        bind_global_params(idx, m_funcall_pipeline[real_idx].subrequest);
    } else {
        // Otherwise: Just a return a subrequest which is in place.
        // If it is a function call and we have function pipelining ON,
        // it is still the right subrequest we can use.
        LOG_DEBUG("Accessing the primary subrequest");
        bind_global_params(idx, m_subrequests[real_idx]);
    }
}

void ov::npuw::JustInferRequest::bind_global_results(std::size_t idx) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    if (comp_model_desc.replaced_by) {
        // Don't do here - function call will take the right tensor
        // itself. Note it may be implemented more efficently than now
        // (and in some cases, the tensor can be pre-set)
        LOG_DEBUG("Skipping bind_glo - function will do it for itself");
        return;
    }
    IBaseInferRequest::bind_global_results(idx, m_subrequests[idx]);
}

void ov::npuw::JustInferRequest::function_prologue(std::size_t idx) {
    LOG_DEBUG("Preparing Subgraph[" << idx << "] funcall prologue");
    LOG_BLOCK();

    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];

    NPUW_ASSERT(comp_model_desc.replaced_by);
    const auto real_idx = comp_model_desc.replaced_by.value();
    auto& func_desc = m_npuw_model->m_compiled_submodels[real_idx];

    const bool is_spatial = func_desc.spatial.has_value();
    const bool is_dynamic = func_desc.attention.has_value();
    const bool is_pyramid = func_desc.pyramid_attention.has_value();
    const bool is_hfa = func_desc.host_flash_attention.has_value();

    // Generalized: check if input is neither param nor mask
    auto is_non_param_mask = [](const auto& info, std::size_t in_idx) {
        const bool not_param = std::none_of(info.params.begin(), info.params.end(), [&](auto&& p) {
            return p.idx == in_idx;
        });
        const bool not_mask = in_idx != info.mask_idx;
        return not_param && not_mask;
    };

    // Function call prologue:
    // 1. Walk through function dependencies and set the respective tensors
    //    as parameters
    for (size_t i = 0; i < func_desc.param_base; i++) {
        LOG_DEBUG("Binding parameter[" << i << "]...");
        LOG_BLOCK();
        const auto& iport = func_desc.compiled_model->inputs()[i];

        auto link_iter = m_npuw_model->m_submodels_input_to_prev_output.find({idx, i});
        if (link_iter != m_npuw_model->m_submodels_input_to_prev_output.end()) {
            std::size_t prod_idx;
            std::size_t prod_port;
            std::tie(prod_idx, prod_port) = link_iter->second;

            const auto& i_tensor = [&]() {
                if (!m_npuw_model->m_compiled_submodels[prod_idx].replaced_by) {
                    // Producer is a normal model -> take its tensor directly
                    const auto& oport =
                        m_npuw_model->m_compiled_submodels[prod_idx].compiled_model->outputs()[prod_port];
                    return m_subrequests[prod_idx]->get_tensor(oport);
                } else {
                    // Producer is a function - maybe the same as we're calling now.
                    // Take its tensor from the storage
                    return m_funcall_result.at({prod_idx, prod_port});
                }
            }();

            if (is_spatial) {
                // Spatial case - defer
                m_spatial_io[real_idx].inputs.at(i) = i_tensor;
            } else if (is_dynamic) {
                // Set tensor only if it is non-dynamic (dynamic are managed by the infer_dynamic)
                if (is_non_param_mask(*func_desc.attention, i)) {
                    m_subrequests[real_idx]->set_tensor(iport, i_tensor);
                } else {
                    m_attention_io[idx].inputs.at(i) = i_tensor;
                }
            } else if (is_pyramid) {
                // Pyramid attention
                auto pyramid_id = m_pyramid_selector->pyramid_id();
                const auto& info = func_desc.pyramid_attention.value()._attention_infos[pyramid_id];
                if (is_non_param_mask(info, i)) {
                    m_subrequests[real_idx]->set_tensor(iport, i_tensor);
                } else {
                    m_attention_io[idx].inputs.at(i) = i_tensor;
                }
            } else if (is_hfa) {
                // Host Flash Attention case - defer, use dedicated HFA I/O structure
                m_hfa_io[idx].inputs.at(i) = i_tensor;
            } else {
                // Default case
                m_subrequests[real_idx]->set_tensor(iport, i_tensor);
            }
        }  // if (link_iter)
    }  // for(param_base)

    // 1.5: Do attention prologue if needed
    if (is_dynamic) {
        m_profile["attn(act)"] += ov::npuw::perf::ms_to_run([&]() {
            function_prologue_attn(real_idx, idx);
        });
    }

    if (is_pyramid) {
        m_profile["attn(act)"] += ov::npuw::perf::ms_to_run([&]() {
            function_prologue_pyramid_attn(real_idx, idx);
        });
    }

    // 2. Unpack the function closure -- right here, if pipelining if not enabled.
    // If it is enabled, the flow is a little bit different - see run_subrequest_for_success()
    // for details.
    if (!is_pipelined(idx) && m_closure_update_required && !func_desc.forced_to_fcall) {
        LOG_DEBUG("Unpacking closures...");
        LOG_BLOCK();
        unpack_closure(idx, m_subrequests[real_idx]);
    }

    // 3. Tell the function which results to produce (this time).
    // Note it covers both internal tensors used by other subgraphs as well as
    // the Result tensors for the entire network.
    // ..Since the tensors allocated for outputs of the networks ARE taken from the
    // "funcall_results" if those are produced by funcall results.
    for (std::size_t i = 0; i < func_desc.compiled_model->outputs().size(); i++) {
        LOG_DEBUG("Binding result[" << i << "]...");
        auto& oport = func_desc.compiled_model->outputs()[i];
        auto o_tensor = m_funcall_result.at({idx, i});
        if (is_hfa) {
            // HFA case - defer, store in dedicated HFA I/O structure
            m_hfa_io[idx].outputs.at(i) = o_tensor;
        } else if (!is_spatial) {
            // Non-spatial case - set immediately
            m_subrequests[real_idx]->set_tensor(oport, o_tensor);
        } else {
            // Spatial case - defer
            m_spatial_io[real_idx].outputs.at(i) = o_tensor;
        }
    }
    LOG_DEBUG("Done");
}

void ov::npuw::JustInferRequest::function_prologue_attn(std::size_t real_idx, std::size_t idx) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
    NPUW_ASSERT(comp_model_desc.attention.has_value());

    auto& r = m_subrequests[real_idx];

    const auto& dynamic = comp_model_desc.attention.value();
    auto mask_iport = comp_model_desc.compiled_model->inputs()[dynamic.mask_idx];

    const auto& graph_mask = m_attention_io[idx].inputs.at(dynamic.mask_idx);
    const auto this_case = m_attention_selector->this_case();
    auto pos_id = m_attention_selector->length();

    if (pos_id == -1) {
        // Dynamic range couldn't be identified - fallback to the default
        // (worst case) behavior
        r->set_tensor(mask_iport, graph_mask);
    } else {
        const auto past_len = m_attention_selector->past_length();
        const auto present_len = dynamic.query_size;
        // FIXME: get the right dim
        const uint32_t kv_dim = 3;

        auto set_or_copy = [&](const auto& view) {
            if (!needs_copy(idx)) {
                r->set_tensor(mask_iport, view);
            } else {
                const auto& dst = r->get_tensor(mask_iport);
                dst->set_shape(view->get_shape());
                view->copy_to(dst._ptr);
            }
        };

        // Now set the mask. Here comes very strong chunking & SDPA knowledge again
        using namespace ov::npuw::runtime;
        if (this_case == attention::Selector::Case::GENERATE) {
            // Take a view from our "attend_all" mask
            const auto& view = ov::npuw::util::view(ov::get_tensor_impl(dynamic.attend_all), kv_dim, 0, past_len + 1);
            set_or_copy(view);
        } else if (this_case == attention::Selector::Case::PREFILL) {
            // Use our in-graph synthesized mask
            if (m_cached_attention_mask) {
                // All sub models are sharing the same attention mask, we can use the cached attention
                // mask directly to avoid redundant tensor copy
                m_subrequests[real_idx]->set_tensor(mask_iport, m_cached_attention_mask);
                return;
            }

            // Handle attention mask concatenation for SDPA:
            // The attention mask is composed with 2 parts:
            // The 1st part is for the "present", which is at the tail: starting from past_len to context_len
            // The 2nd part is for the "past", which is at the beginning: starting from 0 to past_len
            auto full_mask_shape = graph_mask->get_shape();
            auto actual_mask_shape = full_mask_shape;
            actual_mask_shape[kv_dim] = present_len + past_len;

            // Reshape the input to the proper shape
            const auto& dst = r->get_tensor(mask_iport);
            dst->set_shape(actual_mask_shape);

            // Copy "present" attention mask
            const auto& present_dst_view = ov::npuw::util::view(dst, kv_dim, past_len, present_len);
            const auto& present_src_view =
                ov::npuw::util::view(graph_mask, kv_dim, full_mask_shape[kv_dim] - present_len, present_len);
            present_src_view->copy_to(present_dst_view._ptr);

            // Copy "past" attention mask
            if (past_len > 0) {
                const auto& past_dst_view = ov::npuw::util::view(dst, kv_dim, 0, past_len);
                const auto& past_src_view = ov::npuw::util::view(graph_mask, kv_dim, 0, past_len);
                past_src_view->copy_to(past_dst_view._ptr);
            }
            m_cached_attention_mask = dst;
        } else {
            NPUW_ASSERT(false && "Reached the unreachable code");
        }
    }
}

void ov::npuw::JustInferRequest::function_prologue_pyramid_attn(std::size_t real_idx, std::size_t idx) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
    NPUW_ASSERT(comp_model_desc.pyramid_attention.has_value());

    auto& r = m_subrequests[real_idx];
    auto pyramid_id = m_pyramid_selector->pyramid_id();

    const auto& dynamic = comp_model_desc.pyramid_attention.value()._attention_infos[pyramid_id];
    auto mask_iport =
        comp_model_desc.pyramid_attention.value()._compiled_models[pyramid_id]->inputs()[dynamic.mask_idx];

    const auto& graph_mask = m_attention_io[idx].inputs.at(dynamic.mask_idx);
    const auto this_case = m_pyramid_selector->this_case();
    const auto present_len = dynamic.query_size;

    // FIXME: get the right dim
    const uint32_t kv_dim = 3;

    // Get destination tensor once for all code paths
    const auto& dst = r->get_tensor(mask_iport);

    // Lambda: copy attention mask segment from source to destination
    auto copy_mask_segment = [&](std::size_t dst_offset, std::size_t src_offset, std::size_t length) {
        if (length == 0) {
            return;
        }
        const auto& dst_view = ov::npuw::util::view(dst, kv_dim, dst_offset, length);
        const auto& src_view = ov::npuw::util::view(graph_mask, kv_dim, src_offset, length);
        ov::npuw::util::copy_tensor_by_dim(src_view, dst_view, kv_dim, kv_dim);
    };

    auto pos_id = m_pyramid_selector->length();
    if (pos_id == -1) {
        // Pyramid dynamic range couldn't be identified - fallback to default behavior
        r->set_tensor(mask_iport, graph_mask);
        return;
    }

    // Pyramid dynamic range identified
    const auto past_len = m_pyramid_selector->past_length();

    // Early return: reuse cached attention mask if available
    if (m_cached_attention_mask) {
        // All sub models are sharing the same attention mask, we can use the cached attention
        // mask directly to avoid redundant tensor copy
        m_subrequests[real_idx]->set_tensor(mask_iport, m_cached_attention_mask);
        return;
    }

    // Now set the mask. Here comes very strong chunking & SDPA knowledge again
    using namespace ov::npuw::runtime;
    const auto full_mask_shape = graph_mask->get_shape();

    if (this_case == pyramid_attention::Selector::Case::GENERATE) {
        const auto dst_shape = dst->get_shape();

        // Optimization: if shapes match, use the full mask directly
        if (dst_shape == full_mask_shape) {
            r->set_tensor(mask_iport, graph_mask);
            m_cached_attention_mask = graph_mask;
            return;
        }

        // FIXME: No need to copy whole attention mask, just mark the new tokens to valid

        std::size_t dst_present_offset = dst_shape[kv_dim] - present_len;

        // Copy present mask: tail of source -> tail of destination
        copy_mask_segment(dst_present_offset, full_mask_shape[kv_dim] - present_len, present_len);

        // Copy past mask: head of source [0, dst_present_offset) -> head of destination
        copy_mask_segment(0, 0, dst_present_offset);

        m_cached_attention_mask = dst;
    } else if (this_case == pyramid_attention::Selector::Case::PREFILL) {
        // Copy present mask: tail of source -> [past_len, past_len + present_len)
        copy_mask_segment(past_len, full_mask_shape[kv_dim] - present_len, present_len);

        // Copy past mask: head of source -> [0, past_len)
        copy_mask_segment(0, 0, past_len);

        m_cached_attention_mask = dst;
    } else {
        NPUW_ASSERT(false && "Unsupported pyramid attention case");
    }
}

void ov::npuw::JustInferRequest::recreate_subrequests(std::size_t idx) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    auto real_idx = comp_model_desc.replaced_by.value_or(idx);

    const auto is_piped = is_pipelined(idx);
    auto new_rqs = create_infer_requests(idx, is_piped ? 2 : 1);

    // NB: Regardless if this subrequest was a function call
    // or not, always use the real_idx here - for regular
    // subrequests, real_id == idx, but for function calls it
    // is critical here to update the function body, not the
    // function calls (which are left empty now in the vector)
    m_subrequests[real_idx] = new_rqs.at(0);
    if (is_piped) {
        m_funcall_pipeline[real_idx].subrequest = new_rqs.at(1);
    }

    // Recreate pyramid infer requests if this function has pyramid attention
    if (comp_model_desc.replaced_by) {
        auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
        if (proto_comp_model_desc.pyramid_attention) {
            setup_pyramid_infer_requests(real_idx, is_piped, true);
        }
        // Recreate HFA tile infer requests if this function has host flash attention
        if (proto_comp_model_desc.host_flash_attention) {
            setup_hfa_infer_requests(real_idx, is_piped, /* is_recreate */ true, /* enable_hfa_optimizations */ true);
        }
    }

    // After an infer request is recreated, the internal cross-request
    // connections should be re-established (in/out tensors reset properly)
    // Note: these two proceduers do the full I/O reset procedure what's
    // overkill - only affected subrequest(s) could be updated instead,
    // but it is a more complex thing and can be implemented separately
    connect_subrequests();
    m_subrequest_devices[idx] = *comp_model_desc.device_it;
}

void ov::npuw::JustInferRequest::setup_pyramid_infer_requests(std::size_t real_idx, bool is_piped, bool is_recreate) {
    auto& submodel_desc = m_npuw_model->m_compiled_submodels[real_idx];
    if (!submodel_desc.pyramid_attention.has_value()) {
        return;
    }

    LOG_INFO((is_recreate ? "Recreating" : "Creating") << " pyramid infer requests...");
    LOG_BLOCK();

    const auto& pyramid_models = submodel_desc.pyramid_attention.value()._compiled_models;
    const size_t num_pyramid_models = pyramid_models.size();

    // Clear existing requests if recreating
    if (is_recreate) {
        submodel_desc.pyramid_infer_requests.clear();
        submodel_desc.pyramid_pipeline_requests.clear();
    }

    // Allocate storage for infer requests
    submodel_desc.pyramid_infer_requests.resize(num_pyramid_models);
    if (is_piped) {
        submodel_desc.pyramid_pipeline_requests.resize(num_pyramid_models);
    }

    // Create infer requests for all but the last pyramid model
    for (size_t model_idx = 0; model_idx + 1 < num_pyramid_models; ++model_idx) {
        try {
            // Create main infer request
            submodel_desc.pyramid_infer_requests[model_idx] = pyramid_models[model_idx]->create_infer_request();
            // Create pipeline infer request if pipelined
            if (is_piped) {
                submodel_desc.pyramid_pipeline_requests[model_idx] = pyramid_models[model_idx]->create_infer_request();
            }
        } catch (const std::exception& ex) {
            LOG_ERROR("Failed to " << (is_recreate ? "recreate" : "create") << " infer request for pyramid model["
                                   << model_idx << "]: " << ex.what());
            NPUW_ASSERT(false && "Pyramid model infer request creation/recreation failed");
        } catch (...) {
            LOG_ERROR("Failed to " << (is_recreate ? "recreate" : "create") << " infer request for pyramid model["
                                   << model_idx << "]: Unknown error");
            NPUW_ASSERT(false && "Pyramid model infer request creation/recreation failed with unknown error");
        }

        // Share input tensors between pyramid and main infer requests
        const size_t num_inputs = pyramid_models[model_idx]->inputs().size();
        NPUW_ASSERT(num_inputs == submodel_desc.compiled_model->inputs().size());
        for (size_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
            auto pyramid_input = pyramid_models[model_idx]->inputs()[input_idx];
            auto main_input = submodel_desc.compiled_model->inputs()[input_idx];

            // Get tensor from main infer request and share its memory with the pyramid infer request
            auto main_tensor_ptr = m_subrequests[real_idx]->get_tensor(main_input)->data();
            auto pyramid_tensor = submodel_desc.pyramid_infer_requests[model_idx]->get_tensor(pyramid_input);
            auto shared_tensor = ov::get_tensor_impl(
                ov::Tensor(pyramid_tensor->get_element_type(), pyramid_tensor->get_shape(), main_tensor_ptr));
            submodel_desc.pyramid_infer_requests[model_idx]->set_tensor(pyramid_input, shared_tensor);

            // Repeat for pipeline infer request if pipelined
            if (is_piped) {
                auto pipeline_tensor = submodel_desc.pyramid_pipeline_requests[model_idx]->get_tensor(pyramid_input);
                auto pipeline_tensor_ptr = m_funcall_pipeline[real_idx].subrequest->get_tensor(main_input)->data();
                auto shared_pipeline_tensor = ov::get_tensor_impl(
                    ov::Tensor(pipeline_tensor->get_element_type(), pipeline_tensor->get_shape(), pipeline_tensor_ptr));
                submodel_desc.pyramid_pipeline_requests[model_idx]->set_tensor(pyramid_input, shared_pipeline_tensor);
            }
        }
    }

    // For the last pyramid model, reuse the original model's infer requests
    if (num_pyramid_models > 0) {
        const size_t last_model_idx = num_pyramid_models - 1;
        LOG_INFO("Reusing " << (is_recreate ? "recreated " : "") << "original infer requests for last pyramid model["
                            << last_model_idx << "]");
        submodel_desc.pyramid_infer_requests[last_model_idx] = m_subrequests[real_idx];
        if (is_piped) {
            submodel_desc.pyramid_pipeline_requests[last_model_idx] = m_funcall_pipeline[real_idx].subrequest;
        }
    }

    if (!is_recreate && num_pyramid_models > 0) {
        LOG_INFO("Successfully created " << (num_pyramid_models - 1)
                                         << " new pyramid infer requests and reused 1 original request");
    }
}

void ov::npuw::JustInferRequest::setup_hfa_infer_requests(std::size_t real_idx,
                                                          bool is_piped,
                                                          bool is_recreate,
                                                          bool enable_hfa_optimizations) {
    auto& submodel_desc = m_npuw_model->m_compiled_submodels[real_idx];
    if (!submodel_desc.host_flash_attention.has_value()) {
        return;
    }

    LOG_INFO((is_recreate ? "Recreating" : "Creating") << " HFA tile infer requests...");
    LOG_BLOCK();

    const auto& hfa = submodel_desc.host_flash_attention.value();

    // Clear existing requests if recreating
    if (is_recreate) {
        submodel_desc.hfa_infer_requests.clear();
        submodel_desc.hfa_pipeline_requests.clear();
    }

    // Allocate storage for infer requests: [REGULAR_TILE] and [FINAL_TILE]
    submodel_desc.hfa_infer_requests.resize(CompiledModel::CompiledModelDesc::HFATileIdx::COUNT);
    if (is_piped) {
        submodel_desc.hfa_pipeline_requests.resize(CompiledModel::CompiledModelDesc::HFATileIdx::COUNT);
    }

    // Create infer request for regular tile model
    try {
        LOG_INFO("Creating infer request for HFA regular tile model...");
        submodel_desc.hfa_infer_requests[CompiledModel::CompiledModelDesc::HFATileIdx::REGULAR_TILE] =
            hfa._compiled_tile_model->create_infer_request();
        if (is_piped) {
            submodel_desc.hfa_pipeline_requests[CompiledModel::CompiledModelDesc::HFATileIdx::REGULAR_TILE] =
                hfa._compiled_tile_model->create_infer_request();
        }
    } catch (const std::exception& ex) {
        LOG_ERROR("Failed to " << (is_recreate ? "recreate" : "create")
                               << " infer request for HFA regular tile model: " << ex.what());
        OPENVINO_THROW("HFA regular tile model infer request creation failed: ", ex.what());
    } catch (...) {
        LOG_ERROR("Failed to " << (is_recreate ? "recreate" : "create")
                               << " infer request for HFA regular tile model: Unknown error");
        OPENVINO_THROW("HFA regular tile model infer request creation failed with unknown error");
    }

    // For final tile model, reuse the main compiled_model's infer request
    // because compiled_model points to _compiled_final_tile_model for HFA
    LOG_INFO("Reusing " << (is_recreate ? "recreated " : "") << "main infer request for HFA final tile model");
    submodel_desc.hfa_infer_requests[CompiledModel::CompiledModelDesc::HFATileIdx::FINAL_TILE] =
        m_subrequests[real_idx];
    if (is_piped) {
        submodel_desc.hfa_pipeline_requests[CompiledModel::CompiledModelDesc::HFATileIdx::FINAL_TILE] =
            m_funcall_pipeline[real_idx].subrequest;
    }

    // Share input tensors between HFA tile models and main infer request
    // Note: Both tile models have the same input structure
    const size_t num_inputs = hfa._compiled_tile_model->inputs().size();
    for (size_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
        auto tile_input = hfa._compiled_tile_model->inputs()[input_idx];
        auto final_tile_input = hfa._compiled_final_tile_model->inputs()[input_idx];

        // Directly share tensor from main infer request to regular tile request
        auto main_tensor = m_subrequests[real_idx]->get_tensor(final_tile_input);
        submodel_desc.hfa_infer_requests[CompiledModel::CompiledModelDesc::HFATileIdx::REGULAR_TILE]->set_tensor(
            tile_input,
            main_tensor);

        // Repeat for pipeline infer request if pipelined
        if (is_piped) {
            auto pipeline_tensor = m_funcall_pipeline[real_idx].subrequest->get_tensor(final_tile_input);
            submodel_desc.hfa_pipeline_requests[CompiledModel::CompiledModelDesc::HFATileIdx::REGULAR_TILE]->set_tensor(
                tile_input,
                pipeline_tensor);
        }
    }

    LOG_INFO("Successfully " << (is_recreate ? "recreated" : "created")
                             << " HFA tile infer requests with shared input tensors");

    // Initialize HFA optimizations (mask cache + state double-buffering) if enabled
    if (enable_hfa_optimizations) {
        LOG_INFO("HFA optimizations are ENABLED (mask cache + state double-buffering)");

        // Initialize runtime context if needed
        if (!m_hfa_runtime_ctx) {
            m_hfa_runtime_ctx.emplace();
        }

        if (is_recreate) {
            m_hfa_runtime_ctx->reset();
        }

        LOG_INFO("Pre-allocating HFA mask tile buffers...");

        // Initialize pre-allocated buffers
        m_hfa_runtime_ctx->initialize_mask_cache(
            hfa,
            *submodel_desc.device_it,
            [this](const ov::element::Type& dtype, const ov::Shape& shape, const std::string& device) {
                return allocMem(dtype, shape, device);
            });

        LOG_INFO("Pre-allocated " << m_hfa_runtime_ctx->num_mask_tile_buffers() << " mask tile buffer(s)");

        // Initialize state buffers and double-buffering for first inference
        LOG_INFO("Initializing HFA state tensors and double-buffering...");

        // Get pre-cached indices
        const auto& tile_in = hfa._sdpa_attention_info._tile_input_indices;

        // Get state tensors from regular tile request
        auto state_acc =
            submodel_desc.hfa_infer_requests[CompiledModel::CompiledModelDesc::HFATileIdx::REGULAR_TILE]->get_tensor(
                hfa._compiled_tile_model->inputs()[tile_in.acc]);
        auto state_max =
            submodel_desc.hfa_infer_requests[CompiledModel::CompiledModelDesc::HFATileIdx::REGULAR_TILE]->get_tensor(
                hfa._compiled_tile_model->inputs()[tile_in.max]);
        auto state_sum =
            submodel_desc.hfa_infer_requests[CompiledModel::CompiledModelDesc::HFATileIdx::REGULAR_TILE]->get_tensor(
                hfa._compiled_tile_model->inputs()[tile_in.d]);

        // Initialize state tensors with zeros/minus infinity
        runtime::host_flash_attention::HFARuntimeContext::initialize_state_tensors(state_acc, state_max, state_sum);

        // Setup double-buffering with initialized state
        runtime::host_flash_attention::HFARuntimeContext::StateBuffers initial_buffers{state_acc, state_max, state_sum};

        m_hfa_runtime_ctx->initialize_state_buffers(
            initial_buffers,
            hfa,
            *submodel_desc.device_it,
            [this](const ov::element::Type& dtype, const ov::Shape& shape, const std::string& device) {
                return allocMem(dtype, shape, device);
            });

        LOG_INFO("HFA state tensors and double-buffering initialized successfully");
    } else {
        LOG_INFO("HFA optimizations are DISABLED - will extract mask tiles on-the-fly without state caching");

        // Clear runtime context if it exists
        if (m_hfa_runtime_ctx) {
            m_hfa_runtime_ctx.reset();
        }
    }
}

void ov::npuw::JustInferRequest::run_subrequest_for_success(std::size_t idx, bool& failover) {
    failover = false;
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    auto real_idx = comp_model_desc.replaced_by.value_or(idx);

    // Infer is also fail-safe...
    bool job_done = false;
    bool dump_in = false;
    bool next_prepared = false;
    while (!job_done) {
        bool should_recreate = false;
        if (m_subrequest_devices[real_idx] != *m_npuw_model->m_compiled_submodels[real_idx].device_it) {
            // This may happen when there's multiple NPUW's infer
            // requests created and some failure occurs in one of
            // those before another reaches this point.
            LOG_INFO("Recreating subrequest[" << real_idx << "] because model was recompiled for "
                                              << *m_npuw_model->m_compiled_submodels[real_idx].device_it << " device.");
            recreate_subrequests(real_idx);
        }

        // Feeding the global Parameters is now part of the common
        // execution pipeline: See how it is done in
        // `unsafe_run_this_prep_next()`.  Now we only need to bind
        // the subrequest' outputs to global Results, if relevant.
        bind_global_results(idx);

        if (comp_model_desc.replaced_by) {
            function_prologue(idx);
        }
        if (!dump_in) {
            dump_in = true;
            dump_input_tensors(idx);
        }

        std::string error_text;
        try {
            LOG_DEBUG("Trying to run subrequest[" << idx << "]...");
            LOG_BLOCK();
            unsafe_run_this_prep_next(idx, next_prepared);
            job_done = true;
            LOG_DEBUG("Done: " << idx << "(exec subrequest)");
        } catch (const std::exception& ex) {
            error_text = ex.what();
            LOG_ERROR("Subgraph [" << idx << "] - FAILED to run infer request:" << std::endl
                                   << error_text << std::endl);
            should_recreate = true;
        } catch (...) {
            LOG_ERROR("Subgraph [" << idx << "] - FAILED to run infer request: REASON UNKNOWN");
            should_recreate = true;
        }
        if (should_recreate) {
            // Altering iterators here!! Contracts should be changed!
            comp_model_desc.device_it++;

            // Check if failover is actually possible
            if ((m_npuw_model->m_dev_list.cend() == comp_model_desc.device_it) ||
                !m_npuw_model->m_cfg.get<::intel_npu::NPUW_FALLBACK_EXEC>()) {
                OPENVINO_THROW("Execution error: \"", error_text, "\" - no fallback possible");
            }

            failover = true;
            LOG_INFO("- Trying next device...");
            if (!m_npuw_model->compile_for_success(real_idx)) {
                OPENVINO_THROW("Execution error: \"", error_text, "\" - failed to recompile the model in runtime");
            }
            recreate_subrequests(idx);
        }
    }  // while(job_done)

    if (job_done) {
        dump_output_tensors(idx);  // FIXME: Called here unconditionally, need to refactor
        if (is_pipelined(idx) && m_funcall_pipeline[idx].next) {
            // Swap the next (pipelined, semi-prepared) infer request in the chain
            // with the default (to be accessed next) one.
            std::swap(m_subrequests[real_idx], m_funcall_pipeline[real_idx].subrequest);
        }
    }
}

void ov::npuw::JustInferRequest::unsafe_during(std::size_t real_idx, std::size_t idx, const std::function<void()>& f) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];

    if (!comp_model_desc.spatial && !comp_model_desc.host_flash_attention.has_value()) {
        // Normal: trigger request asynchronously, run `f` in this context
        // FIXME: dynamic could hit here too, but it has special logic
        // around execution which makes it harder to run than a plain start_async()
        auto& r = m_subrequests[real_idx];
        r->start_async();
        f();  // expect noexcept
        r->wait();
    } else {
        // Spatial or HFA: Run f asynchronously while executing spatial/HFA inference
        auto future = std::async(std::launch::async, f);
        unsafe_infer(real_idx, idx);
        future.wait();
    }
}

// ====================================================================================================
// Host Flash Attention (HFA) Helper Functions
// ====================================================================================================

// Extract tile slice with optional type conversion
void ov::npuw::JustInferRequest::hfa_extract_and_copy_tile(const ov::SoPtr<ov::ITensor>& source_tensor,
                                                           const ov::SoPtr<ov::ITensor>& dest_tensor,
                                                           uint32_t sequence_dim,
                                                           int64_t sequence_offset,
                                                           int64_t sequence_length,
                                                           const std::string& tensor_name) {
    if (!dest_tensor->is_continuous()) {
        OPENVINO_THROW("HFA tile extraction error: destination tensor for '",
                       tensor_name,
                       "' is not continuous - cannot perform direct copy");
    }

    auto source_view = ov::npuw::util::view(source_tensor, sequence_dim, sequence_offset, sequence_length);
    const auto dest_type = dest_tensor->get_element_type();
    const auto source_type = source_tensor->get_element_type();

    if (dest_type == source_type) {
        ov::npuw::util::copy_tensor_by_dim(source_view, dest_tensor, sequence_dim, sequence_dim);
    } else {
        LOG_WARN("Performing type conversion for " << tensor_name << " tile: " << source_type << " -> " << dest_type);

        // Copy to intermediate buffer first
        auto intermediate_tensor = ov::Tensor(source_type, source_view->get_shape());
        ov::npuw::util::copy_tensor_by_dim(source_view,
                                           ov::get_tensor_impl(intermediate_tensor),
                                           sequence_dim,
                                           sequence_dim);

        // Convert element-by-element
        const size_t total_elements = intermediate_tensor.get_size();
        if (dest_type == ov::element::f32 && source_type == ov::element::f16) {
            // FP16 -> FP32 conversion
            auto src_data = intermediate_tensor.data<ov::float16>();
            auto dst_data = dest_tensor->data<float>();
            for (size_t i = 0; i < total_elements; ++i) {
                dst_data[i] = static_cast<float>(src_data[i]);
            }
        } else if (dest_type == ov::element::f16 && source_type == ov::element::f32) {
            // FP32 -> FP16 conversion
            auto src_data = intermediate_tensor.data<float>();
            auto dst_data = dest_tensor->data<ov::float16>();
            for (size_t i = 0; i < total_elements; ++i) {
                dst_data[i] = static_cast<ov::float16>(src_data[i]);
            }
        } else {
            OPENVINO_THROW("Unsupported type conversion for ", tensor_name, " tile: ", source_type, " -> ", dest_type);
        }
    }
}

// Check if tensor can be reused directly (zero-copy optimization)
bool ov::npuw::JustInferRequest::hfa_can_reuse_tensor_zero_copy(const ov::SoPtr<ov::ITensor>& source_tensor,
                                                                const ov::SoPtr<ov::ITensor>& dest_tensor,
                                                                uint32_t sequence_dim,
                                                                int64_t sequence_offset,
                                                                int64_t tile_length) {
    const auto source_shape = source_tensor->get_shape();
    const int64_t source_full_length = static_cast<int64_t>(source_shape[sequence_dim]);

    // Zero-copy conditions:
    // 1. Offset must be 0 (no slicing from middle)
    // 2. Tile length must match full source length (using entire tensor)
    // 3. Element types must match (no conversion needed)
    return (sequence_offset == 0 && tile_length == source_full_length &&
            dest_tensor->get_element_type() == source_tensor->get_element_type());
}

// ====================================================================================================
// Host Flash Attention (HFA) Tiled Inference
// ====================================================================================================
//
// Implements Flash Attention using tile-based processing to reduce memory footprint.
//
// Algorithm:
//   - Split KV cache into tiles of size tile_size
//   - For tiles [0..N-2]: Use regular_tile_model, output intermediate states (acc, max, d)
//   - For tile [N-1]: Use final_tile_model, output final result with division + transpose
//   - Maintain numerically stable online softmax across tiles
//
// Optimizations:
//   - Zero-copy when tile size equals full tensor size
//   - Pre-cached input/output indices to avoid repeated map lookups
//   - Automatic FP16/FP32 conversion when needed
//
// ====================================================================================================

void ov::npuw::JustInferRequest::run_hfa_tiled_inference(std::size_t real_idx, std::size_t idx) {
    // ================================================================================================
    // SECTION 1: Configuration and Validation
    // ================================================================================================

    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
    auto& hfa_desc = comp_model_desc.host_flash_attention.value();

    NPUW_ASSERT(hfa_desc.is_valid() && "HFA configuration must be valid");
    NPUW_ASSERT(comp_model_desc.hfa_infer_requests.size() == CompiledModel::CompiledModelDesc::HFATileIdx::COUNT &&
                "HFA infer requests must be created");

    // Calculate tile configuration
    const int64_t tile_size = hfa_desc._tile_size;
    const int64_t total_kv_length = m_hfa_selector->context_length();
    const int64_t num_tiles = total_kv_length / tile_size;

    NPUW_ASSERT(total_kv_length % tile_size == 0 && "HFA total KV length must be multiple of tile size for now");

    // ================================================================================================
    // SECTION 2: Input/Output Tensor Extraction
    // ================================================================================================

    const auto& hfa_inputs = m_hfa_io[idx].inputs;
    const auto& hfa_outputs = m_hfa_io[idx].outputs;
    const auto& sdpa_info = hfa_desc._sdpa_attention_info;

    // Use pre-cached SDPA indices
    const auto& sdpa_in = sdpa_info._sdpa_indices;

    auto past_key_tensor = hfa_inputs.at(sdpa_in.past_key);
    auto past_value_tensor = hfa_inputs.at(sdpa_in.past_value);
    auto query_tensor = hfa_inputs.at(sdpa_in.query);
    auto present_key_tensor = hfa_inputs.at(sdpa_in.present_key);
    auto attention_mask_tensor = hfa_inputs.at(sdpa_in.attention_mask);
    auto present_value_tensor = hfa_inputs.at(sdpa_in.present_value);

    auto attention_output_tensor = hfa_outputs.at(0);

    // ================================================================================================
    // SECTION 3: State Initialization
    // ================================================================================================

    // Get tile infer requests
    auto& regular_tile_request =
        comp_model_desc.hfa_infer_requests[CompiledModel::CompiledModelDesc::HFATileIdx::REGULAR_TILE];
    auto& final_tile_request =
        comp_model_desc.hfa_infer_requests[CompiledModel::CompiledModelDesc::HFATileIdx::FINAL_TILE];

    // Use pre-cached indices (populated during compilation)
    const auto& tile_in = sdpa_info._tile_input_indices;
    const auto& tile_out = sdpa_info._tile_output_indices;

    // Get pre-initialized state buffers from runtime context (initialized during setup phase)
    // If optimizations are disabled, get tensors directly from request (no double-buffering)
    ov::SoPtr<ov::ITensor> state_acc, state_max, state_sum;

    if (m_hfa_runtime_ctx && m_hfa_runtime_ctx->has_state_buffers()) {
        // Use pre-initialized state from current buffer (double-buffering enabled)
        const auto& current_buffer = m_hfa_runtime_ctx->get_current_state_buffers();

        state_acc = current_buffer.acc;
        state_max = current_buffer.max;
        state_sum = current_buffer.sum;

        regular_tile_request->set_tensor(hfa_desc._compiled_tile_model->inputs()[tile_in.acc], state_acc);
        regular_tile_request->set_tensor(hfa_desc._compiled_tile_model->inputs()[tile_in.max], state_max);
        regular_tile_request->set_tensor(hfa_desc._compiled_tile_model->inputs()[tile_in.d], state_sum);
    } else {
        // Optimizations disabled: use tensors directly without double-buffering
        state_acc = regular_tile_request->get_tensor(hfa_desc._compiled_tile_model->inputs()[tile_in.acc]);
        state_max = regular_tile_request->get_tensor(hfa_desc._compiled_tile_model->inputs()[tile_in.max]);
        state_sum = regular_tile_request->get_tensor(hfa_desc._compiled_tile_model->inputs()[tile_in.d]);

        // Initialize state tensors for each inference run (no caching)
        runtime::host_flash_attention::HFARuntimeContext::initialize_state_tensors(state_acc, state_max, state_sum);
    }

    // Set query tensor once (constant across all tiles)
    regular_tile_request->set_tensor(hfa_desc._compiled_tile_model->inputs()[tile_in.q], query_tensor);
    final_tile_request->set_tensor(hfa_desc._compiled_final_tile_model->inputs()[tile_in.q], query_tensor);

    // Set output state tensors for regular tile model (will be updated in-place after each tile execution)
    regular_tile_request->set_tensor(hfa_desc._compiled_tile_model->outputs()[tile_out.acc], state_acc);
    regular_tile_request->set_tensor(hfa_desc._compiled_tile_model->outputs()[tile_out.max], state_max);
    regular_tile_request->set_tensor(hfa_desc._compiled_tile_model->outputs()[tile_out.d], state_sum);

    // Set accumulated state tensors as inputs for final tile (will read from regular tiles' outputs)
    final_tile_request->set_tensor(hfa_desc._compiled_final_tile_model->inputs()[tile_in.acc], state_acc);
    final_tile_request->set_tensor(hfa_desc._compiled_final_tile_model->inputs()[tile_in.max], state_max);
    final_tile_request->set_tensor(hfa_desc._compiled_final_tile_model->inputs()[tile_in.d], state_sum);

    // Final attention output
    final_tile_request->set_tensor(hfa_desc._compiled_final_tile_model->outputs()[0], attention_output_tensor);

    // ================================================================================================
    // SECTION 4: Tile Processing Loop
    // ================================================================================================

    // Dimension configuration for tensor slicing
    const uint32_t K_SEQ_DIM = static_cast<uint32_t>(sdpa_info._k_seq_dim);
    const uint32_t V_SEQ_DIM = static_cast<uint32_t>(sdpa_info._v_seq_dim);
    constexpr uint32_t MASK_KV_SEQ_DIM = 3;

    size_t next_available_mask_buffer_idx = 0;  // Track next available pre-allocated mask buffer for cache misses

    // Helper lambda: Process a single tile
    auto process_tile = [&](auto& request,
                            auto& model,
                            const ov::SoPtr<ov::ITensor>& k_source,
                            const ov::SoPtr<ov::ITensor>& v_source,
                            int64_t kv_offset,
                            int64_t mask_offset,
                            int64_t tile_length,
                            bool async = false) {
        // Get tile input buffers
        auto k_tile_buffer = request->get_tensor(model->inputs()[tile_in.k]);
        auto v_tile_buffer = request->get_tensor(model->inputs()[tile_in.v]);
        auto mask_tile_buffer = request->get_tensor(model->inputs()[tile_in.mask]);

        // Extract K tile
        if (hfa_can_reuse_tensor_zero_copy(k_source, k_tile_buffer, K_SEQ_DIM, kv_offset, tile_length)) {
            request->set_tensor(model->inputs()[tile_in.k], k_source);
        } else {
            hfa_extract_and_copy_tile(k_source, k_tile_buffer, K_SEQ_DIM, kv_offset, tile_length, "K");
        }

        // Extract V tile
        if (hfa_can_reuse_tensor_zero_copy(v_source, v_tile_buffer, V_SEQ_DIM, kv_offset, tile_length)) {
            request->set_tensor(model->inputs()[tile_in.v], v_source);
        } else {
            hfa_extract_and_copy_tile(v_source, v_tile_buffer, V_SEQ_DIM, kv_offset, tile_length, "V");
        }

        // Extract mask tile with caching (if enabled) to avoid redundant extraction
        if (attention_mask_tensor) {
            // Check if zero-copy is possible (rare case where full mask matches tile)
            if (hfa_can_reuse_tensor_zero_copy(attention_mask_tensor,
                                               mask_tile_buffer,
                                               MASK_KV_SEQ_DIM,
                                               mask_offset,
                                               tile_length)) {
                request->set_tensor(model->inputs()[tile_in.mask], attention_mask_tensor);
            } else if (m_hfa_runtime_ctx.has_value()) {
                // Cache is enabled - try to find cached tile
                auto cached_tile =
                    m_hfa_runtime_ctx->find_cached_mask_tile(attention_mask_tensor, mask_offset, tile_length);
                if (cached_tile) {
                    // Cache hit - reuse previously extracted tile
                    request->set_tensor(model->inputs()[tile_in.mask], cached_tile);
                    LOG_DEBUG("HFA: Cache hit for mask tile [offset=" << mask_offset << ", length=" << tile_length
                                                                      << "]");
                } else {
                    // Cache miss - extract and cache this tile
                    LOG_DEBUG("HFA: Cache miss for mask tile [offset=" << mask_offset << ", length=" << tile_length
                                                                       << "], extracting...");

                    // Use pre-allocated mask buffer for this tile
                    ov::SoPtr<ov::ITensor> cached_mask_tile =
                        m_hfa_runtime_ctx->get_mask_tile_buffer(next_available_mask_buffer_idx);

                    // Extract mask data into the pre-allocated buffer
                    hfa_extract_and_copy_tile(attention_mask_tensor,
                                              cached_mask_tile,
                                              MASK_KV_SEQ_DIM,
                                              mask_offset,
                                              tile_length,
                                              "Mask");

                    // Cache the extracted tile for future reuse
                    m_hfa_runtime_ctx->cache_mask_tile(attention_mask_tensor,
                                                       mask_offset,
                                                       tile_length,
                                                       cached_mask_tile);

                    // Use the cached tensor for this inference
                    request->set_tensor(model->inputs()[tile_in.mask], cached_mask_tile);

                    // Move to next pre-allocated buffer for next cache miss
                    next_available_mask_buffer_idx++;
                }
            } else {
                // Cache is disabled - extract mask tile on-the-fly directly into tile buffer
                LOG_DEBUG("HFA: Extracting mask tile on-the-fly [offset=" << mask_offset << ", length=" << tile_length
                                                                          << "] (cache disabled)");
                hfa_extract_and_copy_tile(attention_mask_tensor,
                                          mask_tile_buffer,
                                          MASK_KV_SEQ_DIM,
                                          mask_offset,
                                          tile_length,
                                          "Mask");
            }
        }

        // Execute tile (async mode pre-initializes next state buffer in parallel if optimizations enabled)
        if (async) {
            request->start_async();
            if (m_hfa_runtime_ctx && m_hfa_runtime_ctx->has_state_buffers()) {
                m_hfa_runtime_ctx->prepare_next_state_buffers();
            }
            request->wait();
        } else {
            request->infer();
        }
    };

    int64_t mask_tile_offset = 0;
    int64_t kv_tile_offset = 0;

    // Process regular tiles (all but the last one)
    // Each regular tile processes past KV cache and outputs intermediate states (acc, max, d)
    for (int64_t tile_idx = 0; tile_idx < num_tiles - 1; ++tile_idx) {
        process_tile(regular_tile_request,
                     hfa_desc._compiled_tile_model,
                     past_key_tensor,
                     past_value_tensor,
                     kv_tile_offset,
                     mask_tile_offset,
                     tile_size);

        kv_tile_offset += tile_size;
        mask_tile_offset += tile_size;
    }

    // Process final tile separately
    // Final tile processes present KV tokens and produces final attention output
    if (num_tiles > 0) {
        const size_t present_seq_length = present_key_tensor->get_shape()[K_SEQ_DIM];
        const int64_t final_tile_length = static_cast<int64_t>(present_seq_length);

        // Verify that final tile can process entire present KV in one inference
        NPUW_ASSERT(final_tile_length == tile_size &&
                    "Final tile must process entire present KV sequence in a single inference. "
                    "This is guaranteed during compilation (tile_size = query_size = present_seq_length).");

        // Calculate mask offset for final tile (points to tail of mask corresponding to present tokens)
        const int64_t mask_total_length = attention_mask_tensor->get_shape()[MASK_KV_SEQ_DIM];
        const int64_t final_mask_offset = mask_total_length - final_tile_length;

        process_tile(final_tile_request,
                     hfa_desc._compiled_final_tile_model,
                     present_key_tensor,
                     present_value_tensor,
                     0,
                     final_mask_offset,
                     final_tile_length,
                     true);  // async: pre-init next state buffer
    }

    // Switch to other buffer for next inference (only if optimizations enabled)
    if (m_hfa_runtime_ctx && m_hfa_runtime_ctx->has_state_buffers()) {
        m_hfa_runtime_ctx->switch_buffers();
    }
}

void ov::npuw::JustInferRequest::unsafe_infer_spatial(std::size_t real_idx, std::size_t) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
    NPUW_ASSERT(comp_model_desc.spatial.has_value());

    auto& r = m_subrequests[real_idx];

    // Run over the specified range... Note: the full inputs/outputs
    // must be prepared in the m_spatial_io at this point
    const auto& spatial = comp_model_desc.spatial.value();
    const auto num_outputs = comp_model_desc.compiled_model->outputs().size();
    NPUW_ASSERT(m_spatial_selector);

    // Create a sparse vector with full input sizes.
    // For the access simplicity, its size is aligned with function's
    // number of input parameters (activations) so some slots may be
    // not used here.
    // FIXME: All these preparations could be done statically (just once)
    std::vector<ov::Shape> full_in_shapes(comp_model_desc.param_base);
    for (auto&& param : spatial.params) {
        full_in_shapes[param.idx] = m_spatial_io[real_idx].inputs.at(param.idx)->get_shape();
    }

    // Now handle the range, even if it is not a multiply of nway (slice):
    //
    // |<- - - - full range  - - - ->|
    // +------+------+------+------+-+
    // | nway | nway | nway | nway | |
    // +------+------+------+------+-+
    //                              ^tail
    // The block is always compiled to produce nway. If we need a smaller tensor
    // on the last iteration, the sub-nway will be copied from the input range to
    // a temporary tensor, and then the sub-nwway range will be copied from the
    // request's output range.

    std::size_t offset = 0u;
    for (std::size_t i = 0u; i < spatial.nway_iters; i++, offset += spatial.nway) {
        if (!m_spatial_selector->need_submit(offset, spatial.nway)) {
            continue;
        }

        // Collect spatial inputs for this offset
        for (auto&& param : spatial.params) {
            const auto& iport = comp_model_desc.compiled_model->inputs()[param.idx];
            const auto& iview =
                ov::npuw::util::view(m_spatial_io[real_idx].inputs.at(param.idx), param.dim, offset, spatial.nway);
            r->set_tensor(iport, iview);
        }  // for(params)

        // Now set the spatial outputs
        for (std::size_t out_idx = 0u; out_idx < num_outputs; out_idx++) {
            const auto& oport = comp_model_desc.compiled_model->outputs()[out_idx];
            r->set_tensor(oport,
                          ov::npuw::util::view(m_spatial_io[real_idx].outputs.at(out_idx),
                                               spatial.out_dim,
                                               offset,
                                               spatial.nway));
        }  // for(outputs)

        // Now run the part
        r->infer();
    }  // for(full_nway_times)

    // Now process the tail, if required
    if (spatial.tail_size && m_spatial_selector->need_submit(offset, spatial.tail_size)) {
        // Copy the sub-ranges to spatial inputs
        // NOTE: tails buffers are read from/written to at 0th offset!
        for (auto&& param : spatial.params) {
            auto in_view =
                ov::npuw::util::view(m_spatial_io[real_idx].inputs.at(param.idx), param.dim, offset, spatial.tail_size);

            const auto& iport = comp_model_desc.compiled_model->inputs()[param.idx];
            auto out_view =
                ov::npuw::util::view(m_spatial_io[real_idx].input_tails.at(param.idx), param.dim, 0, spatial.tail_size);

            in_view->copy_to(out_view._ptr);
            r->set_tensor(iport, m_spatial_io[real_idx].input_tails.at(param.idx));
        }  // for(params)

        // Now set the tail tensors
        for (std::size_t out_idx = 0u; out_idx < num_outputs; out_idx++) {
            const auto& oport = comp_model_desc.compiled_model->outputs()[out_idx];
            r->set_tensor(oport, m_spatial_io[real_idx].output_tails.at(out_idx));
        }  // for(outputs)

        // Now run the tail infer
        r->infer();

        // Now copy the views from the output full-nway tensor to the output tensors
        for (std::size_t out_idx = 0u; out_idx < num_outputs; out_idx++) {
            auto in_view = ov::npuw::util::view(m_spatial_io[real_idx].output_tails.at(out_idx),
                                                spatial.out_dim,
                                                0,
                                                spatial.tail_size);

            auto out_view = ov::npuw::util::view(m_spatial_io[real_idx].outputs.at(out_idx),
                                                 spatial.out_dim,
                                                 offset,
                                                 spatial.tail_size);
            in_view->copy_to(out_view._ptr);
        }  // for(outputs)
    }
}

void ov::npuw::JustInferRequest::unsafe_infer(std::size_t real_idx, std::size_t idx) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
    auto& r = m_subrequests[real_idx];
    if (comp_model_desc.spatial) {
        unsafe_infer_spatial(real_idx, idx);
    } else if (comp_model_desc.host_flash_attention) {
        run_hfa_tiled_inference(real_idx, idx);
    } else {
        r->infer();  // Run normally
    }
}

void ov::npuw::JustInferRequest::unsafe_run_this_prep_next(std::size_t idx, bool& next_prepared) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    auto real_idx = comp_model_desc.replaced_by.value_or(idx);
    const std::size_t next_idx = next(idx + 1);

    if (comp_model_desc.replaced_by) {
        // This is a function call!
        if (real_idx == real(next_idx)) {
            // The next subgraph is a call to the same function...
            // At this point, THIS infer request is already prepared.
            // Run it, then prepare it again for the next entrace
            if (is_pipelined(real_idx)) {
                // function pipelining is here! and the next rq is ours.
                NPUW_ASSERT(m_funcall_pipeline[idx].next.value() == next_idx);
                unsafe_during(real_idx, idx, [&]() {
                    LOG_DEBUG("Unpacking closures for the NEXT subrequest[" << next_idx << "]...");
                    LOG_BLOCK();
                    // Note: do it here unconditionally - if this request fails,
                    // have to resubmit all the data to the recompiled pair anyway
                    bind_global_parameters(next_idx);
                    unpack_closure(next_idx, m_funcall_pipeline[real_idx].subrequest);
                });
            } else {
                // Function pipelining is not used. THIS infer request
                // is also the NEXT one. Nothing much to do here
                unsafe_infer(real_idx, idx);
                bind_global_parameters(next_idx);
            }
        } else {
            // The next subgraph is NOT a call to the same function!
            // Trigger execution of the current one
            // FIXME: pipelining?
            if (next_idx == 0) {
                // Note: even if m_function_pipelining is ON,
                // SWAP won't happen here - see the below check for .next
                unsafe_infer(real_idx, idx);
            } else {
                unsafe_during(real_idx, idx, [&]() {
                    if (!next_prepared) {
                        bind_global_parameters(next_idx);
                        next_prepared = true;
                    }
                    if (is_pipelined(idx) && m_funcall_pipeline[idx].next) {
                        const auto my_next_idx = m_funcall_pipeline[idx].next.value();
                        LOG_DEBUG("Unpacking closures for the NEXT subrequest[" << my_next_idx << "]...");
                        LOG_BLOCK();
                        unpack_closure(my_next_idx, m_funcall_pipeline[real_idx].subrequest);
                    }
                });
            }
        }
    } else {
        // This is a regular subgraph. Start it async to prepare the next
        // parameters
        if (next_idx == 0) {
            unsafe_infer(real_idx, idx);
        } else {
            unsafe_during(real_idx, idx, [&]() {
                if (!next_prepared) {
                    bind_global_parameters(next_idx);
                    next_prepared = true;
                }
            });
        }
    }  // if (replaced_by)
}

void ov::npuw::JustInferRequest::subscribe_subrequest(std::size_t idx, Completed cb) {
    get_real_subrequest(idx)->set_callback(std::move(cb));
}

void ov::npuw::JustInferRequest::complete_subrequest(std::size_t idx) {
    // do nothing here
}

void ov::npuw::JustInferRequest::cancel_subrequest(std::size_t idx) {
    m_subrequests[idx]->cancel();
}

bool ov::npuw::JustInferRequest::supports_async_pipeline() const {
    return false;
}

void ov::npuw::JustInferRequest::update_subrequest_links(std::size_t) {
    connect_subrequests();
}

bool ov::npuw::JustInferRequest::is_pipelined(std::size_t idx) const {
    const auto& desc = m_npuw_model->m_compiled_submodels[real(idx)];
    return m_use_function_pipelining && desc.replaced_by && !desc.forced_to_fcall;
}
