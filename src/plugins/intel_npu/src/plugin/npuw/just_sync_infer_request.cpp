// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "just_sync_infer_request.hpp"

#include <algorithm>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "compiled_model.hpp"
#include "infer_request_utils.hpp"  // to utilize copy_tensor_by_dim
#include "logging.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "plugin.hpp"
#include "pyramid_attention.hpp"
#include "util.hpp"
#include "weights_bank.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"


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
        LOG_VERB("DEBUG assign_memory 0");

        if (comp_model_desc.flash_attention) {
            NPUW_ASSERT(!comp_model_desc.compiled_model || comp_model_desc.replaced_by);
        } else {
            LOG_VERB("DEBUG assign_memory -0");

            if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
                // no model & no funcall - optimized out, do nothing
                continue;
            }
        }
        LOG_VERB("DEBUG assign_memory 1");

        // Simulate subgraph execution: poll its input list first
        const auto& read_list = m_sim.read_list(idx);
        LOG_VERB("DEBUG assign_memory 2");

        // Now, get the outputs for the subgraph. If it is "regular", there's
        // nothing to do - this subgraph owns its outputs on its own.
        // If it is a function, though - look up in the function's memory storage.
        if (comp_model_desc.replaced_by) {
            LOG_VERB("DEBUG assign_memory 3");

            const auto real_idx = comp_model_desc.replaced_by.value();
            const auto& proto_comp_model_desc = m_model->m_compiled_submodels[real_idx];
            LOG_VERB("DEBUG assign_memory 4");
            size_t num_outs = 0;
            if (proto_comp_model_desc.flash_attention) {
                LOG_VERB("DEBUG assign_memory 5");
                auto & fam = proto_comp_model_desc.flash_attention->_compiled_models;
                // taking last FA model with its sizes, intermidiate tensors are handled internally
                num_outs = fam[fam.size() - 1]->outputs().size();
                LOG_VERB("DEBUG assign_memory 6");
            } else {
                num_outs = proto_comp_model_desc.compiled_model->outputs().size();
            }

            for (std::size_t out_idx = 0u; out_idx < num_outs; out_idx++) {
                LOG_VERB("DEBUG assign_memory 7+" << out_idx);
                const LinkFrom this_out = LinkFrom{idx, out_idx};
                assign(this_out);
            }
        }

        // Here happens the imaginary execution... Hocus pocus, done - that's a
        // simulation after all
        // After the execution, mark that the read_list was read.
        for (auto&& from : read_list) {
            LOG_VERB("DEBUG assign_memory 8");
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

        auto get_proto_model = [&]() {
            if (proto_comp_model_desc.flash_attention) {
                const auto & acm = proto_comp_model_desc.flash_attention->_compiled_models;
                return acm[acm.size() - 1];
            }
            return proto_comp_model_desc.compiled_model;
        };

        const auto& oport = get_proto_model()->outputs()[from.second];
        ov::Shape oshape = oport.get_shape();

        if (proto_comp_model_desc.spatial) {
            oshape[proto_comp_model_desc.spatial->out_dim] = proto_comp_model_desc.spatial->range;
        }
        const auto& device = m_model->funcall_mem_device(real_idx);
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

    // Create infer requests
    // Preallocate funcall tensors & substitute function call requests
    bool failover_happened = false;
    bool has_spatial = false;
    bool has_dynamic = false;
    bool has_pyramid = false;
    bool has_flash = false;
    std::size_t dynamic_sub_idx = -1;
    std::size_t pyramid_sub_idx = -1;
    std::size_t flash_sub_idx = -1;
    for (size_t i = 0; i < m_num_submodels; i++) {
        LOG_INFO("Creating infer request for Subgraph[" << i << "]...");
        LOG_BLOCK();
        auto& comp_model_desc = m_npuw_model->m_compiled_submodels[i];

        // if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
        //     // no model & no funcall - optimized out, do nothing
        //     LOG_INFO("OPTIMIZED OUT");
        //     continue;
        // }


        if (comp_model_desc.flash_attention) {
            NPUW_ASSERT(!comp_model_desc.compiled_model || comp_model_desc.replaced_by);
        } else {
            if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
                // no model & no funcall - optimized out, do nothing
                continue;
            }
        }

        // FIXME: Shouldn't this be handled by the base class? (in create_tensor)
        // A special case for function calls
        if (comp_model_desc.replaced_by) {
            // Pre-allocate output tensors for this function call
            const auto real_idx = comp_model_desc.replaced_by.value();
            auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
            auto get_proto_model = [&]() {
                if (proto_comp_model_desc.flash_attention) {
                    const auto & acm = proto_comp_model_desc.flash_attention->_compiled_models;
                    return acm[acm.size() - 1];
                }
                return proto_comp_model_desc.compiled_model;
            };
            auto& proto_comp_model = get_proto_model();
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
                        //TODO: es this might no work at all for input and output there might be not a model but enssamble of ports
                        const auto& iport = get_proto_model()->inputs()[p.idx];
                        m_spatial_io[real_idx].input_tails[p.idx] =
                            allocOut(iport, m_npuw_model->funcall_mem_device(real_idx));
                    }
                    const auto num_outs = get_proto_model()->outputs().size();
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

            if (proto_comp_model_desc.flash_attention) {
                // Sanity check first
                if (has_flash && flash_sub_idx != real_idx) {
                    OPENVINO_THROW("Only single flash attention type is permitted for model");
                }
                has_flash = true;
                flash_sub_idx = real_idx;
                m_attention_io[i].inputs.resize(proto_comp_model_desc.param_base);
            }  // if(flash-attention)

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
        ov::npuw::IBaseInferRequest::RqPtrs rqs;

        bool need_regual_ir = true;
        // Flash-attention infer-requests are different so we dont need to reuse or create original one
        if (comp_model_desc.replaced_by) {
            const auto real_idx = comp_model_desc.replaced_by.value();
            auto& proto_comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
            if (proto_comp_model_desc.flash_attention) {
                LOG_INFO("FLASH - attention: creating inferrequest internally for: " << real_idx);
                NPUW_ASSERT(is_piped && "flash-attention pipelining not supported yet");
                setup_flash_infer_requests(real_idx, is_piped, false);
                rqs = m_npuw_model->m_compiled_submodels[real_idx].flash_infer_requests;
                need_regual_ir = false;
            }
        }

        if (need_regual_ir) {
            rqs = create_infer_requests(i, is_piped ? 2 : 1, &recompiled);
        }

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

    alloc_io();
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

    if (has_flash) {
        // TODO: es runtime features??? not sure we need it here at all while we need to recreate inferrequests
        m_flash_selector.reset(new runtime::flash_attention::All(8));

        const auto& flash_dyn = m_npuw_model->m_compiled_submodels.at(flash_sub_idx).flash_attention.value();

        const auto tile_count = flash_dyn.num_tiles();
        if (!m_npuw_model->m_cfg.get<::intel_npu::NPUW_ATTN_DYN>()) {
            LOG_DEBUG("flash-attention: no dynamic attention");
            // Even if the attention is detected and ready to go pyramid,
            // force it on the full range
            m_flash_selector.reset(new runtime::flash_attention::All(tile_count));
        } else {
            m_flash_selector = runtime::flash_attention::PositionIDs::find(flash_dyn, *this);
            if (!m_flash_selector) {
                LOG_WARN("Flash-attention dynamic capability is enabled, but no run-time features were found.");
                // Create All selector with the number of pyramid models
                m_flash_selector.reset(new runtime::flash_attention::All(tile_count));
            }
            LOG_DEBUG("flash-attention: + dynamic attention num_tiles="
                << m_flash_selector->tile_id() << ", len=" << m_flash_selector->length() << "past_len" << m_flash_selector->past_length());
        }
    }
}

void ov::npuw::JustInferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                            const ov::SoPtr<ov::ITensor>& tensor) {
    // Check that it's I/O
    NPUW_ASSERT(m_port_to_tensor.at(port).persistent);

    // Assigning via .at() to ensure it is a known port
    m_port_to_tensor.at(port).tensor = tensor;

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

ov::npuw::TensorPtr ov::npuw::JustInferRequest::alloc_global_out(std::size_t out_idx) {
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
    // TODO: flash attention should be done without selector
    if (m_flash_selector) {
        // m_pyramid_selector->prepare(get_history_size());

        // Get the pyramid model ID based on current sequence length (updated in prepare())
        //auto pyramid_id = m_pyramid_selector->pyramid_id();

        for (auto&& id : m_funcall_heads) {
            auto& comp_model_desc = m_npuw_model->m_compiled_submodels[id];

            if (comp_model_desc.flash_attention.has_value()) {
                m_subrequests[id] = comp_model_desc.flash_infer_requests[npuw::function::FlashAttention::eConcat];
                // if (is_pipelined(id)) {
                //     m_funcall_pipeline[id].subrequest = comp_model_desc.pyramid_pipeline_requests[pyramid_id];
                // }
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
    const bool is_flash = func_desc.flash_attention.has_value();

    // Generalized: check if input is neither param nor mask
    auto is_non_param_mask = [](const auto& info, std::size_t in_idx) {
        const bool not_param = std::none_of(info.params.begin(), info.params.end(), [&](auto&& p) {
            return p.idx == in_idx;
        });
        const bool not_mask = in_idx != info.mask_idx;
        return not_param && not_mask;
    };
    auto get_compiled = [&]() {
        if (is_flash) {
            return func_desc.flash_attention->_compiled_models[0];
        }
        return func_desc.compiled_model;
    };
    auto get_compiled_back_by_id = [&](size_t prod_idx) {
        auto submodel_desc = m_npuw_model->m_compiled_submodels[prod_idx];//.compiled_model->outputs()[prod_port];
        if (submodel_desc.flash_attention) {
            return submodel_desc.flash_attention->_compiled_models.back();
        }
        return submodel_desc.compiled_model;
    };
    auto get_back_subrequest_by_id = [&](size_t prod_idx) {
        auto submodel_desc = m_npuw_model->m_compiled_submodels[prod_idx];//.compiled_model->outputs()[prod_port];
        if (submodel_desc.flash_attention) {
            return submodel_desc.flash_infer_requests.back();
        }
        return m_subrequests[prod_idx];
    };

    // Function call prologue:
    // 1. Walk through function dependencies and set the respective tensors
    //    as parameters
    for (size_t i = 0; i < func_desc.param_base; i++) {
        LOG_DEBUG("Binding parameter[" << i << "]...");
        LOG_BLOCK();



        const auto& iport = get_compiled()->inputs()[i];
        auto link_iter = m_npuw_model->m_submodels_input_to_prev_output.find({idx, i});

        if (link_iter != m_npuw_model->m_submodels_input_to_prev_output.end()) {
            std::size_t prod_idx;
            std::size_t prod_port;
            std::tie(prod_idx, prod_port) = link_iter->second;

            const auto& i_tensor = [&]() {
                if (!m_npuw_model->m_compiled_submodels[prod_idx].replaced_by) {
                    // Producer is a normal model -> take its tensor directly
                    const auto& oport = get_compiled_back_by_id(prod_idx)->outputs()[prod_port];
                    return get_back_subrequest_by_id(prod_idx)->get_tensor(oport);
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
            } else if (is_flash) {
                // flash attention
                // if (is_non_param_mask(*func_desc.flash_attention, i)) {
                //     m_subrequests[real_idx]->set_tensor(iport, i_tensor);
                // } else
                {
                    LOG_DEBUG("Binding m_attention_io at"<< idx);
                    m_attention_io[idx].inputs.at(i) = i_tensor;
                }
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

    if (is_flash) {
        LOG_DEBUG("function_prologue flash_attn");
        m_profile["attn(act)"] += ov::npuw::perf::ms_to_run([&]() {
            function_prologue_flash_attn(real_idx, idx);
        });
    }

    // 2. Unpack the function closure -- right here, if pipelining if not enabled.
    // If it is enabled, the flow is a little bit different - see run_subrequest_for_success()
    // for details.
    if (!is_pipelined(idx) && m_closure_update_required && !func_desc.forced_to_fcall) {
        LOG_DEBUG("Unpacking closures...");
        LOG_BLOCK();
        if (is_flash) {
            LOG_DEBUG("Unpacking for flash_attention look not implemented...");
        } else {
            unpack_closure(idx, m_subrequests[real_idx]);
        }
        LOG_DEBUG("Done.");
    }

    // 3. Tell the function which results to produce (this time).
    // Note it covers both internal tensors used by other subgraphs as well as
    // the Result tensors for the entire network.
    // ..Since the tensors allocated for outputs of the networks ARE taken from the
    // "funcall_results" if those are produced by funcall results.

    for (std::size_t i = 0; i < get_compiled_back_by_id(real_idx)->outputs().size(); i++) {
        LOG_DEBUG("Binding result[" << i << "]...");
        auto& oport = get_compiled_back_by_id(real_idx)->outputs()[i];
        auto o_tensor = m_funcall_result.at({idx, i});
        if (is_flash) {
            //TODO: what to do with flash attention IO?
            LOG_DEBUG("binding results for flash_attn - loooks not implemented yet");
            //m_spatial_io[real_idx].outputs.at(i) = o_tensor;
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

void ov::npuw::JustInferRequest::function_prologue_flash_attn(std::size_t real_idx, std::size_t idx) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
    NPUW_ASSERT(comp_model_desc.flash_attention.has_value());


    //TODO: es why do we need that at all ???
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
        if (proto_comp_model_desc.flash_attention) {
            setup_flash_infer_requests(real_idx, is_piped, true);
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

void ov::npuw::JustInferRequest::setup_flash_infer_requests(std::size_t real_idx, bool is_piped, bool is_recreate) {
    auto& submodel_desc = m_npuw_model->m_compiled_submodels[real_idx];
    if (!submodel_desc.flash_attention.has_value()) {
        return;
    }

    LOG_INFO((is_recreate ? "Recreating" : "Creating") << " flash infer requests...");
    LOG_BLOCK();

    const auto& flash_models = submodel_desc.flash_attention.value()._compiled_models;
    const size_t num_flash_models = flash_models.size();

    // Clear existing requests if recreating
    if (is_recreate) {
        submodel_desc.flash_infer_requests.clear();
    }
    submodel_desc.flash_infer_requests.resize(num_flash_models);

     // Create infer requests for all flash models
     for (size_t model_idx = 0; model_idx < num_flash_models; ++model_idx) {
        try {
            // Create main infer request
            submodel_desc.flash_infer_requests[model_idx] = flash_models[model_idx]->create_infer_request();
            // Create pipeline infer request if pipelined
            // if (is_piped) {
            //     submodel_desc.pyramid_pipeline_requests[model_idx] = pyramid_models[model_idx]->create_infer_request();
            // }
        } catch (const std::exception& ex) {
            LOG_ERROR("Failed to " << (is_recreate ? "recreate" : "create") << " infer request for flash-attention model["
                                   << model_idx << "]: " << ex.what());
            NPUW_ASSERT(false && "FlashAttention model infer request creation/recreation failed");
        } catch (...) {
            LOG_ERROR("Failed to " << (is_recreate ? "recreate" : "create") << " infer request for flash-attention model["
                                   << model_idx << "]: Unknown error");
            NPUW_ASSERT(false && "Flash-attention model infer request creation/recreation failed with unknown error");
        }
    }
    using FA = ov::npuw::function::FlashAttention;

    // Share input tensors between flash-concat model and main infer requests
    const auto & concat_model = flash_models[FA::eConcat];
    auto & concat_infer_request = submodel_desc.flash_infer_requests[FA::eConcat];
    const size_t num_inputs = concat_model->inputs().size();
    // TODO: have to avoid compiled_model inputs
    //LOG_INFO("num_inputs=" << num_inputs << ", submodel_desc.compiled_model->inputs().size()=" << submodel_desc.compiled_model->inputs().size());
    //NPUW_ASSERT(num_inputs == submodel_desc.compiled_model->inputs().size());


    // TODO: might be not perfect since shape binding might fail one day
    auto cache_inputs_by_shape = [&] (uint8_t model_id)  {
        std::map<ov::Shape, ov::Output<const ov::Node>> cache;
        const auto & model = flash_models[model_id];
        //auto & infer_request = submodel_desc.flash_infer_requests[model_id];
        const size_t num_inputs = model->inputs().size();

        for (size_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
            auto input = model->inputs()[input_idx];
            LOG_DEBUG("model[" << model_id << "] input["<< input_idx << "]=" << input.get_shape());
            cache[input.get_shape()] = input;
        }
        return cache;
    };

    using cache_t = std::map<ov::Shape, ov::Output<const ov::Node>>;
    std::unordered_map<uint8_t, cache_t> cached_inputs = {
        {FA::eConcat, cache_inputs_by_shape(FA::eConcat)},
        {FA::eTile, cache_inputs_by_shape(FA::eTile)},
        {FA::eDivide, cache_inputs_by_shape(FA::eDivide)},
    };

    // assign tensor for shape matched input
    auto model_type = [](uint8_t flash_model_idx) {
        return flash_model_idx == FA::eConcat ? "concat" : (flash_model_idx == FA::eTile? "tile" : "divide_tile");
    };
    auto reuse_input_tensor = [&](uint8_t flash_model_idx, size_t input_idx) -> bool {
        auto submodel_input = submodel_desc.compiled_model->inputs()[input_idx];
        if (!cached_inputs[flash_model_idx].count(submodel_input.get_shape())) {
            LOG_DEBUG("submodel_input["<< input_idx << "]=" << submodel_input.get_shape()
                << "- not a direct " << model_type(flash_model_idx) << " input, skipping");
            return false;
        }

        auto cached_input = cached_inputs[flash_model_idx].at(submodel_input.get_shape());
        auto main_tensor_ptr = m_subrequests[real_idx]->get_tensor(submodel_input)->data();
        auto flash_infer_request = submodel_desc.flash_infer_requests[flash_model_idx];
        auto flash_tensor = flash_infer_request->get_tensor(cached_input);
        auto shared_tensor = ov::get_tensor_impl(
            ov::Tensor(flash_tensor->get_element_type(), flash_tensor->get_shape(), main_tensor_ptr));
        flash_infer_request->set_tensor(cached_input, shared_tensor);

        LOG_DEBUG("submodel_input["<< input_idx << "]=" << submodel_input.get_shape() << "- shared to flash Model " << model_type(flash_model_idx));
        return true;
    };

    // TODO: es we removed imput tensor completely as opposed to pyramid -
    // where that infer-request masched shape of biggest pyramid
    // reuse tensor and connecting to tile and concat models
    // for (size_t input_idx = 0; input_idx < submodel_desc.compiled_model->inputs().size(); ++input_idx) {
    //     for (auto cached_idx : cached_inputs) {
    //         reuse_input_tensor(cached_idx.first, input_idx);
    //     }
    // }




            // binding remained inputs to a repeating block model as well as acc / max / d
            // Q-parameter input
            // auto in_q = tile_model->outputs[5];
            // auto convert_or_ = connected_output_node(submodel_input.second.get_node_shared_ptr());

            // bool is_convert = is_type<ov::op::v0::Convert>(convert_or_);
            // if (is_convert) {
            //     convert_or_ = connected_output_node(convert_or_);
            // }
            // bool is_add = is_type<ov::op::v1::Add>(convert_or_);
            // bool is_matmul = is_type<ov::op::v0::MatMul>(convert_or_);

            // NPUW_ASSERT(is_add || is_matmul);

            // // Tile submodel has following parameters
            // // ov::ParameterVector{in_past_a, in_past_m, in_past_d, in_full_k, in_full_v, in_q=matmull, in_m=add},
            // size_t tile_model_input_idx = 0;
            // if (is_add) {
            //     LOG_DEBUG("submodel_input["<< submodel_input.first << "]=" << submodel_input.second.get_shape()
            //         << " - not yet sharing tensor with subrequest[" << real_idx << "] M-input it will be attached during inference");
            //     //tile_model_input_idx = 6;
            //     continue;
            // } else if (is_matmul) {
            //     LOG_DEBUG("submodel_input["<< submodel_input.first << "]=" << submodel_input.second.get_shape()
            //         << " - sharing tensor with subrequest[" << real_idx << "] flash-attention Q-value");

            //     tile_model_input_idx = 5;
            // } else {
            //     LOG_ERROR("invalid parameter for submodel["<< submodel_input.first << "]=" << submodel_input.second.get_shape());
            //     NPUW_ASSERT(false);
            // }

            // binding infer requests inputs
            // Get tensor from main infer request and share its memory with the flash infer request

            // Q-parameter input
            // const auto q_input_id = 5;


            // auto tile_input = tile_model->inputs()[q_input_id];
            // auto tile_tensor = tile_infer_request->get_tensor(tile_input);

            // for (auto submodel_input : main_inputs_non_kv_cache) {
            //     auto main_tensor_ptr = m_subrequests[real_idx]->get_tensor(submodel_input.second);
            //     main
            // }


    //         auto shared_tensor = ov::get_tensor_impl(
    //             ov::Tensor(tile_tensor->get_element_type(), tile_tensor->get_shape(), main_tensor_ptr));
    //         concat_infer_request->set_tensor(tile_input, shared_tensor);
    //     }
    // }

    // TODO: should we share concat->output and tile_inputs by tensors, also between tile models
    if (!is_recreate && num_flash_models > 0) {
        LOG_INFO("Successfully created " << (num_flash_models)
                                         << " new flash-attention infer requests and reused 1 original request");
    }


    LOG_INFO("Done.");
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
    LOG_DEBUG("unsafe_during: spatial=" << comp_model_desc.spatial.has_value()
        << ", flash=" << comp_model_desc.flash_attention.has_value()
        << ", pyramid=" << comp_model_desc.pyramid_attention.has_value()
        << ", attention=" << comp_model_desc.attention.has_value());

    if (!comp_model_desc.spatial && !comp_model_desc.flash_attention) {
        // Normal: trigger request asynchronously, run `f` in this context
        // FIXME: dynamic could hit here too, but it has special logic
        // around execution which makes it harder to run than a plain start_async()
        auto& r = m_subrequests[real_idx];
        r->start_async();
        f();  // expect noexcept
        r->wait();
    } else {
        // Spatial... Do the opposite - run f
        // asynchronously, and meanwhile run the spatial inference
        auto future = std::async(std::launch::async, f);
        unsafe_infer(real_idx, idx);
        future.wait();
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

void ov::npuw::JustInferRequest::unsafe_infer_flash_attention(std::size_t real_idx, std::size_t) {
    LOG_DEBUG("unsafe_infer_flash_attention");
    LOG_BLOCK();
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
    NPUW_ASSERT(comp_model_desc.flash_attention.has_value());

    auto& r = m_subrequests[real_idx];

    // Run all flash_attention requests in sequence
    using FA = ov::npuw::function::FlashAttention;
    //auto fa_req = {FA::eConcat, FA::eTile, FA::eDivide};

    // concat should be immidiately available:
   // auto m_concat = comp_model_desc.compiled_model[FA::eConcat];
    auto r_concat = comp_model_desc.flash_infer_requests[FA::eConcat];
    r_concat->infer();

    LOG_DEBUG("tensor for port: " << r_concat->get_outputs()[0].get_shape());
    auto full_k = r_concat->get_tensor(r_concat->get_outputs()[0]);
    LOG_DEBUG("tensor for port: " << r_concat->get_outputs()[1].get_shape());
    auto full_v = r_concat->get_tensor(r_concat->get_outputs()[1]);

    // const auto& iport = comp_model_desc.compiled_model->inputs()[0];
    // LOG_DEBUG("tensor for port: " << iport.get_shape());

    for (auto && main_input : r->get_inputs()) {
        LOG_DEBUG("main inputs for external request: " << main_input.get_shape());
    }

    // for (auto && main_input : comp_model_desc.compiled_model->inputs()) {
    //     LOG_DEBUG("main inputs for compiled model: " << main_input.get_shape());
    // }



    //auto m_input = comp_model_desc.compiled_model->inputs()[4];
    // TODO: where to get m_tensors ?
    // we dont have infer-request that might satisfy - so have to pick from attention_io i guess

    auto r_tile = comp_model_desc.flash_infer_requests[FA::eTile];
    auto r_last_tile = comp_model_desc.flash_infer_requests[FA::eDivide];

    auto tile_inputs = r_tile->get_inputs();
    auto last_tile_inputs = r_last_tile->get_inputs();


    for (auto && main_input : r_concat->get_inputs()) {
        LOG_DEBUG("concat input: " << main_input.get_shape());
    }

    for (auto && main_input : r_tile->get_inputs()) {
        LOG_DEBUG("tile input: " << main_input.get_shape());
    }

    for (auto && main_input : r_tile->get_inputs()) {
        LOG_DEBUG("last-tile input: " << main_input.get_shape());
    }

    // TODO: need to follow spatial case colution with dim-index selection per parameter
    auto k_tensor_spatial_dim = 2;
    auto v_tensor_spatial_dim = 3;
    auto m_tensor_spatial_dim = 3;

    // TODO: how to specify that
    const size_t TSZ = 1024;
    for (size_t offset = 8192 - TSZ; offset != 8192; offset += TSZ) {
        auto last_tile = (offset + TSZ == 8192);
        // this_k = np.copy(full_k[:, :, offset:offset+TSZ, :])
        // this_v = np.copy(full_v[:, :, :, offset:offset+TSZ])
        // this_m = np.copy(full_m[:, :, :, offset:offset+TSZ])

        ov::SoPtr<ov::IAsyncInferRequest> r_current_tile = last_tile ? r_last_tile : r_tile;
        auto current_inputs = last_tile ?  last_tile_inputs : tile_inputs;

        const auto& this_k = ov::npuw::util::view(full_k, k_tensor_spatial_dim, offset, TSZ);
        LOG_DEBUG("copy this_k at offset=" << offset);
        this_k->copy_to(r_current_tile->get_tensor(current_inputs[3])._ptr);

        const auto& this_v = ov::npuw::util::view(full_v, v_tensor_spatial_dim, offset, TSZ);
        LOG_DEBUG("copy this_v at offset=" << offset);
        this_v->copy_to(r_current_tile->get_tensor(current_inputs[4])._ptr);

        // TODO: incomplete
        //        const auto& this_m = ov::npuw::util::view(full_m, m_tensor_spatial_dim, offset, TSZ);
        // TODO: need to bind past_a, past_m, past_d
        LOG_DEBUG("running "<< (last_tile ? "last_tile" : "tile") <<" inference at offset=" << offset);
        r_current_tile->infer();
    }

    // t.start()
    // results = model([past_a, past_m, past_d, this_k, this_v, ii[Inputs.Q.value], this_m])
    // t.stop()

    // tile inferes are need to reuse parts of concat_outputs and it's own outputs
    LOG_DEBUG("Done.");
}

void ov::npuw::JustInferRequest::unsafe_infer(std::size_t real_idx, std::size_t idx) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
    auto& r = m_subrequests[real_idx];
    if (comp_model_desc.spatial) {
        unsafe_infer_spatial(real_idx, idx);
    } else if (comp_model_desc.flash_attention) {
        unsafe_infer_flash_attention(real_idx, idx);
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
