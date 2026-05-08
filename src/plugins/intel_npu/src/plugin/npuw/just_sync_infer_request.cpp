// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "just_sync_infer_request.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>

#include "attn/attn_subgraph.hpp"
#include "compiled_model.hpp"
#include "host_flash_attention.hpp"
#include "infer_request_utils.hpp"  // to utilize copy_tensor_by_dim
#include "logging.hpp"
#include "moe/moe_subgraph.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "plugin.hpp"
#include "pyramid_attention.hpp"
#include "v1/elements/failsafe.hpp"
#include "weights_bank.hpp"

// ====================================================================================================
// ISubrequestAccessor Interface Implementation
// ====================================================================================================

ov::SoPtr<ov::IAsyncInferRequest> ov::npuw::JustInferRequest::get_subrequest(size_t idx) {
    return m_subrequests[idx];
}

const void* ov::npuw::JustInferRequest::get_submodel_desc(size_t idx) {
    return &m_npuw_model->m_compiled_submodels[idx];
}

ov::npuw::TensorPtr ov::npuw::JustInferRequest::allocate_mem(const ov::element::Type& type,
                                                             const ov::Shape& shape,
                                                             const std::string& device) {
    return allocMem(type, shape, device);
}

bool ov::npuw::JustInferRequest::is_gather_closure(size_t idx, size_t cidx) {
    return m_npuw_model->is_gather_closure(idx, cidx);
}

bool ov::npuw::JustInferRequest::unpack_required(size_t idx, size_t cidx) {
    return m_npuw_model->unpack_required(idx, cidx);
}

bool ov::npuw::JustInferRequest::needs_copy_closure(size_t idx, size_t cidx) {
    return IBaseInferRequest::needs_copy(idx, cidx);
}

std::string ov::npuw::JustInferRequest::subgraph_device(size_t idx) {
    return m_npuw_model->submodel_device(idx);
}

void ov::npuw::JustInferRequest::set_active_subrequest(size_t idx, ov::SoPtr<ov::IAsyncInferRequest> request) {
    m_subrequests[idx] = std::move(request);
}

ov::SoPtr<ov::IAsyncInferRequest> ov::npuw::JustInferRequest::get_pipeline_subrequest(size_t idx) const {
    return idx < m_funcall_pipeline.size() ? m_funcall_pipeline[idx].subrequest : ov::SoPtr<ov::IAsyncInferRequest>{};
}

void ov::npuw::JustInferRequest::set_pipeline_subrequest(size_t idx, ov::SoPtr<ov::IAsyncInferRequest> request) {
    if (idx < m_funcall_pipeline.size()) {
        m_funcall_pipeline[idx].subrequest = std::move(request);
    }
}

bool ov::npuw::JustInferRequest::is_subrequest_pipelined(size_t idx) const {
    return is_pipelined(idx);
}

std::size_t ov::npuw::JustInferRequest::history_size() const {
    return get_history_size();
}

bool ov::npuw::JustInferRequest::subgraph_needs_copy(std::size_t idx) const {
    return needs_copy(idx);
}

const ov::SoPtr<ov::ICompiledModel>& ov::npuw::JustInferRequest::compiled_submodel(size_t idx) const {
    return m_npuw_model->m_compiled_submodels.at(idx).compiled_model;
}

const ov::npuw::v1::subgraphs::CompiledPipeline& ov::npuw::JustInferRequest::subgraph_pipeline(size_t idx) const {
    return m_npuw_model->m_compiled_submodels.at(idx).pipeline;
}

std::size_t ov::npuw::JustInferRequest::subgraph_param_base(size_t idx) const {
    return m_npuw_model->m_compiled_submodels.at(idx).param_base;
}

// ====================================================================================================
// Memory Access Simulation & Function Memory Management
// ====================================================================================================

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
    auto it = m_remaining_reads.find(from);
    if (it == m_remaining_reads.end()) {
        // No cross-subgraph consumers for this output. This can legitimately happen
        // when a prototype output is unused by some call instances (e.g. in KV-sharing
        // models like Gemma4, non-head layers receive shared K/V directly and never
        // consume the layernorm output that head layers forward to K/V projection).
        // Return 0 so the tensor slot is immediately available for reuse.
        return 0u;
    }
    return it->second;
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

    // Create infer requests
    // Preallocate funcall tensors & substitute function call requests
    bool has_spatial = false;
    bool has_moe = false;
    std::size_t moe_real_idx = -1;  // Track which real function has MoE
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

            // Initialize the MoE IO placeholders, if required
            if (ov::npuw::moe::has_compiled_experts(proto_comp_model_desc.pipeline)) {
                // Sanity check: ensure only one MoE function type exists
                if (has_moe && moe_real_idx != real_idx) {
                    OPENVINO_THROW("Only single MoE type is permitted for model");
                }
                has_moe = true;
                moe_real_idx = real_idx;
            }  // if(moe_experts)

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
        auto rqs = create_infer_requests(i, is_piped ? 2 : 1);
        m_subrequests[i] = rqs.at(0);
        if (is_piped) {
            m_funcall_pipeline[i].subrequest = rqs.at(1);
        }

        LOG_INFO("DONE");
    }  // for(submodels)

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
    initialize_subgraph_behaviors();
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

    // Initialize MoE executor if MoE was detected
    if (has_moe) {
        initialize_moe_executor();
    }
}

void ov::npuw::JustInferRequest::initialize_subgraph_behaviors() {
    m_subgraph_behaviors.resize(m_num_submodels);
    m_subgraph_runtime_states.resize(m_num_submodels);
    for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
        const auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
        if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
            continue;
        }
        const auto* spec = get_runtime_behavior_spec(idx);
        if (spec == nullptr) {
            continue;
        }
        if (spec->factory) {
            m_subgraph_behaviors[idx] = spec->factory(spec->context);
        }
        if (m_subgraph_behaviors[idx]) {
            continue;
        }
        const auto* post_legacy_hook = spec->context.get_if<ov::npuw::v1::subgraphs::PostLegacyHook>();
        if (!post_legacy_hook) {
            continue;
        }
        m_subgraph_behaviors[idx] = std::make_unique<ov::npuw::v1::subgraphs::DirectBehavior>(
            [post_legacy_hook = *post_legacy_hook](ov::npuw::v1::subgraphs::InferContext& ctx) {
                ctx.legacy_infer();
                post_legacy_hook(ctx);
            });
    }
}

ov::npuw::v1::subgraphs::InferContext ov::npuw::JustInferRequest::make_behavior_context(std::size_t real_idx,
                                                                                        std::size_t idx) {
    const bool is_function_call = m_npuw_model->m_compiled_submodels[idx].replaced_by.has_value();
    return ov::npuw::v1::subgraphs::InferContext{
        *m_npuw_model,
        *this,
        idx,
        real_idx,
        m_subrequests[real_idx],
        real_idx < m_subgraph_runtime_states.size() ? &m_subgraph_runtime_states[real_idx] : nullptr,
        [this, real_idx, idx]() {
            legacy_infer(real_idx, idx);
        },
        is_function_call ? std::function<void()>{[this, idx]() {
            function_prologue(idx);
        }}
                         : std::function<void()>{},
        [this, real_idx, idx]() {
            OPENVINO_ASSERT(m_moe_executor != nullptr, "Expected MoE executor for opaque MoE run");
            m_moe_executor->run(real_idx, idx);
        }};
}

bool ov::npuw::JustInferRequest::bind_behavior_input(std::size_t idx,
                                                     std::size_t real_idx,
                                                     std::size_t input_idx,
                                                     const ov::SoPtr<ov::ITensor>& tensor,
                                                     RqPtr request) {
    auto* behavior = get_subgraph_behavior(idx);
    if (behavior == nullptr) {
        return false;
    }
    auto ctx = ov::npuw::v1::subgraphs::InferContext{
        *m_npuw_model,
        *this,
        idx,
        real_idx,
        std::move(request),
        real_idx < m_subgraph_runtime_states.size() ? &m_subgraph_runtime_states[real_idx] : nullptr,
        {},
        {},
        {}};
    return behavior->bind_function_input(ctx, input_idx, tensor);
}

const ov::npuw::v1::subgraphs::RuntimeBehaviorSpec* ov::npuw::JustInferRequest::get_runtime_behavior_spec(
    std::size_t idx) const {
    if (idx >= m_npuw_model->m_compiled_submodels.size()) {
        return nullptr;
    }
    const auto& desc = m_npuw_model->m_compiled_submodels[idx];
    const auto behavior_idx = desc.replaced_by.value_or(idx);
    const auto& behavior_desc = m_npuw_model->m_compiled_submodels[behavior_idx];
    if (!behavior_desc.pipeline.runtime_behavior.has_value()) {
        return nullptr;
    }
    return &behavior_desc.pipeline.runtime_behavior.value();
}

ov::npuw::v1::subgraphs::ISubgraphBehavior* ov::npuw::JustInferRequest::get_subgraph_behavior(std::size_t idx) const {
    if (idx >= m_subgraph_behaviors.size()) {
        return nullptr;
    }
    return m_subgraph_behaviors[idx].get();
}

bool ov::npuw::JustInferRequest::behavior_handles_function_prologue(std::size_t idx) const {
    const auto* spec = get_runtime_behavior_spec(idx);
    return spec != nullptr && spec->handles_function_prologue;
}

void ov::npuw::JustInferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                            const ov::SoPtr<ov::ITensor>& tensor) {
    std::unique_lock lock(m_io_storages_mutex);
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

    // Adjust spatial input range, if supported
    if (m_spatial_selector) {
        m_spatial_selector->prepare();
    }

    for (std::size_t idx = 0; idx < m_num_submodels; idx++) {
        auto* behavior = get_subgraph_behavior(idx);
        if (behavior == nullptr) {
            continue;
        }
        auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
        const auto real_idx = comp_model_desc.replaced_by.value_or(idx);
        auto ctx = make_behavior_context(real_idx, idx);
        behavior->prepare(ctx);
    }

    // Submit global parameters (if needed) for the first subgraph
    bind_global_parameters(next(0));

    // If funcall pipelining is enabled, prefill the function "heads"
    // with constant arguments. The list of heads is empty otherwise.
    for (auto&& id : m_funcall_heads) {
        LOG_DEBUG("Pre-initializing weights for subgraph[" << id << "]");
        unpack_closure(id, m_subrequests[id]);
    }

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
    const bool is_moe = ov::npuw::moe::has_compiled_state(func_desc.pipeline);
    auto* behavior = get_subgraph_behavior(idx);
    std::optional<ov::npuw::v1::subgraphs::InferContext> behavior_ctx;
    if (behavior != nullptr) {
        behavior_ctx.emplace(make_behavior_context(real_idx, idx));
    }

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
            } else if (behavior != nullptr && behavior_ctx.has_value() &&
                       behavior->bind_function_input(*behavior_ctx, i, i_tensor)) {
                continue;
            } else if (is_moe) {
                // MoE layer: delegate to executor for input binding
                if (m_moe_executor->function_prologue_moe_input(idx, real_idx, i, i_tensor)) {
                    continue;
                }
            } else {
                // Default case
                m_subrequests[real_idx]->set_tensor(iport, i_tensor);
            }
        }  // if (link_iter)
    }  // for(param_base)

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
        if (ov::npuw::moe::has_compiled_experts(func_desc.pipeline)) {
            // MoE case - delegate to executor for output binding
            m_moe_executor->function_prologue_moe_output(idx, i, o_tensor);
        } else if (behavior != nullptr && behavior_ctx.has_value() &&
                   behavior->bind_function_output(*behavior_ctx, i, o_tensor)) {
            continue;
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

void ov::npuw::JustInferRequest::initialize_moe_executor() {
    LOG_INFO("Creating MoE executor...");

    // Create MoE executor with dependency injection
    m_moe_executor = std::make_unique<ov::npuw::moe::MoEExecutor>(
        *this,  // ISubrequestAccessor
        [this](const ov::element::Type& type, const ov::Shape& shape, const std::string& device) {
            // Allocator callback
            return allocMem(type, shape, device);
        });

    LOG_INFO("MoE executor created");

    // Prepare MoE resources for each sublayer
    size_t pool_size = m_npuw_model->m_cfg.get<::intel_npu::NPUW_MOE_POOL_SIZE>();
    for (size_t i = 0; i < m_num_submodels; i++) {
        auto& desc = m_npuw_model->m_compiled_submodels[i];

        size_t real_idx = desc.replaced_by.value_or(i);
        auto& real_desc = m_npuw_model->m_compiled_submodels[real_idx];

        // Check if the real function body has MoE experts
        if (ov::npuw::moe::has_compiled_experts(real_desc.pipeline)) {
            m_moe_executor->prepare(i, real_idx, m_num_submodels, pool_size);
        }
    }

    LOG_INFO("MoE executor initialized successfully");
}

void ov::npuw::JustInferRequest::run_subrequest_for_success(std::size_t idx) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    const auto real_idx = comp_model_desc.replaced_by.value_or(idx);
    bool next_prepared = false;
    auto* behavior = get_subgraph_behavior(idx);

    // Feeding the global Parameters is now part of the common
    // execution pipeline: See how it is done in
    // `unsafe_run_this_prep_next()`.  Now we only need to bind
    // the subrequest' outputs to global Results, if relevant.
    bind_global_results(idx);

    if (comp_model_desc.replaced_by && !behavior_handles_function_prologue(idx)) {
        function_prologue(idx);
    }
    if (behavior != nullptr) {
        auto ctx = make_behavior_context(real_idx, idx);
        behavior->prologue(ctx);
    }
    dump_input_tensors(idx);

    LOG_DEBUG("Trying to run subrequest[" << idx << "]...");
    LOG_BLOCK();
    unsafe_run_this_prep_next(idx, next_prepared);

    LOG_DEBUG("Done: " << idx << "(exec subrequest)");

    dump_output_tensors(idx);  // FIXME: Called here unconditionally, need to refactor
    if (behavior != nullptr) {
        auto ctx = make_behavior_context(real_idx, idx);
        behavior->epilogue(ctx);
    }
    if (is_pipelined(idx) && m_funcall_pipeline[idx].next) {
        // Swap the next (pipelined, semi-prepared) infer request in the chain
        // with the default (to be accessed next) one.
        std::swap(m_subrequests[real_idx], m_funcall_pipeline[real_idx].subrequest);
    }
}

void ov::npuw::JustInferRequest::unsafe_during(std::size_t real_idx, std::size_t idx, const std::function<void()>& f) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];

    if (!comp_model_desc.spatial && ov::npuw::attn::get_compiled_hfa(comp_model_desc.pipeline.context) == nullptr &&
        !ov::npuw::moe::has_compiled_experts(comp_model_desc.pipeline)) {
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

void ov::npuw::JustInferRequest::legacy_infer(std::size_t real_idx, std::size_t idx) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[real_idx];
    auto& r = m_subrequests[real_idx];
    if (comp_model_desc.spatial) {
        unsafe_infer_spatial(real_idx, idx);
    } else {
        r->infer();  // Run normally
    }
}

void ov::npuw::JustInferRequest::unsafe_infer(std::size_t real_idx, std::size_t idx) {
    if (auto* behavior = get_subgraph_behavior(idx)) {
        auto ctx = make_behavior_context(real_idx, idx);
        behavior->run(ctx);
        return;
    }
    legacy_infer(real_idx, idx);
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
