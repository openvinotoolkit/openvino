// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "just_sync_infer_request.hpp"

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "compiled_model.hpp"
#include "logging.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "plugin.hpp"
#include "util.hpp"

ov::npuw::JustInferRequest::JustInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model)
    : IBaseInferRequest(compiled_model) {
    m_use_function_pipelining = m_npuw_model->m_cfg.get<::intel_npu::NPUW_FUNCALL_ASYNC>();
    if (m_use_function_pipelining) {
        LOG_WARN("Function call pipelining is enabled for " << m_npuw_model->m_name
                                                            << ", expect a higher memory consumption");
        m_funcall_pipeline.resize(m_num_submodels);
    }

    // Create infer requests
    // Preallocate funcall tensors & substitute function call requests
    bool failover_happened = false;
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
            // Pre-allocate output tesnors for this function call
            const auto real_idx = comp_model_desc.replaced_by.value();
            auto& proto_comp_model = m_npuw_model->m_compiled_submodels[real_idx].compiled_model;
            for (size_t out_idx = 0; out_idx < proto_comp_model->outputs().size(); out_idx++) {
                const auto& port = proto_comp_model->outputs()[out_idx];
                m_funcall_result[LinkFrom{i, out_idx}] =
                    ov::get_tensor_impl(ov::Tensor(port.get_element_type(), port.get_shape()));
            }
            if (real_idx != i) {
                // If this function call is NOT the function body, do nothing here - the original
                // request will be used.
                LOG_INFO("REUSE " << real_idx);
                continue;
            }
        }  // if(replaced_by)

        // Special cases are handled -- so nothing to do here
        bool recompiled = false;
        auto rqs = create_infer_requests(i, m_use_function_pipelining ? 2 : 1, &recompiled);
        failover_happened |= recompiled;
        m_subrequests[i] = rqs.at(0);
        m_subrequest_devices[i] = *comp_model_desc.device_it;
        if (comp_model_desc.replaced_by && m_use_function_pipelining) {
            m_funcall_pipeline[i].subrequest = rqs.at(1);
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

    // Preallocate input tensors
    LOG_INFO("Preallocating input tensors...");
    for (size_t i = 0; i < m_npuw_model->inputs().size(); i++) {
        const auto& port = m_npuw_model->inputs()[i];
        const auto& to_submodel = m_npuw_model->m_inputs_to_submodels_inputs.at(i);
        m_input_tensors.push_back(ov::get_tensor_impl(ov::Tensor(port.get_element_type(), port.get_shape())));
        m_port_to_tensor[port] = TensorStorage{m_input_tensors.back(), true};

        if (to_submodel != ov::npuw::CompiledModel::NO_LINK) {
            const auto& submodel_idx = to_submodel.first;
            m_port_to_subrequest_idx[port] = submodel_idx;
            m_reader_to_orig_port[to_submodel] = port;
            m_port_orig_to_sub[port] =
                get_real_subrequest(to_submodel.first)->get_compiled_model()->inputs()[to_submodel.second];
        } else {
            // Quick hack to support models with unused Parameters...
            m_port_to_subrequest_idx[port] = ov::npuw::JustInferRequest::INVALID_IDX;
        }
    } // for(inputs)
    // One more map to fill...
    for (auto&& it : m_npuw_model->m_param_subscribers) {
        const auto& prim_port = m_npuw_model->inputs()[it.first];
        for (auto&& to_submodel : it.second) {
            m_reader_to_orig_port[to_submodel] = prim_port;
            m_port_orig_to_sub[prim_port] =
                get_real_subrequest(to_submodel.first)->get_compiled_model()->inputs()[to_submodel.second];
        }
    }

    // Preallocate output tensors
    LOG_INFO("Preallocating output tensors...");
    for (size_t i = 0; i < m_npuw_model->outputs().size(); i++) {
        LOG_BLOCK();
        const auto& port = m_npuw_model->outputs()[i];
        LOG_INFO("Output " << i << " of " << m_npuw_model->outputs().size() << ": " << port);

        // FIXME: Yes, the CompiledModel::ToSubmodel == JustInferRequest::LinkFrom
        const auto& from_submodel = m_npuw_model->m_outputs_to_submodels_outputs.at(i);

        LOG_INFO("Produced by Subgraph[" << from_submodel.first << "] / " << from_submodel.second);
        auto funcall_result_iter = m_funcall_result.find(from_submodel);

        auto tensor = funcall_result_iter != m_funcall_result.end()
                          ? funcall_result_iter->second  // Function calls have their tensors allocated, so just use one
                          : ov::get_tensor_impl(ov::Tensor(port.get_element_type(), port.get_shape()));

        const auto& submodel_idx = from_submodel.first;
        m_port_to_subrequest_idx[port] = submodel_idx;
        m_port_orig_to_sub[port] =
            get_real_subrequest(from_submodel.first)->get_compiled_model()->outputs()[from_submodel.second];
        m_output_tensors.push_back(tensor);
        m_port_to_tensor[port] = TensorStorage{tensor, true};
    }
    connect_subrequests();

    // Build the parameter/result mapping {{{
    m_subrequests_gio.resize(m_subrequests.size());

    // Parameters: stage 1...
    for (size_t i = 0; i < m_npuw_model->inputs().size(); i++) {
        std::size_t sub_idx{}, in_idx{};
        std::tie(sub_idx, in_idx) = m_npuw_model->m_inputs_to_submodels_inputs.at(i);
        m_subrequests_gio.at(sub_idx).global_params[i] = in_idx;
    }  // for(inputs)

    // Parameters: stage 2...
    for (auto&& it : m_npuw_model->m_param_subscribers) {
        const auto param_idx = it.first;
        for (auto&& to_submodel : it.second) {
            std::size_t sub_idx{}, in_idx{};
            std::tie(sub_idx, in_idx) = to_submodel;
            m_subrequests_gio.at(sub_idx).global_params[param_idx] = in_idx;
        }
    }

    // Results
    for (size_t i = 0; i < m_npuw_model->outputs().size(); i++) {
        std::size_t sub_idx{}, out_idx{};
        std::tie(sub_idx, out_idx) = m_npuw_model->m_outputs_to_submodels_outputs.at(i);
        m_subrequests_gio.at(sub_idx).global_results[i] = out_idx;
    }
    // }}}
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

        LOG_DEBUG("Subgraph[" << subm_idx_from << "]/" << port_idx_from << " --> "
                              << "Subgraph[" << subm_idx_to << "]/" << port_idx_to);
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

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::JustInferRequest::query_state() const {
    std::vector<ov::SoPtr<ov::IVariableState>> variable_states = {};
    for (const auto& request : m_subrequests) {
        if (!request)  // optimized out
            continue;
        for (auto&& state : request->query_state()) {
            if (!state._so)
                state._so = request._so;
            variable_states.emplace_back(state);
        }
    }
    return variable_states;
}

std::vector<ov::ProfilingInfo> ov::npuw::JustInferRequest::get_profiling_info() const {
    std::vector<ov::ProfilingInfo> info;
    for (size_t i = 0; i < m_subrequests.size(); ++i) {
        if (!m_subrequests[i])  // optimized out
            continue;
        auto&& subreq_info = m_subrequests[i]->get_profiling_info();
        for (auto&& rec : subreq_info)
            rec.node_name = std::string("subgraph") + std::to_string(i) + ": " + rec.node_name;
        info.insert(info.end(), subreq_info.begin(), subreq_info.end());
    }
    return info;
}

void ov::npuw::JustInferRequest::prepare_for_infer() {
    LOG_DEBUG("Preparing to infer...");
    LOG_BLOCK();

    // If funcall pipelining is enabled, prefill the function "heads"
    // with constant arguments
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
    LOG_DEBUG("Binding parameters for Subgraph[" << idx << "]");
    LOG_BLOCK();

    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    const auto real_idx = comp_model_desc.replaced_by.value_or(idx);
    if (real_idx != idx && m_use_function_pipelining) {
        LOG_DEBUG("This is a function and pipelining enabled - will be done in-pipeline, skipping");
        return;
    }

    if (real_idx != idx) {
        // Don't do here - function call will take parameter
        // itself. Note it may be implemented more efficently
        // than now (and in some cases, parameter can be pre-set)
        LOG_DEBUG("Skipping this too now - function will do it for itself");
        // FIXME ^^^ this is the old branch, it needs to be updated
        // the logic needs to move there.
        return;
    }

    const auto &iodesc = m_subrequests_gio.at(idx);
    for (auto &&it : iodesc.global_params) {
        std::size_t param_idx{}, sub_in_idx{};
        std::tie(param_idx, sub_in_idx) = it;
        const auto& g_port = m_npuw_model->inputs()[param_idx];
        const auto& s_port = m_subrequests[idx]->get_inputs()[sub_in_idx];
        m_subrequests[idx]->set_tensor(s_port, m_port_to_tensor.at(g_port).tensor);
    }

    LOG_DEBUG("Done");
}

void ov::npuw::JustInferRequest::bind_global_results(std::size_t idx) {
    LOG_DEBUG("Binding results for Subgraph[" << idx << "]");
    LOG_BLOCK();

    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    const auto real_idx = comp_model_desc.replaced_by.value_or(idx);
    if (real_idx != idx) {
        // Don't do here - function call will take parameter
        // itself. Note it may be implemented more efficently
        // than now (and in some cases, parameter can be pre-set)
        LOG_DEBUG("Skipping this too now - function will do it for itself");
        // FIXME ^^^ this is the old branch, it needs to be updated
        // the logic needs to move there.
        return;
    }

    const auto &iodesc = m_subrequests_gio.at(idx);
    for (auto &&it : iodesc.global_results) {
        std::size_t result_idx{}, sub_out_idx{};
        std::tie(result_idx, sub_out_idx) = it;
        const auto& g_port = m_npuw_model->outputs()[result_idx];
        const auto& s_port = m_subrequests[idx]->get_outputs()[sub_out_idx];
        m_subrequests[idx]->set_tensor(s_port, m_port_to_tensor.at(g_port).tensor);
    }

    LOG_DEBUG("Done");
}

void ov::npuw::JustInferRequest::function_prologue(std::size_t idx) {
    LOG_DEBUG("Preparing Subgraph[" << idx << "] funcall prologue");
    LOG_BLOCK();

    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];

    NPUW_ASSERT(comp_model_desc.replaced_by);
    const auto real_idx = comp_model_desc.replaced_by.value();
    auto& func_desc = m_npuw_model->m_compiled_submodels[real_idx];

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

            if (!m_npuw_model->m_compiled_submodels[prod_idx].replaced_by) {
                // Producer is a normal model -> take its tensor directly
                const auto& oport = m_npuw_model->m_compiled_submodels[prod_idx].compiled_model->outputs()[prod_port];
                m_subrequests[real_idx]->set_tensor(iport, m_subrequests[prod_idx]->get_tensor(oport));
            } else {
                // Producer is a function - maybe the same as we're calling now.
                // Take its tensor from the storage
                m_subrequests[real_idx]->set_tensor(iport, m_funcall_result.at({prod_idx, prod_port}));
            }
        } else {
            // Take input from the associated parameter.
            LOG_DEBUG("Finding the original model port for " << idx << "/" << i << "...");
            auto& orig_port = m_reader_to_orig_port.at({idx, i});
            m_subrequests[real_idx]->set_tensor(iport, m_port_to_tensor.at(orig_port).tensor);
        }
    }  // for(param_base)

    // 2. Unpack the function closure -- right here, if pipelining if not enabled.
    // If it is enabled, the flow is a little bit different - see run_subrequest_for_success()
    // for details.
    if (!m_use_function_pipelining) {
        LOG_DEBUG("Unpacking closures...");
        LOG_BLOCK();
        unpack_closure(idx, m_subrequests[real_idx]);
    }

    // 3. Tell the function which results to produce (this time)
    for (std::size_t i = 0; i < func_desc.compiled_model->outputs().size(); i++) {
        LOG_DEBUG("Binding result[" << i << "]...");
        auto& oport = func_desc.compiled_model->outputs()[i];
        m_subrequests[real_idx]->set_tensor(oport, m_funcall_result.at({idx, i}));
    }
    LOG_DEBUG("Done");
}

void ov::npuw::JustInferRequest::unpack_closure(std::size_t idx, RqPtr request) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];

    NPUW_ASSERT(comp_model_desc.replaced_by);
    const auto real_idx = comp_model_desc.replaced_by.value();
    auto& func_desc = m_npuw_model->m_compiled_submodels[real_idx];

    // Bind extra parameters from the function's closure
    // First, do easy things & delay heavy stuff
    std::vector<std::size_t> closure_unpack_required;
    for (std::size_t cidx = 0u; cidx < comp_model_desc.closure.size(); cidx++) {
        auto& closure = comp_model_desc.closure[cidx];

        const auto closure_param_id = comp_model_desc.param_base + cidx;
        auto& iport = func_desc.compiled_model->inputs()[closure_param_id];
        auto clparam = request->get_tensor(iport);
        if (closure.get_element_type() != clparam->get_element_type()) {
            // Remember where the unpack is required
            closure_unpack_required.push_back(cidx);
        } else {
            // Easy case, just set one to another. Copy_to is also possible
            // and even may be preferrable for some devices, like this:
            // ```ov::get_tensor_impl(closure)->copy_to(clparam._ptr);'''
            request->set_tensor(iport, ov::get_tensor_impl(closure));
        }
    }  // for(closure)
       // m_ms_unpack += ov::npuw::perf::ms_to_run([&](){
       //    ov::parallel_for(closure_unpack_required.size(), [&](std::size_t j) {
    for (std::size_t j = 0; j != closure_unpack_required.size(); j++) {
        // NB: No need to protect anything here as containers are all
        // preallocated and we only access elements under particular (thread
        // -local) indices.
        auto cidx = closure_unpack_required[j];

        // FIXME: zerops are stored with absolute indexing, this needs to be aligned
        auto& closure = comp_model_desc.closure[cidx];

        const auto closure_param_id = comp_model_desc.param_base + cidx;
        auto& iport = func_desc.compiled_model->inputs()[closure_param_id];
        auto clparam = request->get_tensor(iport);

        if (!comp_model_desc.scales.empty() && comp_model_desc.scales[cidx] && comp_model_desc.zerops[cidx]) {
            // Unpacking this weight requires scaling with zero points...
            ov::npuw::util::unpack(ov::get_tensor_impl(closure),
                                   ov::get_tensor_impl(comp_model_desc.zerops[cidx]),
                                   ov::get_tensor_impl(comp_model_desc.scales[cidx]),
                                   clparam);
        } else if (!comp_model_desc.scales.empty() && comp_model_desc.scales[cidx]) {
            // Unpacking this weight requires scaling
            ov::npuw::util::unpack(ov::get_tensor_impl(closure),
                                   ov::get_tensor_impl(comp_model_desc.scales[cidx]),
                                   clparam);
        } else {
            // Unpacking this weight doesn't require scaling
            ov::npuw::util::unpack(ov::get_tensor_impl(closure), clparam);
        }
    }
    //}); // ov_parallel_for
    // }); // ms_to_run
}

void ov::npuw::JustInferRequest::recreate_subrequests(std::size_t idx) {
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    auto real_idx = comp_model_desc.replaced_by.value_or(idx);

    auto new_rqs = create_infer_requests(idx, m_use_function_pipelining ? 2 : 1);

    // NB: Regardless if this subrequest was a function call
    // or not, always use the real_idx here - for regular
    // subrequests, real_id == idx, but for function calls it
    // is critical here to update the function body, not the
    // function calls (which are left empty now in the vector)
    m_subrequests[real_idx] = new_rqs.at(0);
    if (comp_model_desc.replaced_by && m_use_function_pipelining) {
        m_funcall_pipeline[real_idx].subrequest = new_rqs.at(1);
    }
    // After an infer request is recreated, the internal cross-request
    // connections should be re-established (in/out tensors reset properly)
    // Note: these two proceduers do the full I/O reset procedure what's
    // overkill - only affected subrequest(s) could be updated instead,
    // but it is a more complex thing and can be implemented separately
    connect_subrequests();
    m_subrequest_devices[idx] = *comp_model_desc.device_it;
}

void ov::npuw::JustInferRequest::run_subrequest_for_success(std::size_t idx, bool& failover) {
    failover = false;
    auto& comp_model_desc = m_npuw_model->m_compiled_submodels[idx];
    auto real_idx = comp_model_desc.replaced_by.value_or(idx);

    // Infer is also fail-safe...
    bool job_done = false;
    bool dump_in = false;
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

        // If subrequests refers to the parent model's global
        // Parameter tensors, bind/copy them here. The in-graph
        // connections are managed automatically with inference
        // chaining. Functions are the exception, see
        // function_prologue() below.
        bind_global_parameters(idx);

        // Same applies to Result tensors
        bind_global_results(idx);

        if (comp_model_desc.replaced_by) {
            function_prologue(idx);
        }
        if (!dump_in) {
            dump_in = true;
            dump_input_tensors(idx);
        }
        try {
            LOG_DEBUG("Trying to run subrequest[" << idx << "]...");
            if (m_use_function_pipelining && comp_model_desc.replaced_by) {
                // Start THIS request -- everything is ready for it at this point
                m_subrequests[real_idx]->start_async();
                if (m_funcall_pipeline[idx].next) {
                    const auto next_id = m_funcall_pipeline[idx].next.value();
                    // Prepare the NEXT request while this one is running
                    LOG_DEBUG("Unpacking closures for the NEXT subrequest[" << next_id << "]...");
                    LOG_BLOCK();
                    unpack_closure(next_id, m_funcall_pipeline[real_idx].subrequest);
                }

                // Wait for completion of this one. Once completed,
                // its' brother was already updated all the right
                // closure tensors. Swapping the subrequests will
                // happen in the very end to allow dumping the proper
                // tensors, if needed (see job_done check).
                m_subrequests[real_idx]->wait();
            } else {
                m_subrequests[real_idx]->infer();
            }
            job_done = true;
        } catch (const std::exception& ex) {
            LOG_ERROR("Subgraph [" << idx << "] - FAILED to run infer request:" << std::endl << ex.what());
            should_recreate = true;
        } catch (...) {
            LOG_ERROR("Subgraph [" << idx << "] - FAILED to run infer request: REASON UNKNOWN");
            should_recreate = true;
        }
        if (should_recreate) {
            failover = true;
            LOG_INFO("- Trying next device...");

            // Altering iterators here!! Contracts should be changed!
            comp_model_desc.device_it++;
            if (!m_npuw_model->compile_for_success(real_idx)) {
                OPENVINO_THROW("Failed to compile. No more devices are left!");
            }
            recreate_subrequests(idx);
        }
    }  // while(job_done)

    if (job_done) {
        dump_output_tensors(idx);  // FIXME: Called here unconditionally, need to refactor
        if (m_use_function_pipelining && m_funcall_pipeline[idx].next) {
            // Swap the next (pipelined, semi-prepared) infer request in the chain
            // with the default (to be accessed next) one.
            std::swap(m_subrequests[real_idx], m_funcall_pipeline[real_idx].subrequest);
        }
    }
}

void ov::npuw::JustInferRequest::subscribe_subrequest(std::size_t idx, Completed cb) {
    get_real_subrequest(idx)->set_callback(cb);
}

void ov::npuw::JustInferRequest::complete_subrequest(std::size_t idx) {
    // do nothing here
}

void ov::npuw::JustInferRequest::cancel_subrequest(std::size_t idx) {
    m_subrequests[idx]->cancel();
}

std::size_t ov::npuw::JustInferRequest::total_subrequests() const {
    return m_subrequests.size();
}

bool ov::npuw::JustInferRequest::supports_async_pipeline() const {
    return false;
}

void ov::npuw::JustInferRequest::update_subrequest_links(std::size_t) {
    connect_subrequests();
}
