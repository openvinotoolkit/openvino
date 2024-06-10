// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "compiled_model.hpp"

#include <memory>
#include <iostream>

#include "just_sync_infer_request.hpp"

#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/core/parallel.hpp"
#include "plugin.hpp"
#include "properties.hpp"
#include "npu_private_properties.hpp"
#include "logging.hpp"
#include "config.hpp"
#include "util.hpp"
#include "accuracy/comparator.hpp"

// required for get_properties_per_device()
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"


namespace ov {
namespace npuw {
namespace {
ov::npuw::DeviceProperties get_properties_per_device(const std::shared_ptr<const ov::IPlugin>& plugin,
                                                     const std::string& device_priorities,
                                                     const ov::AnyMap& properties) {
    auto core = plugin->get_core();
    auto device_names = ov::DeviceIDParser::get_hetero_devices(device_priorities);
    DeviceProperties device_properties;
    for (const auto& device_name : device_names) {
        auto properties_it = device_properties.find(device_name);
        if (device_properties.end() == properties_it) {
            device_properties[device_name] = core->get_supported_property(device_name, properties);

            // Do extra handling for the private NPU options
            if (ov::npuw::util::starts_with(device_name, "NPU")) {
                for (auto &&opt : properties) {
                    if (ov::npuw::util::starts_with(opt.first, "NPU")) {
                        device_properties[device_name][opt.first] = opt.second;
                    }
                }
            } // if(NPU)
        }
    }
    return device_properties;
}
} // anonymous namespace
} // namespace npuw
} // namespace ov

ov::npuw::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                       const std::shared_ptr<const ov::IPlugin>& plugin,
                                       const Configuration& cfg)
    : ov::ICompiledModel(model, plugin),
      m_cfg(cfg),
      m_name(model->get_friendly_name()),
      m_loaded_from_cache(false) {
    const std::string dev_list_str = ov::npuw::get_env({
            "OPENVINO_NPUW_DEVICES_" + m_name,
            "OPENVINO_NPUW_DEVICES"
        }, "NPU,CPU");
    m_dev_list = ov::DeviceIDParser::get_hetero_devices(dev_list_str);
    m_meta_devices = ov::npuw::get_properties_per_device(plugin,
                                                         dev_list_str,
                                                         m_cfg.get_device_properties());

    bool dumpDotFile = m_cfg.dump_graph;
    if (std::getenv("OPENVINO_NPUW_VISUALIZE"))
        dumpDotFile = true;

    const auto* acc_check_opt = ov::npuw::get_env({
        "OPENVINO_NPUW_ACC_CHECK_" + m_name,
        "OPENVINO_NPUW_ACC_CHECK"});
    if (acc_check_opt != nullptr && acc_check_opt == std::string("YES")) {
        double threshold = 0.01;
        const char* threshold_opt = ov::npuw::get_env({
            "OPENVINO_NPUW_ACC_THRESH_" + m_name,
            "OPENVINO_NPUW_ACC_THRESH"});
        if (threshold_opt != nullptr) {
            threshold = std::atof(threshold_opt);
        }

        m_acc_check = metrics::NRMSE(threshold);
        m_ref_device = ov::npuw::get_env({"OPENVINO_NPUW_ACC_DEVICE"}, "CPU");
        LOG_INFO("Accuracy check is enabled.");
    }

    LOG_INFO("*** Original model ***");
    const auto& orig_parameters = model->get_parameters();
    { LOG_BLOCK(); for (auto &&p : orig_parameters) LOG_INFO(p); }
    const auto& orig_results = model->get_results();
    { LOG_BLOCK(); for (auto &&r: orig_results) LOG_INFO(r); }

    auto partitioning = getPartitioning(model);
    m_total_stat.gflops = partitioning.total_gflops;
    m_total_stat.ops = partitioning.total_ops;
    const std::vector<ov::npuw::Subgraph>& orderedSubgraphs = partitioning.subgraphs;
    // if (dumpDotFile) {
    //     std::map<std::string, int> map_id;
    //     for (const auto& v : subgraphIds) {
    //         map_id.emplace(v.first->get_friendly_name(), v.second);
    //     }
    //     ov::npuw::debug::dump_subgraphs(model, queryNetworkResult, map_id);
    // }

    // Prepare mapping between original inputs/outputs and compiled
    // submodels inputs/outputs. Example:
    // original input 0 -> submodel 0 input 0,
    // original input 1 -> submodel 1 input 0,
    // original output 0 -> submodel 1 output 0.
    //
    // Note that several partitioning schemes can result in multiple
    // submodels reading from the same parameter. Unfortunately, the
    // OpenVINO plugin model assumes there's just one tensor for this
    // case, but in fact there's many. See how this case is handled
    // below.
    //
    // Also, now some subgraphs may be (reusable) functions.
    // If a function reads a parameter, it can be be set only once
    // before the inference.
    m_inputs_to_submodels_inputs.resize(orig_parameters.size(), NO_LINK);
    m_outputs_to_submodels_outputs.resize(orig_results.size(), NO_LINK);

    for (size_t id = 0; id < orderedSubgraphs.size(); id++) {
        LOG_INFO("Subgraph[" << id << "]");
        LOG_BLOCK();
        if (orderedSubgraphs[id]._optimized_out) {
            LOG_INFO("OPTIMIZED OUT");
            continue;
        }
        auto process_params = [&](const ov::ParameterVector &_parameters) {
            for (size_t i = 0; i < _parameters.size(); i++) {
                LOG_INFO(_parameters[i]);
                for (size_t j = 0; j < orig_parameters.size(); j++) {
                    if (_parameters[i] == orig_parameters[j]) {
                        LOG_BLOCK();
                        LOG_INFO("MATCHED WITH " << orig_parameters[j]);
                        if (NO_LINK == m_inputs_to_submodels_inputs[j]) {
                            // Easy case, first come first served
                            LOG_INFO("Map this parameter directly");
                            m_inputs_to_submodels_inputs[j] = ToSubmodel{id, i};
                        } else {
                            // There's multiple subgraphs reading from the same Parameter.
                            // Record them here, see how this list is later used in the
                            // sync_infer_request.cpp
                            LOG_INFO("Subscribe this parameter to tensor");
                            m_param_subscribers[j].push_back(ToSubmodel{id, i});
                        } // if(NO_LINK)
                        // Regardless of the order, if the reader is a function,
                        // remember that to avoid confusion in function prologue
                    } // if(param == orig_param)
                } // for(orig_params)
            } // for(subgraph_params)
        };
        auto process_results = [&](const ov::ResultVector &_results) {
            for (size_t i = 0; i < _results.size(); i++) {
                if (!_results[i]) {
                    // See the rearrange_to_function_protocol<>()...
                    continue;
                }
                LOG_INFO(_results[i]);
                for (size_t j = 0; j < orig_results.size(); j++) {
                    if (_results[i] == orig_results[j]) {
                        // FIXME: There may be a problem, see below (!)
                        LOG_BLOCK();
                        LOG_INFO("MATCHED WITH " << orig_results[j]);
                        m_outputs_to_submodels_outputs[j] = {id, i};
                    }
                }
            } // for(results)
        };
        // Sheer ugliness here again
        if (orderedSubgraphs[id]._funcall.empty()) {
            process_params(orderedSubgraphs[id]._parameters);
            process_results(orderedSubgraphs[id]._results);
        } else {
            auto &f = partitioning.functions.at(orderedSubgraphs[id]._funcall);
            // Don't use f but match via the original parameters to generate indices
            process_params(orderedSubgraphs[id]._parameters);
            process_results(orderedSubgraphs[id]._results);

            // (!) Problem here: when functions are folded, in the
            // KVcache-like scenarios (where every function call
            // produces its own result), its result(s) will be matched
            // with _all_ instances of the result (see pic).
        }
    } // for(ordered_subgraphs)
    // NOTE(dm): there's a better way to do it, like we do in G-API backends.

    // Store mapping between manually splitted inputs/outputs
    // to connect tensors between compiled submodels
    m_submodels_input_to_prev_output = partitioning.input_to_prev_output;

    // Before the compilation:
    // - initialize the OV models
    // - dump the subgraphs, if necessary
    std::map<std::string, std::size_t> compiledFunctions;
    m_compiled_submodels.resize(orderedSubgraphs.size());
    LOG_INFO("Creating submodels...");
    for (std::size_t id = 0u; id < orderedSubgraphs.size(); id++) {
        LOG_BLOCK();

        const auto& subgraph = orderedSubgraphs[id];
        m_compiled_submodels[id].stat.gflops = subgraph._gflops;
        m_compiled_submodels[id].stat.ops = subgraph._ops;

        if (subgraph._optimized_out) {
            // FIXME: filter out the optimized out subgraph from this process early,
            // only dispatch real compilation to the threads
            LOG_INFO("Skipping Subgraph[" << id << "] - optimized out!");
            continue;
        }
        LOG_INFO("Creating Subgraph[" << id << "]...");
        if (subgraph._funcall.empty()) {
            // NOT a function call - an easy case!
            m_compiled_submodels[id].model =
                std::make_shared<ov::Model>(subgraph._results,
                                            subgraph._sinks,
                                            subgraph._parameters,
                                            m_name + '_' + std::to_string(id));
        } else {
            LOG_BLOCK();
            auto &fcn_template = partitioning.functions.at(subgraph._funcall);
            auto compiled_fcn_iter = compiledFunctions.find(subgraph._funcall);
            if (compiled_fcn_iter == compiledFunctions.end()) {
                // A function call: store the model for function call only once...
                compiledFunctions.insert({subgraph._funcall, id});
                m_compiled_submodels[id].model = fcn_template._model;
                m_compiled_submodels[id].replaced_by = id; // FIXME: UGLY
                LOG_INFO("Subgraph[" << id << "] is a function body for "
                         << subgraph._funcall);
            } else {
                // ...and refer to it in other calls
                m_compiled_submodels[id].replaced_by = compiled_fcn_iter->second;
                LOG_INFO("Subgraph[" << id << "] is a function call to ["
                         << compiled_fcn_iter->second << "]");
            }
            m_compiled_submodels[id].param_base = fcn_template._param_offset;
            m_compiled_submodels[id].closure = subgraph._closure;
            m_compiled_submodels[id].scales  = subgraph._scales;
            m_compiled_submodels[id].zerops  = subgraph._zerops;
        } // if(!funcall)

        if (!m_compiled_submodels[id].model && !m_compiled_submodels[id].replaced_by) {
            OPENVINO_THROW("Fatal: submodel ", id, " is neither a model nor a function call!");
        }
        const std::size_t real_id = m_compiled_submodels[id].replaced_by.value_or(id);

        // FIXME: a hotfix for a crash where we have too many names in submodel's output
        // Do it just once if that's a function
        if (real_id == id) {
            remove_long_output_names(m_compiled_submodels[real_id].model);
        }

        const auto *dump_opt = ov::npuw::get_env({
                "OPENVINO_NPUW_DUMP_SUB_" + m_name + "_" + std::to_string(id),
                "OPENVINO_NPUW_DUMP_SUB_" + m_name,
                "OPENVINO_NPUW_DUMP_SUB"});
        if (dump_opt && dump_opt == std::string("YES")) {
            LOG_INFO("Dumping Subgraph[" << id << "]");
            LOG_BLOCK();
            if (real_id != id) {
                LOG_INFO("NOTE: Dumping Subgraph[" << real_id << "]"
                         << " as it is a function body for Subgraph[" << id << "]");
            }
            const auto model_to_dump = m_compiled_submodels[real_id].model;
            std::string model_dump_path = m_name
                + "_" + ov::npuw::util::fmt(id, m_compiled_submodels.size())
                + (subgraph._funcall.empty() ? "" : "_" + subgraph._funcall)
                + ".xml";
            ov::save_model(model_to_dump, model_dump_path);
            LOG_INFO("Wrote " << model_dump_path);
            // Note: keep here naming as it would be the subgraph
        } // if(dump)
    } // for(orderedSubgraphs)

    // Compile submodels. Some of them can be functions: track which model will be
    // used as function(s): function name -> index of the compiled subgraph
    auto compile = [&](size_t id) {
        const auto& subgraph = orderedSubgraphs[id];
        if (subgraph._optimized_out) {
            return;
        }

        const std::size_t real_id = m_compiled_submodels[id].replaced_by.value_or(id);
        if (!orderedSubgraphs[real_id]._avoid_list.empty()) {
            const auto devices_to_avoid = ov::DeviceIDParser::get_hetero_devices
                (orderedSubgraphs[real_id]._avoid_list);
            for (auto &&d : devices_to_avoid) {
                m_compiled_submodels[real_id].devices_to_avoid.insert(std::move(d));
            }
        }
        m_compiled_submodels[real_id].device_it = m_dev_list.cbegin();

        const auto *forced_device_opt = ov::npuw::get_env({
                "OPENVINO_NPUW_DEVICE_" + m_name + "_" + std::to_string(id)
            });
        if (forced_device_opt) {
            std::string forced_device(forced_device_opt);
            auto forced_dev_it = std::find(m_dev_list.begin(), m_dev_list.end(), forced_device);
            if (forced_dev_it == m_dev_list.end()) {
                LOG_WARN("Target device for Subgraph[" << id << "] was set to "
                         << forced_device << ", but was not found in the device list: "
                         << "[" << dev_list_str << "] -- ignoring");
            } else {
                // FIXME: This is not really a device enforcement as the fallback
                // procedure will be in place still
                LOG_INFO("Force Subgraph[" << id << "] target device to " << *forced_dev_it);
                m_compiled_submodels[real_id].device_it = forced_dev_it;
            }
        } // if(forced_device_opt)

        LOG_INFO("Compiling Subgraph[" << id << "]: "
                 << m_compiled_submodels[real_id].model->get_friendly_name() << "...");
        if (!compile_for_success(id)) {
            OPENVINO_THROW("Failed to compile ",
                           m_compiled_submodels[real_id].model->get_friendly_name(),
                           " for all devices in [", dev_list_str, "]");
        }

        if (m_acc_check) {
            if (submodel_device(real_id) != m_ref_device) {
                LOG_INFO("Compile Subgraph[" << real_id << "] for reference device: "
                         << m_ref_device << ".");
                LOG_BLOCK();
                m_compiled_submodels.at(real_id).ref_compiled_model =
                    compile_submodel(m_compiled_submodels.at(real_id).model, m_ref_device);
                LOG_INFO("Done (reference)");
            } else {
                LOG_INFO("Skip compilation of submodel[" << real_id << "] for reference device: "
                         << m_ref_device << ", as original submodel[" << real_id <<
                         "] has been already compiled for it.");
            }
        }
    }; // compile

    // Parallel compilation is unstable so is disabled by default.
    const auto *par_opt = ov::npuw::get_env({
            "OPENVINO_NPUW_PARC_" + m_name,
            "OPENVINO_NPUW_PARC"});
    if (par_opt && par_opt == std::string("YES")) {
        ov::parallel_for(orderedSubgraphs.size(), compile);
    } else {
        // TODO: Introduce npuw::serial(i, f) instead
        for (std::size_t i = 0u; i < orderedSubgraphs.size(); i++) {
            compile(i);
        }
    }

    // Print stats report when possible
    {
        LOG_INFO("Initial device distribution:");
        LOG_BLOCK();
        log_device_dist();
    }

    m_finalized = true;
    reset_io();
}

void ov::npuw::CompiledModel::remove_long_output_names(const std::shared_ptr<ov::Model>& model) {
    NPUW_ASSERT(model.get() != nullptr);
    for (auto& output : model->outputs()) {
        auto tensor_names = output.get_tensor().get_names();
        if (tensor_names.size() > 32) { // maximum supported
            output.get_tensor().set_names({});
            LOG_INFO("Removed output tensor names for " << model->get_friendly_name());
            LOG_BLOCK();
        }
    }
}

void ov::npuw::CompiledModel::reset_io() {
    // Restore inputs/outputs from compiled submodels
    // FIXME: this method is also called from IBaseInferReqeust::create_infer_request
    // which is called in the CompiledModel(). So this method executes even before it
    // is called for the right thing from the ctor directly.
    if (!m_finalized)
        return; // avoid getting called before it is really the time

    // Don't be like HETERO here - don't override the inputs/outputs(),
    // as the ICompiledModel already creates one for us.
    // Instead, remember the mapping from the original CompiledModel::input/output ports
    // to the Subgraph's input/output ports.

    LOG_INFO("*** Partition graph ***");
    int idx_in = 0, idx_out = 0; // FIXME: use indexed()
    for (const auto& to_submodel : m_inputs_to_submodels_inputs) {
        LOG_BLOCK();
        if (to_submodel == NO_LINK) {
            LOG_WARN("Input (Parameter) " << inputs()[idx_in] <<
                     " is not used by any subgraph. It happens sometimes,"
                     " but better check your model");
            idx_in++; // FIXME: PLEASE use indexed() here
            continue;
        }
        const auto& submodel_idx = to_submodel.first;
        const auto& input_idx    = to_submodel.second;
        LOG_INFO("Input (Parameter) " << inputs()[idx_in]
                 << " from Subgraph[" << submodel_idx << "]/" << input_idx);
        idx_in++;
    }
    for (const auto& to_submodel : m_outputs_to_submodels_outputs) {
        NPUW_ASSERT(to_submodel != NO_LINK);
        LOG_BLOCK(); // in fact, to_submodel <is> from_submodel here, but who cares
        const auto& submodel_idx = to_submodel.first;
        const auto& output_idx   = to_submodel.second;
        LOG_INFO("Output (Result) " << outputs()[idx_out]
                 << " from Subgraph[" << submodel_idx << "]/" << output_idx);
        idx_out++;
    }
}

bool ov::npuw::CompiledModel::compile_for_success(std::size_t id) {
    // Assume device_it is some always-valid starting point.
    // FIXME: And who guarantees it? Abstraction leaks are everywhere
    if (m_compiled_submodels[id].replaced_by
        && m_compiled_submodels[id].replaced_by != id) {
        LOG_BLOCK();
        LOG_INFO("Skip compilation for Subgraph[" << id << "] "
                 "as it was already compiled (function)");
        return true;
    }

    for (auto iter = m_compiled_submodels[id].device_it; iter != m_dev_list.cend(); ++iter) {
        LOG_BLOCK();
        const auto &device_name = *iter;
        if (m_compiled_submodels[id].devices_to_avoid.count(device_name) > 0) {
            LOG_INFO(device_name << " was found in the 'Avoid' list for this subgraph, skipping...");
        } else if (compile_for_device(id, device_name)) {
            m_compiled_submodels[id].device_it = iter;
            return true; // success!
        }
    }
    return false;
}

bool ov::npuw::CompiledModel::compile_for_device(std::size_t id, const std::string &device_to_try) {
    // The API of this method is ugly but it is what it is
    // NOTE(dm): Hetero plugin disables caching here, but we'll keep this to be done
    // at the individual plugins level.
    auto plugin = get_npuw_plugin();

    LOG_INFO("Trying to compile for " << device_to_try << "...");

    // Only function bodies can reach this point.
    // compile_for_device() behavior is not specified for funcalls.
    NPUW_ASSERT(m_compiled_submodels[id].replaced_by.value_or(id) == id);

    // NPU plugin is known to crash when model has no inputs, so even
    // try..catch doesn't work. Ideally such models should be
    // eliminated with constant folding, but with offline partitioning
    // it is not possible + not clear what to do with dynamic shapes.
    // So for now, skip this case explicitly.
    if (npuw::util::starts_with(device_to_try, "NPU") &&
        m_compiled_submodels[id].model->inputs().empty()) {
        LOG_INFO("Avoid compilation for "
                 << device_to_try
                 << " as the model should be constant-folded");
        dump_on_fail(id, device_to_try, "Avoided due to workaround");
        return false;
    }

    try {
        m_compiled_submodels[id].compiled_model =
            compile_submodel(m_compiled_submodels[id].model, device_to_try);
    } catch (const std::exception& ex) {
        LOG_ERROR("Subgraph ["<< id << "] Failed to compile: " << std::endl << ex.what());
        dump_on_fail(id, device_to_try, ex.what());
        return false;
    } catch (...) {
        LOG_ERROR("Subgraph ["<< id << "] Failed to compile: Unknown error");
        dump_on_fail(id, device_to_try, "Unknown error");
        return false;
    }
    // Reached this point - all ok, stop the search
    LOG_INFO("Done (" << device_to_try << ")");
    return true;
}

ov::SoPtr<ov::ICompiledModel>
ov::npuw::CompiledModel::compile_submodel(const std::shared_ptr<ov::Model>& submodel,
                                          const std::string& device) {
    auto plugin = get_npuw_plugin();
    auto core = plugin->get_core();
    // set exclusive_async_requests in case when model is split
    // NOTE(dm): Not sure if it is required for the NPUW plugin, but likely it is
    auto& device_config = m_meta_devices[device];
    if (m_compiled_submodels.size() > 1) {
        auto supported_internal_properties =
                plugin->get_core()->get_property(device, ov::internal::supported_properties);
        if (std::find(supported_internal_properties.begin(), supported_internal_properties.end(),
                      ov::internal::exclusive_async_requests) != supported_internal_properties.end()) {
            // adds property if it is not set yet
            device_config.insert(ov::internal::exclusive_async_requests(true));
        }
    }  // if(subgraphs > 1)
    if (ov::npuw::util::starts_with(device, "NPU")) {
        device_config.insert(ov::intel_npu::from_npuw("YES"));
    }
    return core->compile_model(submodel, device, device_config);
}

void ov::npuw::CompiledModel::dump_on_fail(std::size_t id,
                                           const std::string &device_to_try,
                                           const char *extra) {
    const auto *dump_on_fail_opt = ov::npuw::get_env({
            "OPENVINO_NPUW_DUMP_ON_FAIL_" + m_compiled_submodels[id].model->get_friendly_name(),
            "OPENVINO_NPUW_DUMP_ON_FAIL_" + m_name,
            "OPENVINO_NPUW_DUMP_ON_FAIL"});
    if (dump_on_fail_opt && dump_on_fail_opt == std::string("YES")) {
        ov::npuw::dump_failure(m_compiled_submodels[id].model, device_to_try, extra);
    }
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::CompiledModel::create_just_sync_infer_request() {
    auto this_sptr = std::static_pointer_cast<ov::npuw::CompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::JustInferRequest>(this_sptr);
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::CompiledModel::create_sync_infer_request() const {
    // Synchronous infer request implementation may vary based on the
    // selected strategy
    auto *non_const_this = const_cast<ov::npuw::CompiledModel*>(this); // because of const in API
    return non_const_this->create_just_sync_infer_request();
}

std::shared_ptr<ov::IAsyncInferRequest> ov::npuw::CompiledModel::create_infer_request() const {
    auto internal_request = create_sync_infer_request();
    return std::make_shared<ov::IAsyncInferRequest>(internal_request, get_task_executor(), get_callback_executor());
}

void ov::npuw::CompiledModel::set_property(const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ov::Model> ov::npuw::CompiledModel::get_runtime_model() const {
    // NOTE(dm): See hetero plugin implementation if need to bring this method back
    // (that code should work as-is)
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ::intel_npu::Plugin> ov::npuw::CompiledModel::get_npuw_plugin() const {
    auto plugin = get_plugin();
    OPENVINO_ASSERT(plugin);
    auto npuw_plugin = std::dynamic_pointer_cast<const ::intel_npu::Plugin>(plugin);
    OPENVINO_ASSERT(npuw_plugin);
    return npuw_plugin;
}

ov::Any ov::npuw::CompiledModel::get_property(const std::string& name) const {
    // NOTE(dm): This code is kept intact, probably some NPUW-specific
    // tuning could be done here (e.g. derive the values from the main
    // NPU plugin?)

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto& add_ro_properties = [](const std::string& name, std::vector<ov::PropertyName>& properties) {
        properties.emplace_back(ov::PropertyName{name, ov::PropertyMutability::RO});
    };
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::model_name,
                                                    ov::optimal_number_of_infer_requests,
                                                    ov::execution_devices,
                                                    ov::loaded_from_cache,
                                                    ov::npuw::number_of_submodels};
        return ro_properties;
    };
    const auto& to_string_vector = [](const std::vector<ov::PropertyName>& properties) {
        std::vector<std::string> ret;
        for (const auto& property : properties) {
            ret.emplace_back(property);
        }
        return ret;
    };

    if (ov::supported_properties == name) {
        auto supported_properties = default_ro_properties();
        add_ro_properties(ov::supported_properties.name(), supported_properties);
        add_ro_properties(ov::device::properties.name(), supported_properties);
        add_ro_properties(ov::device::priorities.name(), supported_properties);
        return decltype(ov::supported_properties)::value_type(supported_properties);
    // } else if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_METRICS) == name) {
    //     auto metrics = default_ro_properties();
    //     add_ro_properties(METRIC_KEY(SUPPORTED_METRICS), metrics);
    //     add_ro_properties(METRIC_KEY(SUPPORTED_CONFIG_KEYS), metrics);
    //     return to_string_vector(metrics);
    // } else if (EXEC_NETWORK_METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
    //     return to_string_vector(m_cfg.get_supported());
    } else if (ov::device::properties == name) {
        ov::AnyMap all_devices = {};
        for (auto i = 0; i < m_compiled_submodels.size(); ++i) {
            auto comp_model_desc = m_compiled_submodels[i];
            if (!comp_model_desc.compiled_model) // Handle if optimized out
                continue;
            ov::AnyMap device_properties = {};
            if (all_devices.count(submodel_device(i)) == 0) {
                auto device_supported_props =
                    comp_model_desc.compiled_model->get_property(ov::supported_properties.name());
                for (auto&& property_name : device_supported_props.as<std::vector<ov::PropertyName>>())
                    device_properties[property_name] = comp_model_desc.compiled_model->get_property(property_name);
                all_devices[submodel_device(i)] = device_properties;
            }
        }
        return all_devices;
    } else if (ov::model_name == name) {
        return decltype(ov::model_name)::value_type(m_name);
    } else if (ov::loaded_from_cache == name) {
        return decltype(ov::loaded_from_cache)::value_type{m_loaded_from_cache};
    } else if (ov::optimal_number_of_infer_requests == name) {
        unsigned int value = 0u;
        for (const auto& comp_model_desc : m_compiled_submodels) {
            if (comp_model_desc.compiled_model) { // Some models may be optimized out
                value = std::max(value,
                                 comp_model_desc.compiled_model->get_property(ov::optimal_number_of_infer_requests.name())
                                 .as<unsigned int>());
            }
        }
        return decltype(ov::optimal_number_of_infer_requests)::value_type{value};
    } else if (ov::execution_devices == name) {
        std::vector<std::string> device_names;
        std::set<std::string> s;
        for (auto i = 0; i < m_compiled_submodels.size(); ++i) {
            const auto& comp_model_desc = m_compiled_submodels[i];
            if (!comp_model_desc.compiled_model) // handle optimized out
                continue;
            if (s.count(submodel_device(i)) != 0)
                continue;
            s.insert(submodel_device(i));
            device_names.push_back(submodel_device(i));
        }
        return decltype(ov::execution_devices)::value_type{device_names};
    } else if (ov::npuw::number_of_submodels == name) {
        return decltype(ov::npuw::number_of_submodels)::value_type{m_compiled_submodels.size()};
    }
    return m_cfg.get(name);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

void ov::npuw::CompiledModel::export_model(std::ostream& model_stream) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::string ov::npuw::CompiledModel::submodel_device(const std::size_t idx) const {
    std::size_t real_idx = m_compiled_submodels[idx]
                            .replaced_by.value_or(idx);
    auto comp_subm_desc = m_compiled_submodels[real_idx];

    if (comp_subm_desc.switched_to_ref) {
        return m_ref_device;
    }

    NPUW_ASSERT(comp_subm_desc.device_it != m_dev_list.end());
    return *comp_subm_desc.device_it;
}

void ov::npuw::CompiledModel::log_device_dist() const {
    std::unordered_map<std::string, execution_stats> stats_for_devices;
    execution_stats stats_for_optimized_out{0.f, 0ul};

    for (std::size_t id = 0u; id < m_compiled_submodels.size(); id++) {  // FIXME: zip()
        auto& this_cm = m_compiled_submodels[id];

        auto real_id = m_compiled_submodels[id].replaced_by.value_or(id);
        auto& real_cm = m_compiled_submodels.at(real_id);

        execution_stats& stat = real_cm.compiled_model ?
            stats_for_devices[submodel_device(real_id)] :
            stats_for_optimized_out;

        stat.gflops += real_cm.stat.gflops;
        stat.ops += real_cm.stat.ops;
    }

    auto print_stats = [this](const std::string &device, const execution_stats &stat) {
        float flops_prcnt = 100.f;
        float ops_prcnt = 100.f;
        if (m_total_stat.gflops > 0 && m_total_stat.ops > 0) {
            flops_prcnt = stat.gflops / static_cast<float>(m_total_stat.gflops) * 100;
            ops_prcnt = stat.ops / static_cast<float>(m_total_stat.ops) * 100;
        }
        LOG_INFO(device << ": " << flops_prcnt << "% FLOPS, "
                                << ops_prcnt << "% Layers");
    };
    for (auto &&device_st : stats_for_devices) {
        LOG_BLOCK();
        print_stats(device_st.first, device_st.second);
    }
    if (stats_for_optimized_out.gflops > 0 || stats_for_optimized_out.ops > 0) {
        LOG_BLOCK();
        print_stats("Optimized out", stats_for_optimized_out);
    }
}
