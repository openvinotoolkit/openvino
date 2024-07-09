// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "compiled_model.hpp"

#include <iostream>
#include <memory>

#include "accuracy/comparator.hpp"
#include "just_sync_infer_request.hpp"
#include "logging.hpp"
#include "npu_private_properties.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "plugin.hpp"
#include "util.hpp"

// required for get_properties_per_device()
#include <intel_npu/al/config/config.hpp>
#include <intel_npu/al/config/npuw.hpp>
#include <npuw_private_properties.hpp>

#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"

namespace {
void split_properties(const ov::AnyMap& properties,
                      ov::AnyMap& npu_plugin_properties,
                      ov::AnyMap& npuw_path_properties) {
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        if (it->first.find("NPUW") != it->first.npos) {
            npuw_path_properties.insert(*it);
        } else {
            npu_plugin_properties.insert(*it);
        }
    }
}

std::map<std::string, std::string> any_copy(const ov::AnyMap& params) {
    std::map<std::string, std::string> result;
    for (auto&& value : params) {
        result.emplace(value.first, value.second.as<std::string>());
    }
    return result;
}
}  // anonymous namespace

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
                for (auto&& opt : properties) {
                    if (ov::npuw::util::starts_with(opt.first, "NPU")) {
                        device_properties[device_name][opt.first] = opt.second;
                    }
                }
            }  // if(NPU)
        }
    }
    return device_properties;
}
}  // anonymous namespace
}  // namespace npuw
}  // namespace ov

ov::npuw::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                       const std::shared_ptr<const ov::IPlugin>& plugin,
                                       const ov::AnyMap& properties)
    : ov::ICompiledModel(model, plugin),
      m_options_desc(std::make_shared<::intel_npu::OptionsDesc>()),
      m_cfg(m_options_desc),
      m_name(model->get_friendly_name()),
      m_loaded_from_cache(false) {
    ::intel_npu::registerNPUWOptions(*m_options_desc);

    std::map<std::string, ov::Any> npuw_props;
    split_properties(properties, m_non_npuw_props, npuw_props);

    m_cfg.parseEnvVars();
    m_cfg.update(any_copy(npuw_props));

    const std::string dev_list_str = m_cfg.get<::intel_npu::NPUW_DEVICES>();
    m_dev_list = ov::DeviceIDParser::get_hetero_devices(dev_list_str);
    m_meta_devices = ov::npuw::get_properties_per_device(plugin, dev_list_str, m_non_npuw_props);

    const bool acc_check_opt = m_cfg.get<::intel_npu::NPUW_ACC_CHECK>();
    if (acc_check_opt) {
        const double threshold_opt = m_cfg.get<::intel_npu::NPUW_ACC_THRESH>();

        m_acc_check = metrics::NRMSE(threshold_opt);
        m_ref_device = m_cfg.getString<::intel_npu::NPUW_ACC_DEVICE>();
        LOG_INFO("Accuracy check is enabled.");
    }

    LOG_VERB("*** Original model ***");
    const auto& orig_parameters = model->get_parameters();
    {
        LOG_BLOCK();
        for (auto&& p : orig_parameters) {
            LOG_VERB(p);
        }
    }
    const auto& orig_results = model->get_results();
    {
        LOG_BLOCK();
        for (auto&& r : orig_results) {
            LOG_VERB(r);
        }
    }

    auto partitioning = getPartitioning(model, m_cfg);
    m_total_stat.gflops = partitioning.total_gflops;
    m_total_stat.ops = partitioning.total_ops;
    const std::vector<ov::npuw::Subgraph>& orderedSubgraphs = partitioning.subgraphs;

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
        LOG_VERB("Subgraph[" << id << "]");
        LOG_BLOCK();
        if (orderedSubgraphs[id]._optimized_out) {
            LOG_VERB("OPTIMIZED OUT");
            continue;
        }
        auto process_params = [&](const ov::ParameterVector& _parameters) {
            for (size_t i = 0; i < _parameters.size(); i++) {
                LOG_VERB(_parameters[i]);
                for (size_t j = 0; j < orig_parameters.size(); j++) {
                    if (_parameters[i] == orig_parameters[j]) {
                        LOG_BLOCK();
                        LOG_VERB("MATCHED WITH " << orig_parameters[j]);
                        if (NO_LINK == m_inputs_to_submodels_inputs[j]) {
                            // Easy case, first come first served
                            LOG_VERB("Map this parameter directly");
                            m_inputs_to_submodels_inputs[j] = ToSubmodel{id, i};
                        } else {
                            // There's multiple subgraphs reading from the same Parameter.
                            // Record them here, see how this list is later used in the
                            // sync_infer_request.cpp
                            LOG_VERB("Subscribe this parameter to tensor");
                            m_param_subscribers[j].push_back(ToSubmodel{id, i});
                        }  // if(NO_LINK)
                        // Regardless of the order, if the reader is a function,
                        // remember that to avoid confusion in function prologue
                    }  // if(param == orig_param)
                }      // for(orig_params)
            }          // for(subgraph_params)
        };
        auto process_results = [&](const ov::ResultVector& _results) {
            for (size_t i = 0; i < _results.size(); i++) {
                if (!_results[i]) {
                    // See the rearrange_to_function_protocol<>()...
                    continue;
                }
                LOG_VERB(_results[i]);
                for (size_t j = 0; j < orig_results.size(); j++) {
                    if (_results[i] == orig_results[j]) {
                        // FIXME: There may be a problem, see below (!)
                        LOG_BLOCK();
                        LOG_VERB("MATCHED WITH " << orig_results[j]);
                        m_outputs_to_submodels_outputs[j] = {id, i};
                    }
                }
            }  // for(results)
        };
        // Sheer ugliness here again
        if (orderedSubgraphs[id]._funcall.empty()) {
            process_params(orderedSubgraphs[id]._parameters);
            process_results(orderedSubgraphs[id]._results);
        } else {
            // Match via the original parameters to generate indices
            process_params(orderedSubgraphs[id]._parameters);
            process_results(orderedSubgraphs[id]._results);

            // (!) Problem here: when functions are folded, in the
            // KVcache-like scenarios (where every function call
            // produces its own result), its result(s) will be matched
            // with _all_ instances of the result (see pic).
        }
    }  // for(ordered_subgraphs)
    // NOTE(dm): there's a better way to do it, like we do in G-API backends.

    // Store mapping between manually splitted inputs/outputs
    // to connect tensors between compiled submodels
    m_submodels_input_to_prev_output = partitioning.input_to_prev_output;

    // Before the compilation:
    // - initialize the OV models
    // - dump the subgraphs, if necessary
    std::map<std::string, std::size_t> compiledFunctions;
    m_compiled_submodels.resize(orderedSubgraphs.size());

    const std::string dump_sub_opt = m_cfg.get<::intel_npu::NPUW_DUMP_SUBS>();

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
            m_compiled_submodels[id].model = std::make_shared<ov::Model>(subgraph._results,
                                                                         subgraph._sinks,
                                                                         subgraph._parameters,
                                                                         m_name + '_' + std::to_string(id));
        } else {
            LOG_BLOCK();
            auto& fcn_template = partitioning.functions.at(subgraph._funcall);
            auto compiled_fcn_iter = compiledFunctions.find(subgraph._funcall);
            if (compiled_fcn_iter == compiledFunctions.end()) {
                // A function call: store the model for function call only once...
                compiledFunctions.insert({subgraph._funcall, id});
                m_compiled_submodels[id].model = fcn_template._model;
                m_compiled_submodels[id].replaced_by = id;  // FIXME: UGLY
                LOG_INFO("Subgraph[" << id << "] is a function body for " << subgraph._funcall);
            } else {
                // ...and refer to it in other calls
                m_compiled_submodels[id].replaced_by = compiled_fcn_iter->second;
                LOG_INFO("Subgraph[" << id << "] is a function call to [" << compiled_fcn_iter->second << "]");
            }
            m_compiled_submodels[id].param_base = fcn_template._param_offset;
            m_compiled_submodels[id].closure = subgraph._closure;
            m_compiled_submodels[id].scales = subgraph._scales;
            m_compiled_submodels[id].zerops = subgraph._zerops;
        }  // if(!funcall)

        if (!m_compiled_submodels[id].model && !m_compiled_submodels[id].replaced_by) {
            OPENVINO_THROW("Fatal: submodel ", id, " is neither a model nor a function call!");
        }
        const std::size_t real_id = m_compiled_submodels[id].replaced_by.value_or(id);

        // FIXME: a hotfix for a crash where we have too many names in submodel's output
        // Do it just once if that's a function
        if (real_id == id) {
            remove_long_output_names(m_compiled_submodels[real_id].model);
            fill_empty_tensor_names(m_compiled_submodels[real_id].model);
        }

        if (ov::npuw::util::is_set(id, dump_sub_opt)) {
            LOG_INFO("Dumping Subgraph[" << id << "]");
            LOG_BLOCK();
            if (real_id != id) {
                LOG_INFO("NOTE: Dumping Subgraph[" << real_id << "]"
                                                   << " as it is a function body for Subgraph[" << id << "]");
            }
            const auto model_to_dump = m_compiled_submodels[real_id].model;
            std::string model_dump_path = m_name + "_" + ov::npuw::util::fmt(id, m_compiled_submodels.size()) +
                                          (subgraph._funcall.empty() ? "" : "_" + subgraph._funcall) + ".xml";
            ov::save_model(model_to_dump, model_dump_path);
            LOG_INFO("Wrote " << model_dump_path);
            // Note: keep here naming as it would be the subgraph
        }  // if(dump)
    }      // for(orderedSubgraphs)

    std::map<std::size_t, std::string> forced_sub_devices{};
    const std::string fsd_opt = m_cfg.get<::intel_npu::NPUW_SUBMODEL_DEVICE>();
    forced_sub_devices = ::intel_npu ::OptionParser<std::map<std::size_t, std::string>>::parse(fsd_opt);
    // Compile submodels. Some of them can be functions: track which model will be
    // used as function(s): function name -> index of the compiled subgraph
    auto compile = [&](size_t id) {
        const auto& subgraph = orderedSubgraphs[id];
        if (subgraph._optimized_out) {
            return;
        }

        const std::size_t real_id = m_compiled_submodels[id].replaced_by.value_or(id);
        if (!orderedSubgraphs[real_id]._avoid_list.empty()) {
            const auto devices_to_avoid = ov::DeviceIDParser::get_hetero_devices(orderedSubgraphs[real_id]._avoid_list);
            for (auto&& d : devices_to_avoid) {
                m_compiled_submodels[real_id].devices_to_avoid.insert(std::move(d));
            }
        }

        m_compiled_submodels[id].device_it =
            id != real_id ? m_compiled_submodels[real_id].device_it : m_dev_list.cbegin();

        if (forced_sub_devices.count(id)) {
            std::string forced_device = forced_sub_devices[id];
            auto forced_dev_it = std::find(m_dev_list.begin(), m_dev_list.end(), forced_device);
            if (forced_dev_it == m_dev_list.end()) {
                LOG_WARN("Target device for Subgraph[" << id << "] was set to " << forced_device
                                                       << ", but was not found in the device list: "
                                                       << "[" << dev_list_str << "] -- ignoring");
            } else {
                // FIXME: This is not really a device enforcement as the fallback
                // procedure will be in place still
                LOG_INFO("Force Subgraph[" << id << "] target device to " << *forced_dev_it);
                m_compiled_submodels[real_id].device_it = forced_dev_it;
            }
        }  // if(forced_device_opt)

        LOG_INFO("Compiling Subgraph[" << id << "]: " << m_compiled_submodels[real_id].model->get_friendly_name()
                                       << "...");
        if (!compile_for_success(id)) {
            OPENVINO_THROW("Failed to compile ",
                           m_compiled_submodels[real_id].model->get_friendly_name(),
                           " for all devices in [",
                           dev_list_str,
                           "]");
        }

        if (m_acc_check) {
            if (submodel_device(real_id) != m_ref_device) {
                LOG_INFO("Compile Subgraph[" << real_id << "] for reference device: " << m_ref_device << ".");
                LOG_BLOCK();
                m_compiled_submodels.at(real_id).ref_compiled_model =
                    compile_submodel(m_compiled_submodels.at(real_id).model, m_ref_device);
                LOG_INFO("Done (reference)");
            } else {
                LOG_INFO("Skip compilation of submodel[" << real_id << "] for reference device: " << m_ref_device
                                                         << ", as original submodel[" << real_id
                                                         << "] has been already compiled for it.");
            }
        }
    };  // compile

    // Parallel compilation is unstable so is disabled by default.
    const bool par_opt = m_cfg.get<::intel_npu::NPUW_PARALLEL_COMPILE>();
    if (par_opt) {
        ov::parallel_for(orderedSubgraphs.size(), compile);
    } else {
        // TODO: Introduce npuw::serial(i, f) instead where f is a _funcall
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

    implement_properties();

    m_finalized = true;
    reset_io();
}

void ov::npuw::CompiledModel::remove_long_output_names(const std::shared_ptr<ov::Model>& model) {
    NPUW_ASSERT(model.get() != nullptr);
    for (auto& output : model->outputs()) {
        const auto& tensor_names = output.get_tensor().get_names();
        if (tensor_names.size() > 32) {  // maximum supported
            output.get_tensor().set_names({});
            LOG_INFO("Removed output tensor names for " << model->get_friendly_name());
            LOG_BLOCK();
        }
    }
}

void ov::npuw::CompiledModel::fill_empty_tensor_names(const std::shared_ptr<ov::Model>& model) {
    NPUW_ASSERT(model.get() != nullptr);

    size_t in_tensor_idx = 0;
    size_t out_tensor_idx = 0;

    for (auto& input : model->inputs()) {
        const auto& tensor_names = input.get_tensor().get_names();
        if (tensor_names.empty()) {
            input.get_tensor().set_names({"npuw_in_tensor_" + std::to_string(in_tensor_idx)});
            LOG_INFO("Added input tensor name for " << model->get_friendly_name());
            LOG_BLOCK();
        }
        in_tensor_idx++;
    }
    for (auto& output : model->outputs()) {
        const auto& tensor_names = output.get_tensor().get_names();
        if (tensor_names.empty()) {
            output.get_tensor().set_names({"npuw_out_tensor_" + std::to_string(out_tensor_idx)});
            LOG_INFO("Added output tensor name for " << model->get_friendly_name());
            LOG_BLOCK();
        }
        out_tensor_idx++;
    }
}

void ov::npuw::CompiledModel::reset_io() {
    // Restore inputs/outputs from compiled submodels
    // FIXME: this method is also called from IBaseInferReqeust::create_infer_request
    // which is called in the CompiledModel(). So this method executes even before it
    // is called for the right thing from the ctor directly.
    if (!m_finalized)
        return;  // avoid getting called before it is really the time

    // Don't be like HETERO here - don't override the inputs/outputs(),
    // as the ICompiledModel already creates one for us.
    // Instead, remember the mapping from the original CompiledModel::input/output ports
    // to the Subgraph's input/output ports.

    LOG_VERB("*** Partition graph ***");
    int idx_in = 0, idx_out = 0;  // FIXME: use indexed()
    for (const auto& to_submodel : m_inputs_to_submodels_inputs) {
        LOG_BLOCK();
        if (to_submodel == NO_LINK) {
            LOG_WARN("Input (Parameter) " << inputs()[idx_in]
                                          << " is not used by any subgraph. It happens sometimes,"
                                             " but better check your model");
            idx_in++;  // FIXME: PLEASE use indexed() here
            continue;
        }
        const auto& submodel_idx = to_submodel.first;
        const auto& input_idx = to_submodel.second;
        LOG_VERB("Input (Parameter) " << inputs()[idx_in] << " from Subgraph[" << submodel_idx << "]/" << input_idx);
        idx_in++;
    }
    for (const auto& from_submodel : m_outputs_to_submodels_outputs) {
        NPUW_ASSERT(from_submodel != NO_LINK);
        LOG_BLOCK();  // in fact, to_submodel <is> from_submodel here, but who cares
        const auto& submodel_idx = from_submodel.first;
        const auto& output_idx = from_submodel.second;
        LOG_VERB("Output (Result) " << outputs()[idx_out] << " from Subgraph[" << submodel_idx << "]/" << output_idx);
        idx_out++;
    }
}

bool ov::npuw::CompiledModel::compile_for_success(std::size_t id) {
    // Assume device_it is some always-valid starting point.
    // FIXME: And who guarantees it? Abstraction leaks are everywhere
    if (m_compiled_submodels[id].replaced_by && m_compiled_submodels[id].replaced_by != id) {
        LOG_BLOCK();
        LOG_INFO("Skip compilation for Subgraph[" << id
                                                  << "] "
                                                     "as it was already compiled (function)");
        return true;
    }

    for (auto iter = m_compiled_submodels[id].device_it; iter != m_dev_list.cend(); ++iter) {
        LOG_BLOCK();
        const auto& device_name = *iter;
        if (m_compiled_submodels[id].devices_to_avoid.count(device_name) > 0) {
            LOG_INFO(device_name << " was found in the 'Avoid' list for this subgraph, skipping...");
        } else if (compile_for_device(id, device_name)) {
            m_compiled_submodels[id].device_it = iter;
            return true;  // success!
        }
    }
    return false;
}

bool ov::npuw::CompiledModel::compile_for_device(std::size_t id, const std::string& device_to_try) {
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
    if (npuw::util::starts_with(device_to_try, "NPU") && m_compiled_submodels[id].model->inputs().empty()) {
        LOG_INFO("Avoid compilation for " << device_to_try << " as the model should be constant-folded");
        dump_on_fail(id, device_to_try, "Avoided due to workaround");
        return false;
    }

    try {
        m_compiled_submodels[id].compiled_model = compile_submodel(m_compiled_submodels[id].model, device_to_try);
    } catch (const std::exception& ex) {
        LOG_ERROR("Subgraph [" << id << "] Failed to compile: " << std::endl << ex.what());
        dump_on_fail(id, device_to_try, ex.what());
        return false;
    } catch (...) {
        LOG_ERROR("Subgraph [" << id << "] Failed to compile: Unknown error");
        dump_on_fail(id, device_to_try, "Unknown error");
        return false;
    }
    // Reached this point - all ok, stop the search
    LOG_INFO("Done (" << device_to_try << ")");
    return true;
}

ov::SoPtr<ov::ICompiledModel> ov::npuw::CompiledModel::compile_submodel(const std::shared_ptr<ov::Model>& submodel,
                                                                        const std::string& device) {
    auto plugin = get_npuw_plugin();
    auto core = plugin->get_core();
    // set exclusive_async_requests in case when model is split
    // NOTE(dm): Not sure if it is required for the NPUW plugin, but likely it is
    auto& device_config = m_meta_devices[device];
    if (m_compiled_submodels.size() > 1) {
        auto supported_internal_properties =
            plugin->get_core()->get_property(device, ov::internal::supported_properties);
        if (std::find(supported_internal_properties.begin(),
                      supported_internal_properties.end(),
                      ov::internal::exclusive_async_requests) != supported_internal_properties.end()) {
            // adds property if it is not set yet
            device_config.insert(ov::internal::exclusive_async_requests(true));
        }
    }  // if(subgraphs > 1)
    return core->compile_model(submodel, device, device_config);
}

void ov::npuw::CompiledModel::dump_on_fail(std::size_t id, const std::string& device_to_try, const char* extra) {
    const std::string dof_opt = m_cfg.get<::intel_npu::NPUW_DUMP_SUBS_ON_FAIL>();

    if (ov::npuw::util::is_set(id, dof_opt)) {
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
    auto* non_const_this = const_cast<ov::npuw::CompiledModel*>(this);  // because of const in API
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
    OPENVINO_SUPPRESS_DEPRECATED_START
    auto&& configIterator = m_prop_to_opt.find(name);
    if (configIterator != m_prop_to_opt.cend()) {
        return std::get<1>(configIterator->second)(m_cfg);
    } else if (m_non_npuw_props.count(name)) {
        return m_non_npuw_props.at(name);
    }

    OPENVINO_THROW("Unsupported configuration key: ", name);
    OPENVINO_SUPPRESS_DEPRECATED_END
}

void ov::npuw::CompiledModel::export_model(std::ostream& model_stream) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::string ov::npuw::CompiledModel::submodel_device(const std::size_t idx) const {
    std::size_t real_idx = m_compiled_submodels[idx].replaced_by.value_or(idx);
    const auto& comp_subm_desc = m_compiled_submodels[real_idx];

    if (!comp_subm_desc.compiled_model) {
        return "";
    }

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
        auto real_id = m_compiled_submodels[id].replaced_by.value_or(id);
        auto& real_cm = m_compiled_submodels.at(real_id);

        execution_stats& stat =
            real_cm.compiled_model ? stats_for_devices[submodel_device(real_id)] : stats_for_optimized_out;

        stat.gflops += real_cm.stat.gflops;
        stat.ops += real_cm.stat.ops;
    }

    auto print_stats = [this](const std::string& device, const execution_stats& stat) {
        float flops_prcnt = 100.f;
        float ops_prcnt = 100.f;
        if (m_total_stat.gflops > 0 && m_total_stat.ops > 0) {
            flops_prcnt = stat.gflops / static_cast<float>(m_total_stat.gflops) * 100;
            ops_prcnt = stat.ops / static_cast<float>(m_total_stat.ops) * 100;
        }
        LOG_INFO(device << ": " << flops_prcnt << "% FLOPS, " << ops_prcnt << "% Layers");
    };
    for (auto&& device_st : stats_for_devices) {
        LOG_BLOCK();
        print_stats(device_st.first, device_st.second);
    }
    if (stats_for_optimized_out.gflops > 0 || stats_for_optimized_out.ops > 0) {
        LOG_BLOCK();
        print_stats("Optimized out", stats_for_optimized_out);
    }
}

void ov::npuw::CompiledModel::implement_properties() {
    // This function fills the map: {`property name`: `getter for property value`},
    // that can be used later to return requested properties by user.
    // It does it in 3 steps:
    //
    // 1. Create mappings for OV public properties and hints, exposed
    //    in ::intel_npu::CompiledModel.
    // 2. Fill `m_all_supported_props` vector with property names from
    //    the 1st step. It will be returned as response to `ov::supported_properties`
    //    request. So the vector will define public properties.
    // 3. Create mappings for all remaining (private) NPUW-specific properties
    //    to getters of their values from config.

#define GET_PLUGIN_PROP(property) return get_plugin()->get_property(property.name(), ov::AnyMap());

    // 1.
    // OV Public
    // ===============================================
    m_prop_to_opt = {{ov::supported_properties.name(),
                      {ov::PropertyMutability::RO,
                       [&](const ::intel_npu::Config&) {
                           return m_all_supported_props;
                       }}},
                     {ov::device::id.name(),
                      {ov::PropertyMutability::RO,
                       [&](const ::intel_npu::Config&) {
                           GET_PLUGIN_PROP(ov::device::id);
                       }}},
                     {ov::enable_profiling.name(),
                      {ov::PropertyMutability::RO,
                       [&](const ::intel_npu::Config&) {
                           GET_PLUGIN_PROP(ov::enable_profiling);
                       }}},
                     {ov::model_name.name(),
                      {ov::PropertyMutability::RO,
                       [&](const ::intel_npu::Config&) {
                           return m_name;
                       }}},
                     {ov::optimal_number_of_infer_requests.name(),
                      {ov::PropertyMutability::RO,
                       [&](const ::intel_npu::Config&) {
                           return 1u;
                       }}},
                     {ov::execution_devices.name(),
                      {ov::PropertyMutability::RO,
                       [&](const ::intel_npu::Config&) {
                           return "NPU";
                       }}},
                     {ov::loaded_from_cache.name(),
                      {ov::PropertyMutability::RO,
                       [&](const ::intel_npu::Config&) {
                           return m_loaded_from_cache;
                       }}},
                     // OV Public Hints
                     // =====================================================
                     {ov::hint::performance_mode.name(),
                      {ov::PropertyMutability::RO,
                       [&](const ::intel_npu::Config&) {
                           GET_PLUGIN_PROP(ov::hint::performance_mode);
                       }}},
                     {ov::hint::execution_mode.name(),
                      {ov::PropertyMutability::RO,
                       [&](const ::intel_npu::Config&) {
                           GET_PLUGIN_PROP(ov::hint::execution_mode);
                       }}},
                     {ov::hint::num_requests.name(),
                      {ov::PropertyMutability::RO,
                       [&](const ::intel_npu::Config&) {
                           GET_PLUGIN_PROP(ov::hint::num_requests);
                       }}},
                     {ov::hint::inference_precision.name(),
                      {ov::PropertyMutability::RO,
                       [&](const ::intel_npu::Config&) {
                           GET_PLUGIN_PROP(ov::hint::inference_precision);
                       }}},
                     {ov::hint::enable_cpu_pinning.name(),
                      {ov::PropertyMutability::RO,
                       [&](const ::intel_npu::Config&) {
                           GET_PLUGIN_PROP(ov::hint::enable_cpu_pinning);
                       }}},
                     {ov::hint::model_priority.name(), {ov::PropertyMutability::RO, [&](const ::intel_npu::Config&) {
                                                            GET_PLUGIN_PROP(ov::hint::model_priority);
                                                        }}}};
#undef GET_PLUGIN_PROP

    // 2.
    for (auto& p : m_prop_to_opt) {
        m_all_supported_props.emplace_back(ov::PropertyName(p.first, std::get<0>(p.second)));
    }

    // 3.
#define BIND(N, T)                                                                         \
    {                                                                                      \
        ov::intel_npu::N.name(), {                                                         \
            ov::PropertyMutability::RW, [](const ::intel_npu::Config& config) -> ov::Any { \
                return config.get<::intel_npu::T>();                                       \
            }                                                                              \
        }                                                                                  \
    }

    m_prop_to_opt.insert({BIND(use_npuw, NPU_USE_NPUW),
                          BIND(npuw::devices, NPUW_DEVICES),
                          BIND(npuw::submodel_device, NPUW_SUBMODEL_DEVICE),
                          BIND(npuw::partitioning::online::pipeline, NPUW_ONLINE_PIPELINE),
                          BIND(npuw::partitioning::online::min_size, NPUW_ONLINE_MIN_SIZE),
                          BIND(npuw::partitioning::online::avoid, NPUW_ONLINE_AVOID),
                          BIND(npuw::partitioning::online::dump_plan, NPUW_ONLINE_DUMP_PLAN),
                          BIND(npuw::partitioning::plan, NPUW_PLAN),
                          BIND(npuw::partitioning::fold, NPUW_FOLD),
                          BIND(npuw::partitioning::cwai, NPUW_CWAI),
                          BIND(npuw::partitioning::funcall_for_all, NPUW_FUNCALL_FOR_ALL),
                          BIND(npuw::parallel_compilation, NPUW_PARALLEL_COMPILE),
                          BIND(npuw::partitioning::dcoff_type, NPUW_DCOFF_TYPE),
                          BIND(npuw::partitioning::dcoff_with_scale, NPUW_DCOFF_SCALE),
                          BIND(npuw::funcall_async, NPUW_FUNCALL_ASYNC),
                          BIND(npuw::accuracy::check, NPUW_ACC_CHECK),
                          BIND(npuw::accuracy::threshold, NPUW_ACC_THRESH),
                          BIND(npuw::accuracy::reference_device, NPUW_ACC_DEVICE),
#ifdef NPU_PLUGIN_DEVELOPER_BUILD
                          BIND(npuw::dump::full, NPUW_DUMP_FULL),
                          BIND(npuw::dump::subgraphs, NPUW_DUMP_SUBS),
                          BIND(npuw::dump::subgraphs_on_fail, NPUW_DUMP_SUBS_ON_FAIL),
                          BIND(npuw::dump::inputs_outputs, NPUW_DUMP_IO),
                          BIND(npuw::dump::io_iters, NPUW_DUMP_IO_ITERS)
#endif
    });
#undef BIND
}
