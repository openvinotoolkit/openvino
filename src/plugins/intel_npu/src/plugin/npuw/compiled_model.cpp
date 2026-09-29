// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "compiled_model.hpp"
#include "pipelined_infer_request.hpp"

#include <algorithm>
#include <cstring>
#include <deque>
#include <memory>
#include <set>
#include <sstream>
#include <string>

#include "accuracy/comparator.hpp"
#include "attn/attn_subgraph.hpp"
#include "gqa_compiled_model.hpp"
#include "intel_npu/npu_private_properties.hpp"
#include "just_sync_infer_request.hpp"
#include "logging.hpp"
#include "moe/moe_subgraph.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "partitioning/patterns/opt.hpp"
#include "pipelines/kokoro/kokoro_compiled_model.hpp"
#include "plugin.hpp"
#include "unfold_sync_infer_request.hpp"
#include "util.hpp"
#include "v1/elements/accuracy_checked.hpp"
#include "v1/elements/failsafe.hpp"

// required for get_properties_per_device()
#include "intel_npu/config/config.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/npuw_private_properties.hpp"
#include "llm_compiled_model.hpp"
#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/file_util.hpp"
#include "transformations/convert_precision.hpp"

namespace {
std::string canonical_device_name(const std::string& device_name) {
    const auto dot_pos = device_name.find('.');
    return dot_pos == std::string::npos ? device_name : device_name.substr(0, dot_pos);
}

void pre_load_transform(const std::shared_ptr<ov::Model>& model, const ov::AnyMap& props);

std::size_t find_device_index(const std::vector<std::string>& devices, const std::string& device_name) {
    const auto canonical_name = canonical_device_name(device_name);
    const auto it = std::find_if(devices.begin(), devices.end(), [&](const std::string& candidate) {
        return canonical_device_name(candidate) == canonical_name;
    });
    NPUW_ASSERT(it != devices.end());
    return static_cast<std::size_t>(it - devices.begin());
}

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

bool can_use_weightless_flow(const ::intel_npu::Config& config) {
    return config.get<::intel_npu::NPUW_FOLD>() || !config.get<::intel_npu::NPUW_FOLD_ONLY>().empty() ||
           config.get<::intel_npu::NPUW_CWAI>();
}

bool should_use_weightless_flow(const ov::AnyMap& non_npuw_props,
                                const ::intel_npu::Config& config,
                                const std::unordered_map<const void*, std::size_t>& const_to_offset) {
    if (!can_use_weightless_flow(config)) {
        return false;
    }

    bool is_weightless = true;
    if (auto it = non_npuw_props.find(ov::enable_weightless.name()); it != non_npuw_props.end()) {
        is_weightless = it->second.as<bool>();
    } else if (auto it = non_npuw_props.find(ov::cache_mode.name());
               it != non_npuw_props.end() && it->second.as<ov::CacheMode>() == ov::CacheMode::OPTIMIZE_SPEED) {
        is_weightless = false;
    }

    // Weightless serialization is only valid when WAI metadata was generated.
    if (is_weightless && const_to_offset.empty()) {
        is_weightless = false;
    }

    return is_weightless;
}

ov::npuw::s11n::WeightsContext make_import_weights_ctx(const ov::AnyMap& properties,
                                                       bool is_weightless,
                                                       const ov::npuw::s11n::BF16Cache& bf16_consts) {
    using namespace ov::npuw::s11n;

    std::string weights_path;
    WeightsContext::ConstsCache consts_cache;
    ov::FileHandleProvider handle_provider = nullptr;
    if (is_weightless) {
        if (const auto handle_it = properties.find(ov::intel_npu::npuw::weights_handle_provider.name());
            handle_it != properties.end()) {
            if (handle_it->second.is<ov::FileHandleProvider>()) {
                handle_provider = handle_it->second.as<ov::FileHandleProvider>();
            } else {
                LOG_WARN("WEIGHTS_HANDLE_PROVIDER property is present but is not a FileHandleProvider; falling back to "
                         "other weightless import sources");
            }
        }
        if (!handle_provider && properties.find(ov::weights_path.name()) != properties.end()) {
            weights_path = properties.at(ov::weights_path.name()).as<std::string>();
            NPUW_ASSERT(!weights_path.empty() &&
                        "Empty weights_path. Please provide WEIGHTS_PATH or MODEL_PTR in the configuration.");
        } else if (!handle_provider && properties.find(ov::hint::model.name()) != properties.end()) {
            auto model_ptr = std::const_pointer_cast<ov::Model>(
                                 properties.at(ov::hint::model.name()).as<std::shared_ptr<const ov::Model>>())
                                 ->clone();
            NPUW_ASSERT(
                model_ptr &&
                "Empty model passed in MODEL_PTR. Please provide WEIGHTS_PATH or MODEL_PTR in the configuration.");
            pre_load_transform(model_ptr, {});
            for (const auto& node : model_ptr->get_ordered_ops()) {
                if (!ov::op::util::is_constant(node)) {
                    continue;
                }
                const auto& c = std::static_pointer_cast<ov::op::v0::Constant>(node);
                auto rt_info = c->get_rt_info();
                auto weightless_cache_attr = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
                if (weightless_cache_attr == rt_info.end()) {
                    continue;
                }
                std::size_t offset = weightless_cache_attr->second.as<ov::WeightlessCacheAttribute>().bin_offset;
                std::size_t size = c->get_byte_size();
                consts_cache[{offset, size}] = node;
            }
        } else if (!handle_provider) {
            NPUW_ASSERT(false && "Blob is weightless but no WEIGHTS_PATH nor MODEL_PTR property is provided!");
        }
    }

    WeightsPtr weights = nullptr;
    if (is_weightless) {
        std::shared_ptr<ov::MappedMemory> mapped_memory;
        if (handle_provider) {
            ov::FileHandle handle = handle_provider();
            mapped_memory = ov::load_mmap_object(handle);
        } else if (!weights_path.empty()) {
            mapped_memory = ov::load_mmap_object(ov::util::make_path(weights_path));
        }
        if (mapped_memory) {
            weights = std::make_shared<Weights>(mapped_memory->data(), mapped_memory->size(), mapped_memory);
        }
    }

    return WeightsContext(weights, weights_path, consts_cache, bf16_consts, handle_provider);
}

std::function<std::string(const std::string&)> get_encrypt_callback(const ov::AnyMap& properties) {
    if (auto it = properties.find(ov::cache_encryption_callbacks.name()); it != properties.end()) {
        return it->second.as<ov::EncryptionCallbacks>().encrypt;
    }
    return {};
}

std::function<std::string(const std::string&)> get_decrypt_callback_or_throw(const ov::AnyMap& properties) {
    if (auto it = properties.find(ov::cache_encryption_callbacks.name()); it != properties.end()) {
        if (const auto decrypt = it->second.as<ov::EncryptionCallbacks>().decrypt) {
            return decrypt;
        }
    }
    OPENVINO_THROW("Blob is encrypted, but no decryption callback was provided");
}

std::set<std::string> device_list_to_set(const std::string& device_list) {
    std::set<std::string> result;
    if (!device_list.empty()) {
        const auto devices = ov::DeviceIDParser::get_hetero_devices(device_list);
        for (auto&& d : devices) {
            result.insert(std::move(d));
        }
    }
    return result;
}
}  // anonymous namespace

namespace ov {
namespace npuw {
namespace {
ov::AnyMap make_submodel_import_config(const std::string& device, const ::intel_npu::Config& cfg) {
    ov::AnyMap import_config;
    if (ov::npuw::util::starts_with(device, "NPU") && cfg.get<::intel_npu::NPUW_UNFOLD_IREQS>()) {
        import_config["NPU_RUN_INFERENCES_SEQUENTIALLY"] = "YES";
    }
    return import_config;
}

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

namespace {
template <typename T>
auto cfg_get(const ov::AnyMap& properties) -> typename T::ValueType {
    const auto& opt_name = std::string(T::key());
    if (properties.count(opt_name)) {
        return properties.at(opt_name).as<typename T::ValueType>();
    }
    return T::defaultValue();
}

void pre_load_transform(const std::shared_ptr<ov::Model>& model, const ov::AnyMap& props) {
    ov::pass::ConvertPrecision(ov::element::bf16, ov::element::f16).run_on_model(model);

    if (cfg_get<::intel_npu::NPUW_FOLD>(props) && cfg_get<::intel_npu::NPUW_FUNCALL_FOR_ALL>(props)) {
        // If there's folding enabled AND non-repeating graphs are forced to be
        // functions, do extra lifting for gather (if any)
        ov::pass::GraphRewrite rewr;
        rewr.add_matcher<ov::npuw::patterns::opt::DQLiftGatherAsymCW>();
        rewr.add_matcher<ov::npuw::patterns::opt::DQLiftGatherSymCW>();
        rewr.add_matcher<ov::npuw::patterns::opt::DQLiftGatherSymGQ>();
        rewr.add_matcher<ov::npuw::patterns::opt::DQLiftGatherCW>();
        rewr.run_on_model(model);
    }

    if (cfg_get<::intel_npu::NPUW_FOLD>(props)) {
        // Having folding enabled assumes Scalar bank matching,
        // make this procedure a little bit more reliable by untangling
        // the excess tiny Const connections
        ov::npuw::patterns::opt::untangleConst(model);
    }

    if (cfg_get<::intel_npu::NPUW_SLICE_OUT>(props)) {
        // Add Slice before last MatMul for the prefill model
        ov::pass::GraphRewrite rewr;
        rewr.add_matcher<ov::npuw::patterns::opt::SliceLastMatmul>();
        rewr.add_matcher<ov::npuw::patterns::opt::SliceLastMatmulAdd>();
        rewr.add_matcher<ov::npuw::patterns::opt::SliceLastMatmulTranspose>();
        rewr.add_matcher<ov::npuw::patterns::opt::SliceLastMatmulMultiply>();
        rewr.run_on_model(model);
    }
    model->validate_nodes_and_infer_types();
}
}  // anonymous namespace

std::shared_ptr<ov::npuw::ICompiledModel> ov::npuw::ICompiledModel::create(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    LOG_INFO("Choosing which NPUW CompiledModel to create");
    LOG_BLOCK();
    std::shared_ptr<ov::npuw::ICompiledModel> compiled_model;
    auto use_gqa_key = ov::intel_npu::npuw::gqa::enabled.name();
    auto use_llm_key = ov::intel_npu::npuw::llm::enabled.name();
    auto use_kokoro_key = ov::intel_npu::npuw::kokoro::enabled.name();

    // Drop CACHE_DIR from the config
    // If it's present we will be utilizing .*CompiledModel's import
    // and not the underlying models and submodels
    auto config = properties;
    config.erase(ov::cache_dir.name());

    if (properties.count(use_gqa_key) && properties.at(use_gqa_key).as<bool>() == true) {
        LOG_INFO("ov::npuw::GQACompiledModel will be created.");
        compiled_model = std::make_shared<ov::npuw::GQACompiledModel>(model, plugin, config);
    } else if (properties.count(use_llm_key) && properties.at(use_llm_key).as<bool>() == true) {
        LOG_INFO("ov::npuw::LLMCompiledModel will be created.");
        compiled_model = std::make_shared<ov::npuw::LLMCompiledModel>(model, plugin, config);
    } else if (properties.count(use_kokoro_key) && properties.at(use_kokoro_key).as<bool>() == true) {
        LOG_INFO("ov::npuw::KokoroCompiledModel will be created.");
        compiled_model = std::make_shared<ov::npuw::KokoroCompiledModel>(model, plugin, config);
    } else {
        LOG_INFO("ov::npuw::CompiledModel will be created.");
        compiled_model = std::make_shared<ov::npuw::CompiledModel>(model, plugin, config);
    }
    LOG_INFO("Done");
    return compiled_model;
}

ov::npuw::ICompiledModel::ICompiledModel(const std::shared_ptr<ov::Model>& model,
                                         const std::shared_ptr<const ov::IPlugin>& plugin)
    : ov::ICompiledModel(model, plugin) {}

ov::npuw::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                       const std::shared_ptr<const ov::IPlugin>& plugin,
                                       const ov::AnyMap& properties)
    : CompiledModel(model, plugin, properties, nullptr) {}

ov::npuw::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                       const std::shared_ptr<const ov::IPlugin>& plugin,
                                       const ov::AnyMap& properties,
                                       const ov::npuw::v1::subgraphs::PatternRegistry* subgraph_patterns)
    : ov::npuw::ICompiledModel_v0(model, plugin),
      m_options_desc(std::make_shared<::intel_npu::OptionsDesc>()),
      m_cfg(m_options_desc),
      m_name(model->get_friendly_name()),
      m_loaded_from_cache(false) {
    init_profiling();

    // Note: we need to identify original bf16 constants for potential weightless deserialization later
    // And only then do bf16 to f16 transformation
    m_bf16_consts = ov::npuw::s11n::get_bf16_consts(model);
    pre_load_transform(model, properties);

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

    // Initialize weights bank
    const std::string weights_bank_opt = m_cfg.get<::intel_npu::NPUW_WEIGHTS_BANK>();
    const std::string wbank_alloc = m_cfg.get<::intel_npu::NPUW_WEIGHTS_BANK_ALLOC>();
    m_weights_bank = ov::npuw::weights::bank(weights_bank_opt, plugin->get_core(), wbank_alloc);

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

    // Store original constants' offset for serialization purposes
    store_const_offsets(model);

    std::optional<ov::npuw::v1::subgraphs::PatternRegistry> combined_subgraph_patterns;
    std::vector<ov::npuw::v1::subgraphs::ScopedPatternRegistration> builtin_pattern_registrations;
    combined_subgraph_patterns.emplace();
    if (subgraph_patterns != nullptr) {
        combined_subgraph_patterns->append_from(*subgraph_patterns);
    }
    builtin_pattern_registrations =
        ov::npuw::moe::register_patterns(*combined_subgraph_patterns,
                                         m_cfg.get<::intel_npu::NPUW_MOE_TOKEN_CHUNK_SIZE>());

    {
        auto attn_registrations = ov::npuw::attn::register_patterns(*combined_subgraph_patterns);
        for (auto& r : attn_registrations) {
            builtin_pattern_registrations.push_back(std::move(r));
        }
    }

    ov::npuw::PartitioningContext ctx;
    // Identify based on compiler version, user config and pattern
    ctx.use_host_gather_quant = should_use_quantized_host_gather(model, npuw_props);
    ctx.subgraph_patterns = &combined_subgraph_patterns.value();

    ov::npuw::Partitioning partitioning;
    m_profile["partitioning"].record([&]() {
        partitioning = getPartitioning(model, m_cfg, ctx);
    });

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
                NPUW_ASSERT(_parameters[i]);
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
                }  // for(orig_params)
            }  // for(subgraph_params)
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

    const std::size_t end_sub_idx = orderedSubgraphs.size();

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
            m_compiled_submodels[id].pipeline.registration = subgraph._pipeline.registration;
            m_compiled_submodels[id].pipeline.context = subgraph._pipeline.context;
            if (subgraph._pipeline.compile_stage) {
                subgraph._pipeline.compile_stage(m_compiled_submodels[id].pipeline,
                                                 m_compiled_submodels[id].pipeline.context);
            }
        } else {
            LOG_BLOCK();
            auto& fcn_template = partitioning.functions.at(subgraph._funcall);
            auto compiled_fcn_iter = compiledFunctions.find(subgraph._funcall);
            if (compiled_fcn_iter == compiledFunctions.end()) {
                // A function call: store the model for function call only once...
                compiledFunctions.insert({subgraph._funcall, id});

                // For HFA, use the final tile model instead of the original SDPA model
                // because the original SDPA model won't be compiled
                if (fcn_template._host_flash_attention) {
                    m_compiled_submodels[id].model = fcn_template._host_flash_attention.value()._final_tile_model;
                } else {
                    m_compiled_submodels[id].model = fcn_template._model;
                }

                m_compiled_submodels[id].replaced_by = id;  // FIXME: UGLY

                // Fill in the spatial information, if it is present
                if (fcn_template._spatial) {
                    m_compiled_submodels[id].spatial =
                        compiled::Spatial(fcn_template._spatial.value(), fcn_template._model);
                }
                LOG_INFO("Subgraph[" << id << "] is a function body for " << subgraph._funcall);
            } else {
                // ...and refer to it in other calls
                m_compiled_submodels[id].replaced_by = compiled_fcn_iter->second;
                LOG_INFO("Subgraph[" << id << "] is a function call to [" << compiled_fcn_iter->second << "]");
            }
            auto& closure_desc = m_compiled_submodels[id].closure.get();

            m_compiled_submodels[id].pipeline.registration = fcn_template._pipeline.registration;
            m_compiled_submodels[id].pipeline.context = fcn_template._pipeline.context;
            if (compiled_fcn_iter == compiledFunctions.end()) {
                if (fcn_template._pipeline.compile_stage) {
                    fcn_template._pipeline.compile_stage(m_compiled_submodels[id].pipeline,
                                                         m_compiled_submodels[id].pipeline.context);
                    ov::npuw::moe::clear_partition_state(fcn_template._pipeline.context);
                }
            } else {
                const auto real_id = m_compiled_submodels[id].replaced_by.value();
                m_compiled_submodels[id].pipeline.runtime_behavior =
                    m_compiled_submodels[real_id].pipeline.runtime_behavior;
            }
            m_compiled_submodels[id].host_gather = subgraph._host_gather;
            m_compiled_submodels[id].quant_unpack_gather = subgraph._quant_unpack_gather;
            m_compiled_submodels[id].param_base = fcn_template._param_offset;
            closure_desc.closure = subgraph._closure;
            m_compiled_submodels[id].lazy_closure = subgraph._lazy_closure;
            closure_desc.closure_uid.resize(subgraph._closure.size(), -1);
            m_compiled_submodels[id].scales = subgraph._scales;
            m_compiled_submodels[id].zerops = subgraph._zerops;
            m_compiled_submodels[id].forced_to_fcall = subgraph._forced_to_fcall;
            closure_desc.is_remote.resize(subgraph._closure.size(), false);
        }  // if(!funcall)

        if (!m_compiled_submodels[id].model && !m_compiled_submodels[id].replaced_by) {
            OPENVINO_THROW("Fatal: submodel ", id, " is neither a model nor a function call!");
        }
        const std::size_t real_id = m_compiled_submodels[id].replaced_by.value_or(id);
        auto& pipeline = m_compiled_submodels[id].pipeline;
        pipeline.is_function_call =
            m_compiled_submodels[id].replaced_by.has_value() && m_compiled_submodels[id].replaced_by.value() != id;
        pipeline.function_body_subgraph_idx = m_compiled_submodels[id].replaced_by;

        // FIXME: a hotfix for a crash where we have too many names in submodel's output
        // Do it just once if that's a function
        if (real_id == id) {
            auto fix_tensor_names = [&](const std::shared_ptr<ov::Model>& model) {
                if (model) {
                    remove_long_output_names(model);
                    fill_empty_tensor_names(model);
                }
            };

            // Fix tensor names for MoE expert models
            if (const auto* moe_experts =
                    ov::npuw::moe::get_compiled_experts(m_compiled_submodels[real_id].pipeline.context)) {
                for (const auto& [chunk_size, model] : moe_experts->_models_to_compile) {
                    fix_tensor_names(model);
                }
            }

            // Fix tensor names for pyramid attention models
            if (const auto* pyramid_attn =
                    ov::npuw::attn::get_compiled_pyramid(m_compiled_submodels[real_id].pipeline.context)) {
                for (const auto& model : pyramid_attn->_models_to_compile) {
                    fix_tensor_names(model);
                }
            }

            // Fix tensor names for main model (if not replaced by special models above)
            fix_tensor_names(m_compiled_submodels[real_id].model);
        }

        dump_subgraph_model(id, subgraph._funcall, dump_sub_opt);
    }  // for(orderedSubGraphs)

#ifdef NPU_PLUGIN_DEVELOPER_BUILD
    if (!dump_sub_opt.empty()) {
        dump_subgraph_composition(orderedSubgraphs);
    }
#endif

    std::map<std::size_t, std::string> forced_sub_devices{};
    std::string fsd_opt = m_cfg.get<::intel_npu::NPUW_SUBMODEL_DEVICE>();
    // Change "last" keyword to tail subgraph number
    std::size_t last_pos = fsd_opt.find("last");
    if (last_pos != std::string::npos) {
        fsd_opt.erase(last_pos, 4);
        fsd_opt.insert(last_pos, std::to_string(end_sub_idx));
    }
    forced_sub_devices = ::intel_npu::OptionParser<std::map<std::size_t, std::string>>::parse(fsd_opt);

    // Exclude optimized out subgraphs from compilation target beforehand - otherwise we might get head and repeated
    // block in the same chunk
    std::vector<std::size_t> idx_subgraph_to_compile;
    for (std::size_t i = 0u; i < orderedSubgraphs.size(); i++) {
        if (orderedSubgraphs[i]._optimized_out || m_compiled_submodels[i].replaced_by.value_or(i) != i) {
            continue;  // do nothing here
        } else {
            idx_subgraph_to_compile.push_back(i);
        }
    }

    // Compile submodels. Some of them can be functions: track which model will be
    // used as function(s): function name -> index of the compiled subgraph
    auto compile = [&](size_t i) {
        const auto& id = idx_subgraph_to_compile[i];
        const auto& subgraph = orderedSubgraphs[id];

        NPUW_ASSERT(!subgraph._optimized_out);

        const std::size_t real_id = m_compiled_submodels[id].replaced_by.value_or(id);
        m_compiled_submodels[real_id].devices_to_avoid = device_list_to_set(orderedSubgraphs[real_id]._avoid_list);

        // Build the filtered device list before passing to compile_for_success.
        // A forced device (if valid) produces a single-element list; otherwise the full
        // m_dev_list is used with avoided devices removed.
        std::vector<std::string> devices;
        if (forced_sub_devices.count(id)) {
            const std::string& forced_device = forced_sub_devices[id];
            auto forced_dev_it = std::find(m_dev_list.begin(), m_dev_list.end(), forced_device);
            if (forced_dev_it == m_dev_list.end()) {
                LOG_WARN("Target device for Subgraph[" << id << "] was set to " << forced_device
                                                       << ", but was not found in the device list: "
                                                       << "[" << dev_list_str << "] -- ignoring");
            } else {
                LOG_INFO("Force Subgraph[" << id << "] target device to " << *forced_dev_it);
                devices.push_back(*forced_dev_it);
            }
        }
        if (devices.empty()) {
            const auto& avoid = m_compiled_submodels[real_id].devices_to_avoid;
            for (const auto& device_name : m_dev_list) {
                if (avoid.count(device_name) > 0) {
                    LOG_BLOCK();
                    LOG_INFO(device_name << " was found in the 'Avoid' list for this subgraph, skipping...");
                } else {
                    devices.push_back(device_name);
                }
            }
        }

        LOG_INFO("Compiling Subgraph[" << id << "]: " << m_compiled_submodels[real_id].model->get_friendly_name()
                                       << "...");
        if (!compile_for_success(id, devices)) {
            OPENVINO_THROW("Failed to compile ",
                           m_compiled_submodels[real_id].model->get_friendly_name(),
                           " for all devices in [",
                           dev_list_str,
                           "]");
        }

        LOG_INFO("Done (Subgraph[" << id << "]).");
    };  // compile

    // Parallel compilation is unstable so is disabled by default.
    const bool par_opt = m_cfg.get<::intel_npu::NPUW_PARALLEL_COMPILE>();
    if (par_opt) {
        ov::parallel_for(idx_subgraph_to_compile.size(), compile);
    } else {
        ov::npuw::util::non_parallel_for(idx_subgraph_to_compile.size(), compile);
    }

    const bool m_use_npu_local_pipelining{m_cfg.get<::intel_npu::NPUW_CONTROLFLOW_EN>()};

    {
        const auto submodel_count{m_compiled_submodels.size()};

        for (std::size_t i = 0; i < submodel_count; i++) {
            auto& comp_model_desc = m_compiled_submodels[i];

            if (!comp_model_desc.compiled_model && (i != comp_model_desc.replaced_by)) {
                continue;  // Optimized out
            }

            {
                const auto& ports{comp_model_desc.compiled_model->inputs()};
                comp_model_desc.input_port_type.reserve(ports.size());
                comp_model_desc.input_port_name.reserve(ports.size());

                for (auto& port : ports) {
                    auto type{port.get_element_type()};
                    auto port_name{port.get_node()->get_friendly_name()};
                    comp_model_desc.input_port_type.push_back(type);
                    comp_model_desc.input_port_name.push_back(port_name);
                }
            }
            {
                const auto& ports{comp_model_desc.compiled_model->outputs()};
                comp_model_desc.output_port_name.reserve(ports.size());

                for (auto& port : ports) {
                    auto port_name{port.get_node()->get_friendly_name()};
                    comp_model_desc.output_port_name.push_back(port_name);
                }
            }
        }
    }

    if (m_use_npu_local_pipelining && idx_subgraph_to_compile.size() > 1ull) {
        const auto submodel_count{m_compiled_submodels.size()};      
        
        // LOG_WARN("Using NPU local pipelines with ControlFlowOps.");
        // std::cout << "Using NPU local pipelines with ControlFlowOps for inference.\n";
        auto plugin = get_npuw_plugin();
        auto core = plugin->get_core();
        auto npuw_plugin = std::dynamic_pointer_cast<const ::intel_npu::Plugin>(plugin);

        std::vector<::intel_npu::elf_binary> compiled_elf_blobs{};
        std::vector<::intel_npu::elf_binary> elf_blobs{};
        const auto m_num_submodels{idx_subgraph_to_compile.size()};
        compiled_elf_blobs.reserve(m_num_submodels);
        std::map<std::size_t, std::deque<std::size_t>> submodel_to_blob_mapping{};
        std::map<std::size_t, std::size_t> duplicate_blob_indices_lut;

        elf_blobs.reserve(submodel_count);

        std::vector<std::size_t> submodel_elf_blob_index(submodel_count);
        std::map<size_t, size_t> real_idx_blob_mapping{};
        m_hfa_behaviour_submodel.clear();
        std::map<size_t, size_t> hfa_input_port_pipeline_mapping;
        std::map<size_t, size_t> hfa_pipeline_output_consumer;
        std::map<size_t, std::string> hfa_shared_input_parameters;

        for (std::size_t i = 0; i < submodel_count; i++) {
            auto& comp_model_desc = m_compiled_submodels[i];
            // Multiple submodels with the same replaced_by index share one compiled ELF body (funcall mechanism).
            // submodel_to_blob_mapping groups them: key = body submodel index, value = all funcall indices.
            // e.g., blob_index=0 -> {0,1,...,31} for 32 transformer layers sharing one compiled ELF.
            const auto real_idx = comp_model_desc.replaced_by.value_or(i);
            submodel_elf_blob_index[i] = real_idx;

            if (!comp_model_desc.compiled_model && (i != comp_model_desc.replaced_by)) {
                auto hfa_it{m_hfa_behaviour_submodel.find(real_idx)};

                if (hfa_it != std::end(m_hfa_behaviour_submodel)) {
                    m_hfa_behaviour_submodel[i] = real_idx;
                }
                continue;  // Optimized out
            }

            if (auto* hfa = ov::npuw::attn::get_compiled_hfa(comp_model_desc.pipeline.context)) {
                auto hfa_pipeline_blob{nlp_host_flash_attention_pipeline(hfa)};
                compiled_elf_blobs.push_back(hfa_pipeline_blob);
                // m_submodels_input_to_prev_output
                m_hfa_behaviour_submodel[i] = i;
                hfa_input_port_pipeline_mapping = hfa->input_port_pipeline_mapping;
                hfa_shared_input_parameters = hfa->shared_input_parameters;
                m_pipeline_has_hfa = true;
            } else {
                std::stringstream model_stream;
                comp_model_desc.compiled_model->export_model(model_stream);
                std::string model_str{model_stream.str()};
                std::vector<uint8_t> buffer(model_str.begin(), model_str.end());


                {
                    auto& inputs{comp_model_desc.compiled_model->inputs()};
                    auto& outputs{comp_model_desc.compiled_model->outputs()};

                    std::stringstream io_name_parameters{};

                    auto add_port_names = [&](const std::vector<ov::Output<const ov::Node>>& ports) {
                        for (auto& port : ports) {
                            io_name_parameters << port.get_node()->get_friendly_name() << "\n";
                        }
                    };

                    add_port_names(inputs);
                    add_port_names(outputs);

                    const auto io_name_parameters_str{io_name_parameters.str()};

                    npuw_plugin->schedule_builder_proceed(buffer,
                                                          ::intel_npu::BuilderOption::io_name,
                                                          io_name_parameters_str);
                }
                compiled_elf_blobs.push_back(buffer);
            }

            real_idx_blob_mapping[real_idx] = compiled_elf_blobs.size() - 1ull;
        }

        {
            size_t prev_sm{~0ull};

            for (auto& hfa_sm : m_hfa_behaviour_submodel) {
                const auto sm_idx{hfa_sm.first};

                if (prev_sm != ~0ull) {
                    for (auto& shared_port : hfa_shared_input_parameters) {
                        std::pair<size_t, size_t> key{sm_idx, shared_port.first};
                        m_submodels_input_to_prev_input[key] = {prev_sm, shared_port.first};
                    }
                }
                // hfa_shared_input_parameters
                std::map<std::pair<size_t, size_t>, std::pair<size_t, size_t>> new_input_to_prev_output;

                for (auto& io_mapping : hfa_input_port_pipeline_mapping) {
                    std::pair<size_t, size_t> key{sm_idx, io_mapping.first};

                    auto it{m_submodels_input_to_prev_output.find(key)};

                    if (it != std::end(m_submodels_input_to_prev_output)) {
                        auto prev_input{it->second};
                        m_submodels_input_to_prev_output.erase(it);
                        std::pair<size_t, size_t> new_key{sm_idx, io_mapping.second};
                        new_input_to_prev_output[new_key] = prev_input;
                    }
                }

                for (auto& nm : new_input_to_prev_output) {
                    m_submodels_input_to_prev_output[nm.first] = nm.second;
                }

                prev_sm = sm_idx;
            }

            size_t submodel_index{};
            size_t submodel_index_base{};
            size_t prev_elf_blob_index{~0ull};
            size_t real_idx{};
            size_t prev_real_idx{};

            for (auto real_sm_idx : submodel_elf_blob_index) {
                auto elf_blob_index{real_idx_blob_mapping[real_sm_idx]};

                if (prev_elf_blob_index == elf_blob_index) {
                    submodel_to_blob_mapping[prev_real_idx].push_back(submodel_index);
                    duplicate_blob_indices_lut[prev_real_idx] = real_sm_idx;
                    m_pipeline_submodel_base_indices[submodel_index] = submodel_index_base;
                    submodel_index++;
                    continue;
                } else {
                    submodel_to_blob_mapping[real_idx].push_back(submodel_index);
                    duplicate_blob_indices_lut[real_idx] = real_sm_idx;
                }

                prev_real_idx = real_idx;
                submodel_index_base = submodel_index;
                m_pipeline_submodel_base_indices[submodel_index] = submodel_index_base;

                auto buffer{compiled_elf_blobs[elf_blob_index]};
                std::stringstream global_io_mapping_param{};
                global_io_mapping_param << "[INLINE]";

                // io_global: tells the JLP driver which local ports of this ELF correspond to the
                // model's global I/O ports.  This mapping is STATIC - it uses the original pre-transform
                // port numbers and does NOT change across iterations.  io_global is applied before
                // io_reuse/io_iterate so all port references here are the raw ELF port indices.
                // The driver preserves these associations in the fused blob; after fuse_elf the
                // io_mapping pass embeds "#GI[N]" / "#GO[N]" markers in the port-name text so that
                // schedule_builder_get_io_mapping() can later extract m_pipeline_global_inputs /
                // m_pipeline_global_outputs.
                auto create_global_io_mapping = [&](std::vector<ToSubmodel>& submodel_mapping, std::string io_marker) {
                    uint32_t global_io_port{};

                    for (auto& global_io : submodel_mapping) {
                        const auto target_submodel_index{global_io.first};
                        auto target_submodel_port{global_io.second};

                        if (target_submodel_index == submodel_index) {
                            auto hfa_it{m_hfa_behaviour_submodel.find(target_submodel_index)};

                            if (hfa_it != std::end(m_hfa_behaviour_submodel)) {
                                target_submodel_port = hfa_input_port_pipeline_mapping[target_submodel_port];
                            } 
                            
                            global_io_mapping_param << "\n"
                                                    << io_marker << global_io_port << ":" << target_submodel_port;
                        }
                        global_io_port++;
                    }
                };

                create_global_io_mapping(m_inputs_to_submodels_inputs, "i");
                create_global_io_mapping(m_outputs_to_submodels_outputs, "o");
                auto global_io_mapping_parameters{global_io_mapping_param.str()};

                npuw_plugin->schedule_builder_proceed(buffer,
                                                      ::intel_npu::BuilderOption::io_global,
                                                      global_io_mapping_parameters);

                elf_blobs.push_back(buffer);

                prev_elf_blob_index = elf_blob_index;
                real_idx++;
                submodel_index++;
            }
        }

        size_t prev_submodel_index{~0ull};

        uint32_t elf_index{};
        using size_t_pair = std::map<size_t, size_t>;

        size_t_pair submodel_to_elf_mapping{};
        std::map<size_t, size_t_pair> submodel_io_reuse;
        std::map<size_t, size_t_pair> submodel_io_reuse_new_output_ports;

        std::stringstream io_consolidate_params{};
        io_consolidate_params << "[INLINE]";
        std::deque<size_t> iterate_submodel_indices;
        std::map<size_t, std::deque<size_t>> submodel_reuse_lut;

        for (auto& blob_mapping : submodel_to_blob_mapping) {
            const auto duplicate_blob_index{blob_mapping.first};

            const auto& blob_submodels{blob_mapping.second};
            std::map<std::pair<size_t, size_t>, size_t> prev_submodel_io_mapping{};
            std::map<std::pair<size_t, size_t>, size_t> prev_submodel_input_only_mapping{};

            size_t current_submodel_index{};

            if (blob_submodels.size()) {
                current_submodel_index = blob_submodels.front();
            }

            size_t input_ports{};
            size_t output_ports{};
            const auto blob_index{duplicate_blob_indices_lut[duplicate_blob_index]};
            submodel_reuse_lut[blob_index].push_back(current_submodel_index);
            auto& comp_model_desc = m_compiled_submodels[blob_index];
            input_ports = comp_model_desc.input_port_name.size();
            output_ports = comp_model_desc.output_port_name.size();

           

            // Create IO reuse parameters
            if (blob_submodels.size() > 1ull) {
                const auto last_submodel_index{blob_submodels.back()};

                // submodel_index_lut: maps submodel index -> position within this funcall group.
                // Used to distinguish intra-group connections (same ELF, loop-back)
                // from inter-group connections (cross-ELF, handled by io_consolidate).
                size_t_pair submodel_index_lut{};

                // connected_inputs / connected_outputs: ports that are internally consumed.
                // Ports in these sets are excluded from io_iterate's per-iteration weight range ("i" declaration).
                // connected_inputs collects TWO types:
                //   (1) io_reuse loop-back ports: output[X] feeds back into input[Y] same buffer
                //   (2) cross-ELF incoming ports: prev-ELF output connects to this ELF's input
                // Both types must be excluded so that only true weight-bank ports appear in io_iterate.
                size_t_pair connected_inputs{};
                size_t_pair connected_outputs{};

                {
                    size_t position{};
                    for (auto indx : blob_submodels) {
                        submodel_index_lut[indx] = position++;
                    }
                }

                const auto submodel_index_lut_it_end{std::end(submodel_index_lut)};
                // blob_input_output_mapping: output_port -> input_port for intra-group (io_reuse) connections.
                // Each entry represents: "output[X] and input[Y] share the same buffer" - the driver
                // allocates a single buffer and binds it to both ports so no copy is needed between iterations.
                size_t_pair blob_input_output_mapping{};

                auto make_submodel_io_connection =
                    [&](auto& m_submodels_io, auto& io_mapping, const bool track_io_reuse) {
                        for (const auto& kvp : m_submodels_io) {
                            const auto& subm_idx_to{kvp.first.first};
                            const auto& port_idx_to{kvp.first.second};
                            const auto& subm_idx_from{kvp.second.first};
                            const auto& port_idx_from{kvp.second.second};

                            auto to_submodel_it{submodel_index_lut.find(subm_idx_to)};
                            auto from_submodel_it{submodel_index_lut.find(subm_idx_from)};

                            if (to_submodel_it != submodel_index_lut_it_end &&
                                from_submodel_it != submodel_index_lut_it_end) {
                                if (track_io_reuse) {
                                    blob_input_output_mapping[port_idx_from] = port_idx_to;
                                    connected_outputs[port_idx_from] = port_idx_from;
                                }

                                connected_inputs[port_idx_to] = port_idx_to;
                            }

                            if (subm_idx_from <= prev_submodel_index && to_submodel_it != submodel_index_lut_it_end) {
                                io_mapping[std::pair<size_t, size_t>{subm_idx_from, port_idx_from}] = port_idx_to;
                                connected_inputs[port_idx_to] = port_idx_to;
                            }
                        }
                    };

                make_submodel_io_connection(m_submodels_input_to_prev_output, prev_submodel_io_mapping, true);
                make_submodel_io_connection(m_submodels_input_to_prev_input, prev_submodel_input_only_mapping, false);

                {
                    std::stringstream io_reuse_params{};
                    io_reuse_params << "[INLINE]";

                    for (auto& io_port : blob_input_output_mapping) {
                        io_reuse_params << "\n" << io_port.first << ":" << io_port.second;
                    }

                    submodel_io_reuse[last_submodel_index] = blob_input_output_mapping;

                    // submodel_io_reuse stores the output->input reuse mapping for later use in
                    // io_consolidate: when a cross-ELF source port has been declared in io_reuse,
                    // the driver sees it as an input buffer (not an output), so io_consolidate must
                    // reference it with the "i" prefix instead of "o".

                    npuw_plugin->schedule_builder_proceed(elf_blobs[elf_index],
                                                          ::intel_npu::BuilderOption::io_reuse,
                                                          io_reuse_params.str());
                }

                {
                    std::stringstream io_iterate_params{};
                    io_iterate_params << "[INLINE]\nloop:" << blob_submodels.size();

                    // get_duplicate_io_range: determines which contiguous port ranges to declare in io_iterate.
                    // For INPUTS (is_output=false):
                    //   port_index_mapping[i] = i (identity, no renumbering).
                    //   Ports NOT in connected_inputs are per-iteration weight-bank slots; they form the "i<s>:<e>"
                    //   ranges.  Both io_reuse loop-back ports and cross-ELF incoming ports are excluded.
                    // For OUTPUTS (is_output=true):
                    //   First builds a compact renumbering: io_reuse-consumed ports get ~0ull (REMOVED),
                    //   remaining ports get sequential new indices 0, 1, 2, ...
                    //   Then writes "o<new_start>:<new_end>" for contiguous non-removed ranges.
                    //   This renumbering is critical: io_consolidate must use the NEW port numbers when
                    //   referencing cross-ELF output connections from this ELF.
                    auto get_duplicate_io_range = [&](const size_t& ports,
                                                      size_t_pair& connected_ports,
                                                      const std::string& direction,
                                                      bool is_output) {
                        size_t start_port{~0ull};
                        size_t prev_port{};
                        std::map<size_t, size_t> port_index_mapping;

                        if (is_output) {
                            size_t out_indx{};

                            for (size_t indx{}; indx < ports; ++indx) {
                                auto port_it{connected_ports.find(indx)};
                                port_index_mapping[indx] = out_indx;

                                if (port_it == std::end(connected_ports)) {
                                    out_indx++;
                                } else {
                                    port_index_mapping[indx] = ~0ull;
                                }
                            }
                        } else {
                            for (size_t indx{}; indx < ports; ++indx) {
                                port_index_mapping[indx] = indx;
                            }
                        }

                        for (size_t indx{}; indx < ports; ++indx) {
                            auto input_it{connected_ports.find(indx)};

                            if (input_it == std::end(connected_ports)) {
                                if (start_port == ~0ull) {
                                    start_port = indx;
                                    prev_port = indx;
                                } else if (indx == (ports - 1ull)) {
                                    io_iterate_params << "\n"
                                                      << direction << port_index_mapping[start_port] << ":"
                                                      << port_index_mapping[indx];
                                    start_port = ~0ull;
                                } else if ((prev_port + 1ull) == indx) {
                                    prev_port = indx;
                                }
                            } else if (start_port != ~0ull) {
                                io_iterate_params << "\n"
                                                  << direction << port_index_mapping[start_port] << ":"
                                                  << port_index_mapping[prev_port];
                                start_port = ~0ull;
                            }
                        }

                        if (start_port != ~0ull) {
                            io_iterate_params << "\n"
                                              << direction << port_index_mapping[start_port] << ":"
                                              << port_index_mapping[prev_port];
                            start_port = ~0ull;
                        }

                        return port_index_mapping;
                    };

                    get_duplicate_io_range(input_ports, connected_inputs, "i", false);
                    auto output_port_index_mapping{get_duplicate_io_range(output_ports, connected_outputs, "o", true)};

                    // submodel_io_reuse_new_output_ports stores the orig->new output port mapping
                    // produced by io_iterate.  io_consolidate reads this map to translate
                    // a source ELF's original output port number to the renumbered port that
                    // io_iterate exposed to the outside world.
                    submodel_io_reuse_new_output_ports[last_submodel_index] = output_port_index_mapping;

                    npuw_plugin->schedule_builder_proceed(elf_blobs[elf_index],
                                                          ::intel_npu::BuilderOption::io_iterate,
                                                          io_iterate_params.str());
                }
            }

            auto make_submodel_io_connection_no_reuse = [&](auto& m_submodels_io, auto& io_mapping) {
                for (const auto& kvp : m_submodels_io) {
                    const auto& subm_idx_to{kvp.first.first};
                    const auto& port_idx_to{kvp.first.second};
                    const auto& subm_idx_from{kvp.second.first};
                    const auto& port_idx_from{kvp.second.second};

                    if (current_submodel_index == subm_idx_to) {
                        io_mapping[std::pair<size_t, size_t>{subm_idx_from, port_idx_from}] = port_idx_to;
                    }
                }
            };

            if (!prev_submodel_io_mapping.size() && prev_submodel_index != ~0ull) {
                make_submodel_io_connection_no_reuse(m_submodels_input_to_prev_output, prev_submodel_io_mapping);
            }

            if (!prev_submodel_input_only_mapping.size() && prev_submodel_index != ~0ull) {
                make_submodel_io_connection_no_reuse(m_submodels_input_to_prev_input, prev_submodel_input_only_mapping);
            }

            for (auto indx : blob_submodels) {
                submodel_to_elf_mapping[indx] = elf_index;
                prev_submodel_index = indx;
            }

            if (prev_submodel_io_mapping.size()) {
                size_t prev_src_elf_index{~0ull};

                for (auto io : prev_submodel_io_mapping) {
                    const auto src_blob_index{io.first.first};
                    const auto src_elf_index{submodel_to_elf_mapping[src_blob_index]};
                    
                    auto src_output_port{io.first.second};
                    std::string src_port_str{"o"};

                    const auto dst_input_port{io.second};

                    auto hfa_subm_it{m_hfa_behaviour_submodel.find(current_submodel_index)};

                    if (hfa_subm_it == std::end(m_hfa_behaviour_submodel))
                    {
                        auto& port_name{comp_model_desc.input_port_name[dst_input_port]};
                        m_pipeline_connected_inputs[port_name]++;
                        m_pipeline_shared_inputs[port_name].push_back(
                            std::pair<size_t, size_t>{elf_index, dst_input_port});
                    }
                  
                    auto hfa_prev_subm_it{m_hfa_behaviour_submodel.find(io.first.first)};

                    if (hfa_prev_subm_it != std::end(m_hfa_behaviour_submodel)) {
                        hfa_pipeline_output_consumer[elf_index] = dst_input_port;
                    }

                    auto src_reuse_it{submodel_io_reuse.find(src_blob_index)};
                    auto src_reuse_out_port_it{submodel_io_reuse_new_output_ports.find(src_blob_index)};

                    // Port-prefix selection for io_consolidate:
                    //
                    // Case 1 - source port was declared in io_reuse (output[X] == input[Y], shared buffer):
                    //   io_iterate marked it as REMOVED from the output list.
                    //   The driver knows this buffer only by its INPUT side address.
                    //   -> Switch prefix from "o" to "i" and use the corresponding input port number.
                    //
                    // Case 2 - source ELF has io_iterate output renumbering (but this port was NOT reused):
                    //   io_iterate assigned new sequential indices to surviving output ports.
                    //   -> Keep "o" prefix but translate port number to the new (compact) index.
                    if (src_reuse_it != std::end(submodel_io_reuse)) {
                        auto output_it{src_reuse_it->second.find(src_output_port)};

                        if (output_it != std::end(src_reuse_it->second)) {
                            src_output_port = output_it->second;
                            src_port_str = "i";
                        }
                    } else if (src_reuse_out_port_it != std::end(submodel_io_reuse_new_output_ports)) {
                        auto output_it{src_reuse_out_port_it->second.find(src_output_port)};

                        if (output_it != std::end(src_reuse_out_port_it->second)) {
                            src_output_port = output_it->second;
                        }
                    }

                    if (prev_src_elf_index != src_elf_index) {
                        io_consolidate_params << "\ns" << src_elf_index << ":s" << elf_index;
                    }

                    io_consolidate_params << "\n" << src_port_str << src_output_port << ":i" << dst_input_port;
                    prev_src_elf_index = src_elf_index;
                }
            }

            if (prev_submodel_input_only_mapping.size()) {
                size_t prev_src_elf_index{~0ull};

                for (auto io : prev_submodel_input_only_mapping) {
                    const auto src_blob_index{io.first.first};
                    const auto src_elf_index{submodel_to_elf_mapping[src_blob_index]};

                    auto src_output_port{io.first.second};
                    const auto dst_input_port{io.second};

                    if (prev_src_elf_index != src_elf_index) {
                        io_consolidate_params << "\ns" << src_elf_index << ":s" << elf_index;
                    }

                    io_consolidate_params << "\ni" << src_output_port << ":i" << dst_input_port;
                    prev_src_elf_index = src_elf_index;
                }
            }
            elf_index++;
        }

        // connect shared input ports
        for (auto& input_bundle: m_pipeline_shared_inputs) {
            auto& bundle{input_bundle.second};

            if (bundle.size() > 1ull) {
                auto primary_input{bundle.front()};
                const auto primary_elf_index{primary_input.first};
                const auto primary_input_port{primary_input.second};

                bundle.pop_front();

                for (auto& secondary_input : bundle) {
                    io_consolidate_params << "\ns" << primary_elf_index << ":s" << secondary_input.first;
                    io_consolidate_params << "\ni" << primary_input_port << ":i" << secondary_input.second;
                }
            }
        }

        // HFA pipeline output shared connection
        if (hfa_pipeline_output_consumer.size() > 1ull) {
            auto stage_it{std::begin(hfa_pipeline_output_consumer)};

            const auto primary_elf_stage{stage_it->first};
            const auto primary_elf_stage_port{stage_it->second};
            hfa_pipeline_output_consumer.erase(stage_it);

            for (auto stage : hfa_pipeline_output_consumer) {
                io_consolidate_params << "\ns" << primary_elf_stage << ":s" << stage.first;
                io_consolidate_params << "\ni" << primary_elf_stage_port << ":i" << stage.second;
            }
        }
        // Final ELF fusion pass:
        // fuse_elf merges all individual ELF blobs into a single pipeline ELF.
        // io_consolidate is passed as a co-option (same pfnCreate4 call via pNext chaining)
        // so the driver knows the cross-ELF port wiring while performing the fusion.
        // After this call elf_blobs[0] contains the complete pipeline ELF blob.
        auto io_consolidate_parameter{io_consolidate_params.str()};

        for (auto& blob_mapping : submodel_to_blob_mapping) {
            const auto duplicate_blob_index{blob_mapping.first};
            const auto& blob_submodels{blob_mapping.second};

            size_t current_submodel_index{};

            if (blob_submodels.size()) {
                current_submodel_index = blob_submodels.front();
            }

            const auto blob_index{duplicate_blob_indices_lut[duplicate_blob_index]};

            if (submodel_reuse_lut[blob_index].size() > 1ull) {
                iterate_submodel_indices.push_back(current_submodel_index);
            }
        }

        {
            std::map<std::string, size_t> offsets;

            for (auto sm_idx : iterate_submodel_indices) {
                auto elf_idx{submodel_to_elf_mapping[sm_idx]};
                auto& elf_blob{elf_blobs[elf_idx]};

                std::map<uint32_t, uint32_t> pipeline_global_inputs;
                std::map<uint32_t, uint32_t> pipeline_global_outputs;
                std::map<std::string, std::vector<uint32_t>> pipeline_global_parameters;

                npuw_plugin->schedule_builder_get_io_mapping(elf_blob,
                                                             pipeline_global_inputs,
                                                             pipeline_global_inputs,
                                                             pipeline_global_parameters);

                auto& submodel_offset{m_pipeline_global_parameters_offset[sm_idx]};

                for (auto& params : pipeline_global_parameters) {
                    submodel_offset[params.first] = offsets[params.first];

                    offsets[params.first] += params.second.size();
                }
            }
        }

          
        {
            std::vector<::intel_npu::BuilderOption> options(2);
            options[0] = ::intel_npu::BuilderOption::fuse_elf;
            options[1] = ::intel_npu::BuilderOption::io_consolidate;
            std::vector<std::string> option_parameters(2);
            option_parameters[1] = io_consolidate_parameter;
            npuw_plugin->schedule_builder_proceed(elf_blobs, options, option_parameters);
        }
        auto& pipeline_blob{elf_blobs[0]};
        npuw_plugin->schedule_builder_proceed(pipeline_blob, ::intel_npu::BuilderOption::compress, "");
        npuw_plugin->schedule_builder_proceed(pipeline_blob, ::intel_npu::BuilderOption::chunk_dma, "");
        npuw_plugin->schedule_builder_proceed(pipeline_blob, ::intel_npu::BuilderOption::tiny_dma, "");
        // schedule_builder_get_io_mapping runs the IO_MAPPING pass on the fused pipeline blob.
        // The driver writes port-name text containing "#GI[N]" / "#GO[N]" markers for global ports
        // and "iter<k>" suffixes for per-iteration weight ports.  The resulting maps are used at
        // inference time by:
        //   m_pipeline_global_inputs  -> bind_global_params() to route model inputs to pipeline ports
        //   m_pipeline_global_outputs -> just_sync_infer_request::set_tensor() for model outputs
        //   m_pipeline_global_parameters -> unpack_closure() to bind per-layer weights by iteration index
        npuw_plugin->schedule_builder_get_io_mapping(pipeline_blob,
                                                     m_pipeline_global_inputs,
                                                     m_pipeline_global_outputs,
                                                     m_pipeline_global_parameters);

        if (m_pipeline_has_hfa) {
            m_nlp_branch_select_port_idx = m_pipeline_global_parameters[m_nlp_branch_select_port_name].front();            
        }
        
        std::string blob_str(std::begin(pipeline_blob), std::end(pipeline_blob));
        std::istringstream model_stream(blob_str);

        ov::AnyMap nlp_properties{};
        nlp_properties["NPU_IMPORT_RAW_BLOB"] = true;
        m_compiled_pipeline_model = core->import_model(model_stream, get_context(), nlp_properties);
   
    }

    // Finalize memory in closures and weight banks
    finalize_weights_bank();
    detach_memory();

    // Print stats report when possible
    {
        LOG_INFO("Initial device distribution:");
        LOG_BLOCK();
        log_device_dist();
    }

    implement_properties();
    report_io();
}

ov::npuw::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                       const std::shared_ptr<const ov::IPlugin>& plugin,
                                       const bool serialized)
    : ov::npuw::ICompiledModel_v0(model, plugin),
      m_options_desc(std::make_shared<::intel_npu::OptionsDesc>()),
      m_cfg(m_options_desc),
      m_name(model->get_friendly_name()),
      m_loaded_from_cache(serialized) {
    NPUW_ASSERT(serialized && "This constructor should only be utilized during deserialization!");
    init_profiling();

    ::intel_npu::registerNPUWOptions(*m_options_desc);
    LOG_DEBUG("CompiledModel is being deserialized, skipping the full constructor flow...");
}

void ov::npuw::CompiledModel::init_profiling() {
    // to be called from contructors
    m_profile.report_on_die = ov::npuw::profiling_enabled();
    m_profile.area = m_name + "/compilation";
}

bool ov::npuw::CompiledModel::should_use_quantized_host_gather(const std::shared_ptr<ov::Model>& model,
                                                               const ov::AnyMap& properties) const {
    LOG_INFO("Identifying best HOST_GATHER config value...");
    LOG_BLOCK();
    // Check if was explicitly specified
    auto it_hg = properties.find(intel_npu::npuw::partitioning::host_gather.name());
    std::optional<bool> explicit_hg_value;
    if (it_hg != properties.end()) {
        explicit_hg_value = it_hg->second.as<bool>();
    }

    // Check NPUW_HOST_GATHER:QUANT based on the patterns (for tail vocab)
    ov::npuw::patterns::opt::Context ctx;
    ov::pass::GraphRewrite rewr;
    rewr.add_matcher<ov::npuw::patterns::opt::HostGatherQuantAsymm<ov::op::v0::Constant>>(std::ref(ctx), true);
    rewr.add_matcher<ov::npuw::patterns::opt::HostGatherQuantSymm<ov::op::v0::Constant>>(std::ref(ctx), true);
    rewr.run_on_model(model);

    using CPtr = std::shared_ptr<ov::op::v0::Constant>;
    std::vector<CPtr> to_keep;

    ov::pass::GraphRewrite rewr2;
    ctx.mm_gate = m_cfg.get<::intel_npu::NPUW_MM_GATED>();

    rewr2.add_matcher<ov::npuw::patterns::opt::PreserveConstDictMatMulAsymm>(std::ref(ctx), std::ref(to_keep));
    rewr2.add_matcher<ov::npuw::patterns::opt::PreserveConstDictMatMulFP8>(std::ref(ctx), std::ref(to_keep));
    rewr2.run_on_model(model);
    // FIXME: since 3-model pipeline is the default option, the tail will be separate,
    // so we need to match either head or tail pattern here for host gather quantized feature to work.
    // However, there might be a case where tail pattern is matched, but head is not (for the same model)
    // or vice versa. This would lead to worse performance. Consider adding this check to LLMCompiledModel
    // as well, since there we have uncut model.
    // Head or tail
    const bool pattern_matched = ctx.found_host_gather_quant() || !to_keep.empty();

    // Check the compiler version
    const auto npu_devices = get_plugin()->get_core()->get_property("NPU", ov::available_devices);
    const auto is_suitable_comp = [](int64_t ver, const std::string& arch) {
        if (arch == "3720" || arch == "4000") {
            return ver >= ONEAPI_MAKE_VERSION(7, 21);
        }
        return ver >= ONEAPI_MAKE_VERSION(7, 25);
    };
    const bool compiler_version_enough =
        !npu_devices.empty() &&
        is_suitable_comp(get_plugin()->get_core()->get_property("NPU", ov::intel_npu::compiler_version),
                         get_plugin()->get_core()->get_property("NPU", ov::device::architecture));
    // FIXME: go from
    //     get_plugin()->get_core()->get_property("NPU", ..
    // to
    ///    plugin->get_property(..

    const bool can_enable_hgq = pattern_matched && (compiler_version_enough || npu_devices.empty());

    // Now make a decision based on the checks above
    if (!explicit_hg_value) {
        // Default value is used, can force the best option
        if (can_enable_hgq) {
            LOG_INFO("Forcing quantized tail vocabulary for better performance.");
            return true;
        }
    } else if (!explicit_hg_value.value()) {  // explicit NO
        if (can_enable_hgq) {
            LOG_WARN("Consider removing NPUW_HOST_GATHER:NO from the config for better performance.");
        } else {
            LOG_WARN("Consider enabling NPUW_HOST_GATHER:YES for better performance.");
        }
    } else {  // explicit YES
        if (can_enable_hgq) {
            LOG_WARN("Consider removing NPUW_HOST_GATHER:YES from the config for better performance.");
        }
    }  // explicit_hg_value
    LOG_INFO("DONE.");
    return false;
}

void ov::npuw::CompiledModel::CompiledModelDesc::serialize(ov::npuw::s11n::Stream& stream,
                                                           const ov::npuw::s11n::WeightsContext& ctx,
                                                           std::optional<std::size_t> orc_device_index,
                                                           const ov::npuw::s11n::SubmodelDeserializeCtx* submodel_ctx) {
    using namespace ov::npuw::s11n;

    if (stream.output()) {
        LOG_DEBUG("Serializing CompiledModelDesc...");
    } else {
        LOG_DEBUG("Deserializing CompiledModelDesc...");
    }
    LOG_BLOCK();

    ov::SoPtr<ov::ICompiledModel> imported_compiled_model;
    std::optional<ov::npuw::s11n::SubmodelDeserializeCtx> resolved_submodel_ctx;
    if (orc_device_index.has_value() || (stream.input() && submodel_ctx != nullptr && submodel_ctx->device_by_index)) {
        std::size_t device_index = orc_device_index.value_or(0u);
        stream & device_index;

        bool has_compiled_model = static_cast<bool>(compiled_model);
        stream & has_compiled_model;

        if (stream.output()) {
            if (has_compiled_model) {
                std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
                compiled_model->export_model(buffer);
                auto model_blob = buffer.str();
                stream & model_blob;
            }
        } else {
            NPUW_ASSERT(submodel_ctx != nullptr && "Submodel deserialization context must be provided for ORC import");
            NPUW_ASSERT(submodel_ctx->device_by_index &&
                        "Submodel deserialization context must provide device_by_index for ORC import");
            const auto device = submodel_ctx->device_by_index(device_index);
            const auto import_config = submodel_ctx->import_config_for_device(device);
            if (has_compiled_model) {
                std::string model_blob;
                stream & model_blob;
                std::stringstream buffer(model_blob, std::ios::in | std::ios::out | std::ios::binary);
                imported_compiled_model = submodel_ctx->plugin->get_core()->import_model(buffer, device, import_config);
            }
            compiled_model = imported_compiled_model;
            resolved_submodel_ctx.emplace(submodel_ctx->plugin, device, compiled_model, import_config);
            submodel_ctx = &(*resolved_submodel_ctx);
        }
    }

    stream & replaced_by & param_base & forced_to_fcall & host_gather.dst_idx & host_gather.src_idx &
        host_gather.idx_idx & quant_unpack_gather.dst_idx & quant_unpack_gather.src_w_idx &
        quant_unpack_gather.src_z_idx & quant_unpack_gather.src_s_idx & quant_unpack_gather.idx_idx & spatial;

    // Function calls share pipeline.context with their function body at runtime.
    // There is no need to serialize the compiled moe/attn state for each call –
    // doing so would re-import NPU blobs for every repeated layer (one per call),
    // causing O(N_layers) memory growth on import.  Only the function body
    // (compiled_model is set, or replaced_by is absent) writes/reads state.
    const bool is_fcall = replaced_by.has_value() && !static_cast<bool>(compiled_model);
    if (!is_fcall) {
        ov::npuw::moe::serialize_compiled_state(pipeline.context, stream, submodel_ctx);
        ov::npuw::attn::serialize_compiled_state(pipeline.context, stream, submodel_ctx);

        if (stream.input()) {
            if (ov::npuw::attn::get_compiled_dynamic(pipeline.context) != nullptr) {
                ov::npuw::attn::attach_runtime_behavior(pipeline,
                                                        pipeline.context,
                                                        ov::npuw::attn::BehaviorKind::Dynamic);
            } else if (ov::npuw::attn::get_compiled_pyramid(pipeline.context) != nullptr) {
                ov::npuw::attn::attach_runtime_behavior(pipeline,
                                                        pipeline.context,
                                                        ov::npuw::attn::BehaviorKind::Pyramid);
            } else if (ov::npuw::attn::get_compiled_hfa(pipeline.context) != nullptr) {
                ov::npuw::attn::attach_runtime_behavior(pipeline, pipeline.context, ov::npuw::attn::BehaviorKind::HFA);
            } else if (ov::npuw::moe::get_compiled_experts(pipeline.context) != nullptr) {
                ov::npuw::moe::attach_runtime_behavior(pipeline,
                                                       pipeline.context,
                                                       ov::npuw::moe::BehaviorRole::EXPERTS,
                                                       true);
            } else if (ov::npuw::moe::get_compiled_downstream(pipeline.context) != nullptr) {
                ov::npuw::moe::attach_runtime_behavior(pipeline,
                                                       pipeline.context,
                                                       ov::npuw::moe::BehaviorRole::DOWNSTREAM,
                                                       true);
            }
        }
    }

    auto& closure_desc = closure.get();

    stream & closure_desc.is_remote & closure_desc.closure_uid;

    if (ctx.is_weightless) {
        serialize_weightless(stream, scales, ctx);
        serialize_weightless(stream, zerops, ctx);

        std::size_t closure_size = closure_desc.closure.size();
        stream & closure_size;
        std::vector<ov::Tensor> cpu_closures;
        std::vector<std::size_t> cpu_closure_ids;
        std::vector<ov::npuw::weights::LazyTensor> non_cpu_tensors;
        std::vector<std::size_t> non_cpu_tensors_ids;
        if (stream.output()) {
            for (std::size_t cidx = 0; cidx < closure_desc.closure.size(); ++cidx) {
                if (closure_desc.closure_uid[cidx] == -1) {
                    cpu_closure_ids.push_back(cidx);
                    cpu_closures.push_back(closure_desc.closure[cidx]);
                } else {
                    non_cpu_tensors_ids.push_back(cidx);
                    non_cpu_tensors.push_back(lazy_closure[cidx]);
                }
            }
            stream & cpu_closure_ids;
            serialize_weightless(stream, cpu_closures, ctx);
            stream & non_cpu_tensors_ids & non_cpu_tensors;
        } else {
            closure_desc.closure.resize(closure_size);
            lazy_closure.resize(closure_size);
            stream & cpu_closure_ids;
            serialize_weightless(stream, cpu_closures, ctx);
            std::size_t tidx = 0;
            for (const auto& idx : cpu_closure_ids) {
                closure_desc.closure[idx] = std::move(cpu_closures[tidx++]);
            }
            stream & non_cpu_tensors_ids & non_cpu_tensors;
            std::size_t ltidx = 0;
            for (const auto& idx : non_cpu_tensors_ids) {
                lazy_closure[idx] = std::move(non_cpu_tensors[ltidx++]);
            }
            for (std::size_t cidx = 0; cidx < closure_desc.closure.size(); ++cidx) {
                if (closure_desc.closure_uid[cidx] != -1 && lazy_closure[cidx]) {
                    lazy_closure[cidx].read_weight(ctx);
                }
            }
        }
    } else {
        stream & scales & zerops;

        std::size_t closure_size = closure_desc.closure.size();
        stream & closure_size;
        std::vector<std::size_t> cpu_closure_ids;
        if (stream.output()) {
            std::vector<ov::Tensor> cpu_closures;
            for (std::size_t cidx = 0; cidx < closure_desc.closure.size(); ++cidx) {
                if (closure_desc.closure_uid[cidx] == -1) {
                    cpu_closure_ids.push_back(cidx);
                    cpu_closures.push_back(closure_desc.closure[cidx]);
                }
            }
            stream & cpu_closure_ids;
            for (auto& tensor : cpu_closures) {
                stream & tensor;
            }
        } else {
            stream & cpu_closure_ids;
            closure_desc.closure.resize(closure_size);
            for (const auto& cidx : cpu_closure_ids) {
                stream & closure_desc.closure[cidx];
            }
        }
    }

    LOG_DEBUG("DONE.");
}

ov::npuw::CompiledModel::~CompiledModel() {
    if (m_eval_future.valid()) {
        m_eval_future.wait();
    }
}

void ov::npuw::CompiledModel::export_model(std::ostream& raw_stream) const {
    serialize_orc(raw_stream);
}

std::shared_ptr<ov::npuw::CompiledModel> ov::npuw::CompiledModel::import_model(
    std::istream& stream,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    if (!ov::npuw::orc::is_orc(stream).has_value()) {
        OPENVINO_THROW("Legacy flat NPUW CompiledModel blobs are no longer supported. Re-export the model with the "
                       "current OpenVINO package.");
    }
    return deserialize_orc(stream, plugin, properties);
}

void ov::npuw::CompiledModel::ensure_phase0_compatibility() const {
    for (std::size_t idx = 0; idx < m_compiled_submodels.size(); ++idx) {
        const auto& subm = m_compiled_submodels[idx];
        const auto name = format_subgraph_name(idx, "");
        const auto real_idx = subm.replaced_by.value_or(idx);
        const auto device = submodel_device(real_idx);
        auto fail = [&](const auto& feature) {
            OPENVINO_THROW("Cannot produce ORC-compatible blob: subgraph ",
                           idx,
                           " (\"",
                           name,
                           "\") has ",
                           feature,
                           " - not yet versioned for NPUW phase 0. Recompile without NPUW_ENSURE_COMPATIBILITY or wait "
                           "for a later phase.");
        };

        if (!ov::npuw::util::starts_with(device, "NPU")) {
            fail(std::string("device \"") + device + "\"");
        }
        if (subm.spatial.has_value()) {
            fail("Spatial");
        }
        if (ov::npuw::attn::get_compiled_dynamic(subm.pipeline.context) != nullptr) {
            fail("Attention");
        }
        if (ov::npuw::attn::get_compiled_pyramid(subm.pipeline.context) != nullptr) {
            fail("PyramidAttention");
        }
        if (ov::npuw::attn::get_compiled_hfa(subm.pipeline.context) != nullptr) {
            fail("HostFlashAttention");
        }
        if (ov::npuw::moe::get_compiled_experts(subm.pipeline.context) != nullptr) {
            fail("MoEExperts");
        }
        if (ov::npuw::moe::get_compiled_downstream(subm.pipeline.context) != nullptr) {
            fail("MoEDownstream");
        }
        if (subm.pipeline.runtime_behavior.has_value()) {
            fail("runtime behavior");
        }
    }
}

void ov::npuw::CompiledModel::serialize_orc(std::ostream& stream) const {
    ov::npuw::orc::write_file_header(stream, ov::npuw::orc::schema_npuw::NPUW_ORC_PARTITIONED_SCHEMA);
    serialize_orc_container(stream, true, get_encrypt_callback(m_non_npuw_props));
}

void ov::npuw::CompiledModel::serialize_orc_container(std::ostream& stream,
                                                      bool include_weights_bank,
                                                      const std::function<std::string(const std::string&)>& encrypt,
                                                      const ov::npuw::s11n::BF16Cache* bf16_consts) const {
    using namespace ov::npuw;

    if (m_cfg.get<::intel_npu::NPUW_ENSURE_COMPATIBILITY>()) {
        if (encrypt) {
            OPENVINO_THROW("Cannot produce ORC-compatible blob: encrypted export is not yet supported in "
                           "NPUW_ENSURE_COMPATIBILITY mode");
        }
        ensure_phase0_compatibility();
    }

    bool is_weightless = should_use_weightless_flow(m_non_npuw_props, m_cfg, m_const_to_offset);
    LOG_INFO("Serialization will be done via " << (is_weightless ? "weightless" : "flow with weights") << ".");
    // For top-level CompiledModel export we serialize this model's own BF16 cache.
    // For nested export (e.g. inside LLMCompiledModel) the parent may provide a
    // cache collected before graph splitting / BF16->FP16 conversion, and that
    // propagated view must win so weightless import can reconstruct tensors correctly.
    const auto& bf16_cache = bf16_consts != nullptr ? *bf16_consts : m_bf16_consts;

    ov::AnyMap serializable_props = m_non_npuw_props;
    serializable_props.erase(ov::cache_encryption_callbacks.name());
    s11n::WeightsContext weights_ctx(is_weightless, m_const_to_offset);

    auto write_children = [&](std::ostream& body_stream) {
        for (std::size_t idx = 0; idx < m_compiled_submodels.size(); ++idx) {
            auto& subm = const_cast<CompiledModelDesc&>(m_compiled_submodels[idx]);
            const auto real_idx = subm.replaced_by.value_or(idx);
            const auto device_index = real_idx == idx ? find_device_index(m_dev_list, submodel_device(real_idx)) : 0u;
            ov::npuw::orc::with_leaf_section(body_stream,
                                             CompiledModelDesc::kOrcType,
                                             CompiledModelDesc::kOrcVersion,
                                             [&] {
                                                 auto desc_stream = ov::npuw::s11n::Stream::writer(body_stream);
                                                 subm.serialize(desc_stream, weights_ctx, device_index);
                                             });
        }

        if (include_weights_bank) {
            ov::npuw::orc::with_leaf_section(body_stream, weights::Bank::kOrcType, weights::Bank::kOrcVersion, [&] {
                auto weights_stream = ov::npuw::s11n::Stream::writer(body_stream);
                auto bank_name = m_weights_bank->get_name();
                weights_stream & bank_name;
                if (!is_weightless) {
                    weights_stream&* m_weights_bank;
                }
            });
        }
    };

    const auto root_flags =
        encrypt ? static_cast<ov::npuw::orc::SectionFlags>(ov::npuw::orc::SectionFlag::ENCRYPTED) : 0u;
    ov::npuw::orc::with_section(stream, kOrcType, kOrcVersion, root_flags, [&] {
        ov::npuw::orc::with_leaf_section(stream, ov::npuw::orc::META_SECTION_TYPE, 0u, [&] {
            auto meta_stream = ov::npuw::s11n::Stream::writer(stream);
            meta_stream & m_name;
            meta_stream& inputs() & outputs();
            meta_stream & m_inputs_to_submodels_inputs & m_outputs_to_submodels_outputs & m_param_subscribers &
                m_submodels_input_to_prev_output;
            meta_stream & m_dev_list;
            meta_stream& const_cast<::intel_npu::Config&>(m_cfg);
            meta_stream & serializable_props;
            meta_stream & is_weightless;
            // Persist the BF16 interpretation map that weightless import later feeds
            // into LazyTensor::read_weight() when it decides whether raw bytes should
            // be read as FP16 directly or converted from BF16 source storage.
            meta_stream& const_cast<ov::npuw::s11n::BF16Cache&>(bf16_cache);
            if (encrypt) {
                std::stringstream payload_stream(std::ios::in | std::ios::out | std::ios::binary);
                write_children(payload_stream);
                auto encrypted_payload = encrypt(payload_stream.str());
                meta_stream & encrypted_payload;
            }
        });
        if (!encrypt) {
            write_children(stream);
        }
    });
}

std::shared_ptr<ov::npuw::CompiledModel> ov::npuw::CompiledModel::deserialize_orc(
    std::istream& stream,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    const auto header = ov::npuw::orc::read_file_header(stream);
    if (header.schema_uuid != ov::npuw::orc::schema_npuw::NPUW_ORC_PARTITIONED_SCHEMA) {
        OPENVINO_THROW("Unsupported ORC schema for NPUW CompiledModel");
    }
    return deserialize_orc_container(stream, plugin, properties, true, {});
}

std::shared_ptr<ov::npuw::CompiledModel> ov::npuw::CompiledModel::deserialize_orc_container(
    std::istream& stream,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties,
    bool require_weights_bank,
    const std::function<std::string(const std::string&)>& decrypt) {
    ov::npuw::orc::ScopedReadSection root(stream);
    if (root.header().type != kOrcType || root.header().version > kOrcVersion ||
        ov::npuw::orc::has_flag(root.header().flags, ov::npuw::orc::SectionFlag::LEAF)) {
        OPENVINO_THROW("Unsupported ORC NPUW root section");
    }

    // Variables populated during metadata read but used in the child-reading
    // phase below.  Extracted before the version branch so both paths share
    // the same child-consuming lambdas.
    std::shared_ptr<ov::npuw::CompiledModel> compiled;
    bool is_weightless = false;
    std::string encrypted_payload;

    const bool encrypted = ov::npuw::orc::has_flag(root.header().flags, ov::npuw::orc::SectionFlag::ENCRYPTED);

    // Read the model-level metadata fields.  In v0 these were written as raw
    // s11n bytes directly into the container body; v1 wraps them in an
    // explicit META leaf child so the section tree is fully self-describing.
    auto read_meta_fields = [&]() {
        auto meta_stream = ov::npuw::s11n::Stream::reader(stream);

        std::string model_name;
        ov::ParameterVector parameters;
        ov::NodeVector results;
        meta_stream & model_name & parameters & results;

        auto ov_model = std::make_shared<ov::Model>(ov::as_output_vector(results), parameters, model_name);
        compiled = std::make_shared<ov::npuw::CompiledModel>(ov_model, plugin, true);
        compiled->m_name = std::move(model_name);
        meta_stream & compiled->m_inputs_to_submodels_inputs & compiled->m_outputs_to_submodels_outputs &
            compiled->m_param_subscribers & compiled->m_submodels_input_to_prev_output;
        meta_stream & compiled->m_dev_list;
        meta_stream & compiled->m_cfg;
        compiled->m_cfg.parseEnvVars();
        meta_stream & compiled->m_non_npuw_props;
        meta_stream & is_weightless;
        meta_stream & compiled->m_bf16_consts;
        if (encrypted) {
            meta_stream & encrypted_payload;
        }
    };

    if (root.header().version == 0) {
        // v0: metadata written as raw s11n bytes at the start of the container body.
        read_meta_fields();
    } else if (root.header().version == 1) {
        // v1: metadata wrapped in a META leaf child section.
        ov::npuw::orc::ScopedReadSection meta(stream);
        if (meta.header().type != ov::npuw::orc::META_SECTION_TYPE ||
            !ov::npuw::orc::has_flag(meta.header().flags, ov::npuw::orc::SectionFlag::LEAF)) {
            OPENVINO_THROW("Expected ORC NPUW metadata section, got type ", meta.header().type);
        }
        read_meta_fields();
        meta.expect_end();
    } else {
        OPENVINO_THROW("Unsupported ORC NPUW PartitionedModel version ", root.header().version);
    }

    compiled->m_import_weights_ctx = make_import_weights_ctx(properties, is_weightless, compiled->m_bf16_consts);
    bool have_weights = false;
    auto peek_child_header = [](std::istream& child_source) {
        const auto saved = child_source.tellg();
        auto peek_stream = ov::npuw::s11n::Stream::reader(child_source);
        ov::npuw::orc::SectionHeader header;
        peek_stream & header;
        child_source.seekg(saved);
        return header;
    };
    auto consume_submodel = [&](std::istream& child_source) {
        ov::npuw::orc::ScopedReadSection child(child_source);
        if (child.header().type != CompiledModelDesc::kOrcType) {
            OPENVINO_THROW("Unexpected ORC child type ID ", child.header().type, " in NPUW CompiledModel container");
        }
        if (child.header().version != CompiledModelDesc::kOrcVersion) {
            OPENVINO_THROW("Unsupported ORC NPUW subgraph version ", child.header().version);
        }

        compiled->m_compiled_submodels.emplace_back();
        auto& submodel = compiled->m_compiled_submodels.back();
        auto child_stream = ov::npuw::s11n::Stream::reader(child_source);
        ov::npuw::s11n::SubmodelDeserializeCtx submodel_ctx(
            plugin,
            submodel.compiled_model,
            [&](std::size_t device_index) {
                return compiled->m_dev_list.at(device_index);
            },
            [&](const std::string& device) {
                return make_submodel_import_config(device, compiled->m_cfg);
            });
        submodel.serialize(child_stream, compiled->m_import_weights_ctx, std::nullopt, &submodel_ctx);
        child.expect_end();
    };

    auto consume_weights_bank = [&](std::istream& child_source) {
        ov::npuw::orc::ScopedReadSection child(child_source);
        if (child.header().type != weights::Bank::kOrcType) {
            OPENVINO_THROW("Unexpected ORC child type ID ", child.header().type, " in NPUW CompiledModel container");
        }
        if (child.header().version != weights::Bank::kOrcVersion) {
            OPENVINO_THROW("Unsupported ORC NPUW weights version ", child.header().version);
        }

        auto child_stream = ov::npuw::s11n::Stream::reader(child_source);
        std::string bank_name;
        child_stream & bank_name;
        compiled->m_weights_bank = ov::npuw::weights::bank(bank_name, compiled->get_plugin()->get_core(), "");
        if (is_weightless) {
            child.expect_end();
            compiled->finalize_weights_bank();
        } else {
            child_stream& * compiled->m_weights_bank;
            child.expect_end();
            compiled->reconstruct_closure();
        }
        have_weights = true;
    };

    if (encrypted) {
        root.expect_end();

        const auto decrypt_fn = decrypt ? decrypt : get_decrypt_callback_or_throw(properties);
        std::istringstream decrypted_stream(std::move(decrypt_fn(encrypted_payload)));
        while (decrypted_stream.peek() != std::char_traits<char>::eof() &&
               peek_child_header(decrypted_stream).type == CompiledModelDesc::kOrcType) {
            consume_submodel(decrypted_stream);
        }
        if (require_weights_bank) {
            if (decrypted_stream.peek() == std::char_traits<char>::eof()) {
                OPENVINO_THROW("Missing ORC weights bank container");
            }
            consume_weights_bank(decrypted_stream);
        }
    } else {
        while (!root.done() && peek_child_header(stream).type == CompiledModelDesc::kOrcType) {
            consume_submodel(stream);
        }
        if (require_weights_bank) {
            if (root.done()) {
                OPENVINO_THROW("Missing ORC weights bank container");
            }
            consume_weights_bank(stream);
        } else if (!root.done()) {
            OPENVINO_THROW("Unexpected ORC child after CompiledModelDesc containers");
        }
        root.expect_end();
    }

    if (require_weights_bank && !have_weights) {
        OPENVINO_THROW("Missing ORC weights bank container");
    }

    compiled->implement_properties();
    return compiled;
}

void ov::npuw::CompiledModel::serialize(std::ostream& stream, const ov::npuw::s11n::CompiledContext& enc_ctx) const {
    if (enc_ctx.encrypted) {
        NPUW_ASSERT(enc_ctx.encrypt && "Encryption function isn't provided!");
    }
    // Preserve the caller-provided BF16 cache for nested serialization.
    // LLMCompiledModel relies on this to pass original-model BF16 metadata down
    // into child CompiledModel blobs.
    serialize_orc_container(stream,
                            false,
                            enc_ctx.encrypted ? enc_ctx.encrypt : std::function<std::string(const std::string&)>{},
                            &enc_ctx.bf16_consts);
}

std::shared_ptr<ov::npuw::CompiledModel> ov::npuw::CompiledModel::deserialize(
    std::istream& stream,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties,
    const ov::npuw::s11n::CompiledContext& enc_ctx) {
    if (enc_ctx.encrypted) {
        NPUW_ASSERT(enc_ctx.decrypt && "Decryption function isn't provided!");
    }
    return deserialize_orc_container(
        stream,
        plugin,
        properties,
        false,
        enc_ctx.encrypted ? enc_ctx.decrypt : std::function<std::string(const std::string&)>{});
}

void ov::npuw::CompiledModel::reconstruct_closure() {
    for (size_t idx = 0; idx < m_compiled_submodels.size(); ++idx) {
        auto& comp_model_desc = m_compiled_submodels[idx];

        // Skip optimized out and non-functions
        if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
            continue;
        }

        const auto real_idx = comp_model_desc.replaced_by.value_or(idx);
        auto& desc_closure = comp_model_desc.closure.get();

        for (std::size_t cidx = 0; cidx < desc_closure.closure.size(); ++cidx) {
            if (desc_closure.closure[cidx]) {
                // host-side closure - already set, do nothing
                NPUW_ASSERT(!desc_closure.is_remote[cidx]);
                continue;
            }
            NPUW_ASSERT(desc_closure.closure_uid[cidx] != -1);
            desc_closure.closure[cidx] = m_weights_bank->get(desc_closure.closure_uid[cidx], submodel_device(real_idx));
        }
    }
}

std::size_t ov::npuw::CompiledModel::num_submodels() const {
    return m_compiled_submodels.size();
}

bool ov::npuw::CompiledModel::attention_dynamic_enabled() const {
    return m_cfg.get<::intel_npu::NPUW_ATTN_DYN>();
}

bool ov::npuw::CompiledModel::attention_no_copy() const {
    return m_cfg.get<::intel_npu::NPUW_ATTN_NO_COPY>();
}

bool ov::npuw::CompiledModel::has_pipeline_model() const {
    return m_compiled_pipeline_model != nullptr;
}

bool ov::npuw::CompiledModel::has_hfa_pipeline_model() const {
    return m_pipeline_has_hfa;
}

size_t& ov::npuw::CompiledModel::get_prefill_iteration() const {
    return m_current_prefill_iteration;
}

std::shared_ptr<ov::npuw::weights::Bank> ov::npuw::CompiledModel::get_weights_bank() const {
    return m_weights_bank;
}

void ov::npuw::CompiledModel::set_weights_bank(std::shared_ptr<ov::npuw::weights::Bank> bank) {
    m_weights_bank = std::move(bank);
}

::intel_npu::elf_binary ov::npuw::CompiledModel::nlp_host_flash_attention_pipeline(
    ov::npuw::compiled::HostFlashAttention* hfa) {
    auto create_accumulator_broadcast_model = [](const uint64_t& buffer_size) {
        // -------------------------------------------------------------------
        // 1. Prepare Source Data (Matching offsets: U32 scalar + I64 shape)
        // -------------------------------------------------------------------
        // -------------------------------------------------------------------
        // 1. Prepare Source Data (Matching offsets: U32 scalar + I64 shape)
        // -------------------------------------------------------------------
        uint32_t raw_scalar_data{0u};                                          // Layer id="0" data
        int64_t target_shape_data{static_cast<int64_t>(buffer_size >> 2ull)};  // Layer id="1" data

        // -------------------------------------------------------------------
        // 2. Build the Graph Nodes
        // -------------------------------------------------------------------
        // Layer id="0": raw_scalar_source
        auto raw_scalar_source =
            std::make_shared<ov::opset1::Constant>(ov::element::u32, ov::Shape{1}, &raw_scalar_data);
        raw_scalar_source->set_friendly_name("raw_scalar_source");

        // Layer id="1": target_shape_array
        auto target_shape_array =
            std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{1}, &target_shape_data);
        target_shape_array->set_friendly_name("target_shape_array");

        // Layer id="2": past_acc (Broadcast operation)
        auto past_acc = std::make_shared<ov::opset3::Broadcast>(raw_scalar_source,
                                                                target_shape_array,
                                                                ov::op::BroadcastType::NUMPY);
        past_acc->set_friendly_name("past_acc");

        // Layer id="3": Result
        auto result = std::make_shared<ov::opset1::Result>(past_acc);
        result->set_friendly_name("past_acc_result");

        // -------------------------------------------------------------------
        // 3. Assemble Model Container
        // -------------------------------------------------------------------
        ov::ResultVector results = {result};
        ov::ParameterVector parameters = {};  // Complete constants graph

        auto model{std::make_shared<ov::Model>(results, parameters, "past_acc_init")};
        return model;
    };

    auto create_max_sum_initiation_model = [](const size_t& buffer_size, uint32_t max_raw_scalar_data) {
        uint32_t sum_raw_scalar_data{0u};  // Maps to sum_raw_scalar_source
        int64_t target_shape_data{static_cast<int64_t>(buffer_size >> 2ull)};
        using namespace ov::opset11;
        // -------------------------------------------------------------------
        // 2. Define Subgraph A: Max Past Component
        // -------------------------------------------------------------------
        // Layer id="4": max_raw_scalar_source
        auto max_raw_scalar_source = std::make_shared<Constant>(ov::element::u32, ov::Shape{1}, &max_raw_scalar_data);
        max_raw_scalar_source->set_friendly_name("max_raw_scalar_source");

        // Layer id="5": max_target_shape_array
        auto max_target_shape_array = std::make_shared<Constant>(ov::element::i64, ov::Shape{1}, &target_shape_data);
        max_target_shape_array->set_friendly_name("max_target_shape_array");

        // Layer id="6": past_max (Broadcast numpy mode)
        auto past_max =
            std::make_shared<Broadcast>(max_raw_scalar_source, max_target_shape_array, ov::op::BroadcastType::NUMPY);
        past_max->set_friendly_name("past_max");

        // Layer id="7": Result
        auto result_max = std::make_shared<Result>(past_max);
        result_max->set_friendly_name("past_max_result");

        // -------------------------------------------------------------------
        // 3. Define Subgraph B: Sum / D Past Component
        // -------------------------------------------------------------------
        // Layer id="8": sum_raw_scalar_source
        auto sum_raw_scalar_source = std::make_shared<Constant>(ov::element::u32, ov::Shape{1}, &sum_raw_scalar_data);
        sum_raw_scalar_source->set_friendly_name("sum_raw_scalar_source");

        // Layer id="9": sum_target_shape_array
        auto sum_target_shape_array = std::make_shared<Constant>(ov::element::i64, ov::Shape{1}, &target_shape_data);
        sum_target_shape_array->set_friendly_name("sum_target_shape_array");

        // Layer id="10": past_d (Broadcast numpy mode)
        auto past_d =
            std::make_shared<Broadcast>(sum_raw_scalar_source, sum_target_shape_array, ov::op::BroadcastType::NUMPY);
        past_d->set_friendly_name("past_d");

        // Layer id="11": Result
        auto result_d = std::make_shared<Result>(past_d);
        result_d->set_friendly_name("past_d_result");

        // -------------------------------------------------------------------
        // 4. Construct and Return the Complete OpenVINO Model
        // -------------------------------------------------------------------
        ov::ResultVector results = {result_max, result_d};
        ov::ParameterVector parameters = {};  // Model has no dynamic parameter inputs

        auto model = std::make_shared<ov::Model>(results, parameters, "max_d_init");
        return model;
    };

    auto get_compiled_elf = [](std::shared_ptr<ov::Model>& model, ::intel_npu::elf_binary& blob) {
        ov::Core core;
        auto compiled_model{core.compile_model(model, "NPU")};

        std::stringstream model_stream;
        compiled_model.export_model(model_stream);
        std::string model_str{model_stream.str()};
        std::vector<uint8_t> buffer(model_str.begin(), model_str.end());
        blob = std::move(buffer);
    };

    if (hfa) {
        auto plugin = get_npuw_plugin();
        auto core = plugin->get_core();
        auto npuw_plugin = std::dynamic_pointer_cast<const ::intel_npu::Plugin>(plugin);

        ::intel_npu::elf_binary hfa_final_blob{};
        std::vector<::intel_npu::elf_binary> hfa_intermediate_blob{};

        const auto query_size{static_cast<uint32_t>(hfa->_sdpa_attention_info._query_size)};
        const auto context_size{static_cast<uint32_t>(hfa->_sdpa_attention_info._context_size)};
        const auto mask_port_idx{hfa->_sdpa_attention_info._tile_input_indices.mask};
        const auto key_port_idx{hfa->_sdpa_attention_info._tile_input_indices.k};
        const auto value_port_idx{hfa->_sdpa_attention_info._tile_input_indices.v};

        const auto hfa_tile_stages{context_size / query_size};

        if (hfa_tile_stages) {
            m_nlp_controlflow_branch_select_size = hfa_tile_stages - 1ull;
        }

        ::intel_npu::elf_binary acc_init_elf{};
        ::intel_npu::elf_binary max_d_init_elf{};


        if (hfa->_compiled_final_tile_model) {
            std::stringstream model_stream;
            hfa->_compiled_final_tile_model->export_model(model_stream);
            std::string model_str{model_stream.str()};
            std::vector<uint8_t> hfa_buffer(model_str.begin(), model_str.end());
            hfa_final_blob = hfa_buffer;            
        }

        if (hfa->_compiled_final_tile_model_strided) {
            std::stringstream model_stream;
            hfa->_compiled_final_tile_model_strided->export_model(model_stream);
            std::string model_str{model_stream.str()};
            std::vector<uint8_t> hfa_buffer(model_str.begin(), model_str.end());
            hfa_final_blob = hfa_buffer;
            
            const size_t tile_offset{query_size * m_nlp_controlflow_branch_select_size};
            std::stringstream stat_dma_ss;
            stat_dma_ss << "i" << mask_port_idx << ":stride[" << context_size << "," << (context_size * query_size)
                        << "] offset[" << tile_offset << "]";
            
            std::string stat_dma_option{stat_dma_ss.str()};

            npuw_plugin->schedule_builder_proceed(hfa_final_blob,
                                                  ::intel_npu::BuilderOption::static_dma,
                                                  stat_dma_option);            
        }

        if (hfa->_compiled_tile_model) {
            const auto acc_port_idx{hfa->_sdpa_attention_info._tile_input_indices.acc};            
            const auto max_port_idx{hfa->_sdpa_attention_info._tile_input_indices.max};
            const auto d_port_idx{hfa->_sdpa_attention_info._tile_input_indices.d};
            const auto k_port_idx{hfa->_sdpa_attention_info._tile_input_indices.k};
            const auto v_port_idx{hfa->_sdpa_attention_info._tile_input_indices.v};
            const auto Q_idx{hfa->_sdpa_attention_info._tile_input_indices.q};
            const auto mask_idx{hfa->_sdpa_attention_info._tile_input_indices.mask};

            const auto acc_port_out_idx{hfa->_sdpa_attention_info._tile_output_indices.acc};
            const auto max_port_out_idx{hfa->_sdpa_attention_info._tile_output_indices.max};
            const auto d_port_out_idx{hfa->_sdpa_attention_info._tile_output_indices.d};

            auto& inputs{hfa->_compiled_tile_model->inputs()};

            auto get_port_size_in_bytes = [&](const size_t& idx, ov::element::Type& type) {
                auto& port{inputs[idx]};

                auto shape{port.get_shape()};
                auto bytes{ov::shape_size(shape)};
                type = port.get_element_type();
                bytes *= (type.bitwidth() >> 3u);
                return bytes;
            };

            ov::element::Type acc_type{};
            ov::element::Type max_type{};
            ov::element::Type d_type{};

            auto acc_bytes{get_port_size_in_bytes(acc_port_idx, acc_type)};
            auto max_bytes{get_port_size_in_bytes(max_port_idx, max_type)};
            auto d_bytes{get_port_size_in_bytes(d_port_idx, d_type)};

            get_compiled_elf(create_accumulator_broadcast_model(acc_bytes), acc_init_elf);

            npuw_plugin->schedule_builder_proceed(acc_init_elf, ::intel_npu::BuilderOption::memset, "");
            uint32_t max_raw_scalar_data{0xfbfffbffu};

            if (max_type == ov::element::f32) {
                auto neg_inf{-std::numeric_limits<float>::infinity()};
                max_raw_scalar_data = *reinterpret_cast<uint32_t*>(&neg_inf);
            }

            get_compiled_elf(create_max_sum_initiation_model(max_bytes, max_raw_scalar_data), max_d_init_elf);

            npuw_plugin->schedule_builder_proceed(max_d_init_elf, ::intel_npu::BuilderOption::memset, "");
                        

            std::stringstream model_stream;
            hfa->_compiled_tile_model->export_model(model_stream);
            std::string model_str{model_stream.str()};
            std::vector<uint8_t> hfa_buffer(model_str.begin(), model_str.end());

          
            {
                auto& blob_inputs{hfa->_compiled_tile_model->inputs()};
                auto& blob_outputs{hfa->_compiled_tile_model->outputs()};

                std::stringstream io_name_parameters{};

                auto add_port_names = [&](const std::vector<ov::Output<const ov::Node>>& ports, bool is_input) {
                    size_t idx{};

                    for (auto& port : ports) {
                        auto name{port.get_node()->get_friendly_name()};
                        if (is_input && (idx == k_port_idx) || ((idx == v_port_idx))) {
                            name = "PAST_" + name;
                        }

                        io_name_parameters << name << "\n";
                        idx++;
                    }
                };

                add_port_names(blob_inputs,true);
                add_port_names(blob_outputs,false);

                const auto io_name_parameters_str{io_name_parameters.str()};

                npuw_plugin->schedule_builder_proceed(hfa_buffer,
                                                      ::intel_npu::BuilderOption::io_name,
                                                      io_name_parameters_str);
            }            

            hfa_intermediate_blob.resize(m_nlp_controlflow_branch_select_size);
            size_t tile_offset{};

            auto key_seq_dim{hfa->_sdpa_attention_info._k_seq_dim};
            auto value_seq_dim{hfa->_sdpa_attention_info._v_seq_dim};

            auto key_shape{inputs[key_port_idx].get_shape()};
            auto value_shape{inputs[value_port_idx].get_shape()};

            key_shape[key_seq_dim] = m_nlp_controlflow_branch_select_size * query_size;
            value_shape[value_seq_dim] = key_shape[key_seq_dim];

            auto get_kv_stride_string = [](auto& shape) {
                std::stringstream ss;
                uint32_t entries {};
                size_t stride{1ull};
                for (auto it{std::rbegin(shape)}; it != std::rend(shape); ++it, ++entries) {
                    if (entries) {
                        ss << ",";
                    }
                    stride *= (*it);
                    ss << stride;
                }

                return ss.str();
            };

            const auto key_stride_str{get_kv_stride_string(key_shape)};
            const auto value_stride_str{get_kv_stride_string(value_shape)};

            for (size_t i{}; i < m_nlp_controlflow_branch_select_size; ++i) {
                hfa_intermediate_blob[i] = hfa_buffer;

                std::stringstream stat_dma_ss;
                stat_dma_ss << "i" << mask_port_idx << ":stride[" << context_size << "," << (context_size * query_size)
                            << "] offset[" << tile_offset << "]";

                stat_dma_ss << "\ni" << key_port_idx << ":stride[" << key_stride_str << "] offset[0,"
                            << tile_offset
                            << "]";

                stat_dma_ss << "\ni" << value_port_idx << ":stride[" << value_stride_str << "] offset["
                            << tile_offset << "]";
              
                std::string stat_dma_option{stat_dma_ss.str()};

                npuw_plugin->schedule_builder_proceed(hfa_intermediate_blob[i],
                                                      ::intel_npu::BuilderOption::static_dma,
                                                      stat_dma_option);

                std::stringstream io_reuse_params{};
                io_reuse_params << "[INLINE]";
                io_reuse_params << "\n" << acc_port_out_idx << ":" << acc_port_idx;
                io_reuse_params << "\n" << max_port_out_idx << ":" << max_port_idx;
                io_reuse_params << "\n" << d_port_out_idx << ":" << d_port_idx;

                npuw_plugin->schedule_builder_proceed(hfa_intermediate_blob[i],
                                                      ::intel_npu::BuilderOption::io_reuse,
                                                      io_reuse_params.str());
                tile_offset += query_size;
            }

            // Build the HFA pipeline
            std::vector<::intel_npu::elf_binary> hfa_pipeline_blobs;
            hfa_pipeline_blobs.reserve((hfa_tile_stages << 1u) + 2u);
            hfa_pipeline_blobs.push_back(acc_init_elf);
            hfa_pipeline_blobs.push_back(max_d_init_elf);
            std::stringstream io_consolidate_ss;

            const uint32_t stage_offset{2u};
            std::vector<uint32_t> hfa_branch;
            io_consolidate_ss << "[INLINE]\ns0:s2\no0:i" << acc_port_idx << "\ns1:s2\no0:i" << max_port_idx << "\no1:i"
                              << d_port_idx;

            const auto stage_end{hfa_tile_stages - 1u};

            for (uint32_t stage{}; stage < stage_end; ++stage) {
                hfa_pipeline_blobs.push_back(hfa_intermediate_blob[stage]);
                hfa_branch.push_back(stage);

                auto a{stage + stage_offset};

                if (stage) {
                    const auto am1{a - 1u};
                    
                    io_consolidate_ss << "\ns" << am1 << ":s" << a << "\n";
                    io_consolidate_ss << "i" << k_port_idx << ":i" << k_port_idx << "\n";
                    io_consolidate_ss << "i" << v_port_idx << ":i" << v_port_idx << "\n";
                    io_consolidate_ss << "i" << acc_port_idx << ":i" << acc_port_idx << "\n";
                    io_consolidate_ss << "i" << max_port_idx << ":i" << max_port_idx << "\n";
                    io_consolidate_ss << "i" << d_port_idx << ":i" << d_port_idx << "\n";
                    io_consolidate_ss << "i" << Q_idx << ":i" << Q_idx << "\n";
                    io_consolidate_ss << "i" << mask_port_idx << ":i" << mask_port_idx;                  
                }
            }

            {
                hfa_pipeline_blobs.push_back(hfa_final_blob);
                hfa_branch.push_back(stage_end);

                auto a{stage_end + stage_offset};

                if (stage_end) {
                    const auto am1{a - 1u};

                    io_consolidate_ss << "\ns" << am1 << ":s" << a << "\n";                                
                    io_consolidate_ss << "i" << acc_port_idx << ":i" << acc_port_idx << "\n";
                    io_consolidate_ss << "i" << max_port_idx << ":i" << max_port_idx << "\n";
                    io_consolidate_ss << "i" << d_port_idx << ":i" << d_port_idx << "\n";
                    io_consolidate_ss << "i" << Q_idx << ":i" << Q_idx << "\n";
                    io_consolidate_ss << "i" << mask_port_idx << ":i" << mask_port_idx;
                }
            }

            const auto final_stage{hfa_branch.back() + stage_offset};
            hfa_branch.pop_back();
            std::stringstream branch_ss;
            branch_ss << "[INLINE]";

            for (auto& stage : hfa_branch) {
                const auto stage_entry{stage + 1u};

                branch_ss << "\ns" << stage_entry << ":s" << (stage_entry + 1u) << "\n";
                branch_ss << "s" << stage_entry << ":s" << final_stage;
            }

            std::vector<::intel_npu::BuilderOption> options(3);
            options[0] = ::intel_npu::BuilderOption::fuse_elf;
            options[1] = ::intel_npu::BuilderOption::branch;
            options[2] = ::intel_npu::BuilderOption::io_consolidate;
            std::vector<std::string> option_parameters(3);
            option_parameters[1] = branch_ss.str();
            option_parameters[2] = io_consolidate_ss.str();
            npuw_plugin->schedule_builder_proceed(hfa_pipeline_blobs, options, option_parameters);

            auto& pipeline_blob{hfa_pipeline_blobs[0]};

            npuw_plugin->schedule_builder_proceed(pipeline_blob, ::intel_npu::BuilderOption::compress, "");
            
            std::map<uint32_t, uint32_t> pipeline_global_inputs;
            std::map<uint32_t, uint32_t> pipeline_global_outputs;

            npuw_plugin->schedule_builder_get_io_mapping(pipeline_blob,
                                                         pipeline_global_inputs,
                                                         pipeline_global_outputs,
                                                         hfa->pipeline_parameters);         

            {
                const auto Q_in_idx{hfa->_sdpa_attention_info._sdpa_indices.query};
                const auto K_in_idx{hfa->_sdpa_attention_info._sdpa_indices.present_key};
                const auto V_in_idx{hfa->_sdpa_attention_info._sdpa_indices.present_value};
                const auto mask_in_idx{hfa->_sdpa_attention_info._sdpa_indices.attention_mask};
                size_t past_K_in_idx{~0ull};
                size_t past_V_in_idx{~0ull};

                if (hfa->_sdpa_attention_info._sdpa_indices.past_key_blocks.size()) {
                    past_K_in_idx = hfa->_sdpa_attention_info._sdpa_indices.past_key_blocks[0];
                }

                if (hfa->_sdpa_attention_info._sdpa_indices.past_value_blocks.size()) {
                    past_V_in_idx = hfa->_sdpa_attention_info._sdpa_indices.past_value_blocks[0];
                }

                auto& final_inputs{hfa->_compiled_final_tile_model->inputs()};
                size_t indx{};
                std::map<size_t, std::string> name_lut;

                for (auto& in : final_inputs) {
                    auto name{in.get_node()->get_friendly_name()};
                    name_lut[indx++] = name;
                }

                const auto Q_port_name{name_lut[hfa->_sdpa_attention_info._tile_input_indices.q]};
                const auto K_port_name{name_lut[hfa->_sdpa_attention_info._tile_input_indices.k]};
                const auto V_port_name{name_lut[hfa->_sdpa_attention_info._tile_input_indices.v]};
                const auto ACC_port_name{name_lut[hfa->_sdpa_attention_info._tile_input_indices.acc]};
                const auto MAX_port_name{name_lut[hfa->_sdpa_attention_info._tile_input_indices.max]};
                const auto D_port_name{name_lut[hfa->_sdpa_attention_info._tile_input_indices.d]};
                const auto Mask_port_name{name_lut[hfa->_sdpa_attention_info._tile_input_indices.mask]};
                const auto past_K_port_name{"PAST_" + K_port_name};
                const auto past_V_port_name{"PAST_" + V_port_name};

                m_hfa_port_names.Q = Q_port_name;
                m_hfa_port_names.K = K_port_name;
                m_hfa_port_names.V = V_port_name;
                m_hfa_port_names.mask = Mask_port_name;

                
                hfa->input_port_pipeline_mapping[past_K_in_idx] = hfa->pipeline_parameters[past_K_port_name].back();
                hfa->input_port_pipeline_mapping[past_V_in_idx] = hfa->pipeline_parameters[past_V_port_name].back();

                hfa->input_port_pipeline_mapping[Q_in_idx] = hfa->pipeline_parameters[Q_port_name].front();
                hfa->input_port_pipeline_mapping[K_in_idx] = hfa->pipeline_parameters[K_port_name].back();
                hfa->input_port_pipeline_mapping[V_in_idx] = hfa->pipeline_parameters[V_port_name].back();
                hfa->input_port_pipeline_mapping[mask_in_idx] = hfa->pipeline_parameters[Mask_port_name].front();
                const std::string controflow_section_port_name{"#[NLP]_controlflow_select"};
                m_nlp_branch_select_port_name = controflow_section_port_name;
                hfa->shared_input_parameters[hfa->pipeline_parameters[controflow_section_port_name].front()] =
                    controflow_section_port_name;

                hfa->shared_input_parameters[hfa->pipeline_parameters[Q_port_name].front()] = Q_port_name;
                hfa->shared_input_parameters[hfa->pipeline_parameters[K_port_name].front()] = K_port_name;
                hfa->shared_input_parameters[hfa->pipeline_parameters[V_port_name].front()] = V_port_name;
                hfa->shared_input_parameters[hfa->pipeline_parameters[Mask_port_name].front()] = Mask_port_name;
                hfa->shared_input_parameters[hfa->pipeline_parameters[ACC_port_name].front()] = ACC_port_name;
                hfa->shared_input_parameters[hfa->pipeline_parameters[MAX_port_name].front()] = MAX_port_name;
                hfa->shared_input_parameters[hfa->pipeline_parameters[D_port_name].front()] = D_port_name;
                
                m_pipeline_connected_inputs[ACC_port_name]++;
                m_pipeline_connected_inputs[MAX_port_name]++;
                m_pipeline_connected_inputs[D_port_name]++;
                m_pipeline_connected_inputs[Mask_port_name]++;
                m_pipeline_connected_inputs[Q_port_name]++;
            }
            return std::move(pipeline_blob);
        }
    }

    return ::intel_npu::elf_binary{};
}

void ov::npuw::CompiledModel::finalize_weights_bank() {
    LOG_INFO("Finalizing weights bank...");
    std::shared_future<void> weights_bank_evaluation = std::async(std::launch::async, [&]() {
        // Register lazy tensors
        for (std::size_t idx = 0; idx < m_compiled_submodels.size(); ++idx) {
            auto& comp_model_desc = m_compiled_submodels[idx];

            // Skip optimized out and non-functions
            if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
                continue;
            }

            const auto real_idx = comp_model_desc.replaced_by.value_or(idx);

            for (std::size_t tidx = 0; tidx < comp_model_desc.lazy_closure.size(); ++tidx) {
                if (comp_model_desc.closure.unsafe_get().closure[tidx]) {
                    continue;  // host-side closure
                }
                comp_model_desc.closure.unsafe_get().closure_uid[tidx] =
                    m_weights_bank->registerLT(comp_model_desc.lazy_closure[tidx], submodel_device(real_idx));
            }
        }

        // Evaluate and allocate all LazyTensors inside the bank
        m_weights_bank->evaluate_and_allocate();

        // Set evaluated and allocated ov::Tensors to closures
        for (size_t idx = 0; idx < m_compiled_submodels.size(); ++idx) {
            auto& comp_model_desc = m_compiled_submodels[idx];

            // Skip optimized out and non-functions
            if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
                continue;
            }

            const auto real_idx = comp_model_desc.replaced_by.value_or(idx);
            auto& desc_closure = comp_model_desc.closure.unsafe_get();

            for (std::size_t tidx = 0; tidx < desc_closure.closure.size(); ++tidx) {
                if (desc_closure.closure[tidx]) {
                    // host-side closure - already set, do nothing
                    desc_closure.is_remote[tidx] = false;
                    continue;
                }
                const auto& uid = desc_closure.closure_uid[tidx];
                NPUW_ASSERT(uid != -1);  // All tensors should be registered at this point
                desc_closure.closure[tidx] = m_weights_bank->get(uid, submodel_device(real_idx));
                // FIXME: find a more reliable way to do so
                desc_closure.is_remote[tidx] = m_weights_bank->is_remote(uid);
            }
        }

        m_import_weights_ctx.reset();
    });

    m_eval_future = weights_bank_evaluation;

    for (size_t idx = 0; idx < m_compiled_submodels.size(); ++idx) {
        auto& comp_model_desc = m_compiled_submodels[idx];

        // Skip optimized out and non-functions
        if (!comp_model_desc.compiled_model && !comp_model_desc.replaced_by) {
            continue;
        }

        comp_model_desc.closure.set_future(weights_bank_evaluation);
    }

    LOG_INFO("Done.");
}

void ov::npuw::CompiledModel::store_const_offsets(const std::shared_ptr<ov::Model>& model) {
    for (auto&& node_ptr : model->get_ordered_ops()) {
        if (ov::op::util::is_constant(node_ptr)) {
            const auto& c = std::static_pointer_cast<ov::op::v0::Constant>(node_ptr);
            auto rt_info = c->get_rt_info();
            auto weightless_cache_attr = rt_info.find(ov::WeightlessCacheAttribute::get_type_info_static());
            if (weightless_cache_attr == rt_info.end()) {
                continue;
            }
            std::size_t offset = weightless_cache_attr->second.as<ov::WeightlessCacheAttribute>().bin_offset;
            auto data_ptr = c->get_data_ptr();
            auto inserted = m_const_to_offset.insert({data_ptr, offset});
            if (!inserted.second) {
                NPUW_ASSERT(inserted.first->second == offset &&
                            "Model contains two constants with same pointer and different offset!");
            }
        }
    }
}

void ov::npuw::CompiledModel::detach_memory() {
    LOG_INFO("Detaching model & weight memory...");
    LOG_BLOCK();

    const bool no_runtime_fallback = !m_cfg.get<::intel_npu::NPUW_FALLBACK_EXEC>();

    for (size_t idx = 0; idx < m_compiled_submodels.size(); ++idx) {
        auto& comp_model_desc = m_compiled_submodels[idx];
        auto& proto_comp_model_desc = m_compiled_submodels[comp_model_desc.replaced_by.value_or(idx)];
        if (!proto_comp_model_desc.model || !proto_comp_model_desc.compiled_model) {
            continue;  // optimized-out OR already cleared - skip
        }

        // Ask the failsafe wrapper whether runtime fallover to another device is still
        // possible.  If it is, keep the ov::Model alive so the factory (which captures
        // it by weak_ptr) can recompile.  For non-failsafe compiled models (single
        // device) there is no fallback, so we can always release.
        bool can_clear = true;
        auto failsafe =
            std::dynamic_pointer_cast<ov::npuw::failsafe::CompiledModel>(proto_comp_model_desc.compiled_model._ptr);
        if (failsafe && !failsafe->is_at_last_device() && !no_runtime_fallback) {
            can_clear = false;
        }

        if (can_clear) {
            LOG_INFO("No fallback expected - clear the OV model for Subgraph[" << idx << "]");
            proto_comp_model_desc.model.reset();
        } else {
            LOG_INFO("Runtime fallback still possible - keeping OV model for Subgraph[" << idx << "]");
        }

        // No need to clear pyramid attention data - it's self-contained!
        // The _models_to_compile is already cleared in set_compiled_models()
        // and compiled::PyramidAttention only stores _compiled_models (not original models)
    }
    LOG_INFO("Done");
}

std::string ov::npuw::CompiledModel::global_mem_device() const {
    // Force globally set device if set
    const std::string& device_alloc = m_cfg.get<::intel_npu::NPUW_WEIGHTS_BANK_ALLOC>();
    if (!device_alloc.empty()) {
        return device_alloc;
    }

    // Check if there is at least 1 NPU submodel
    for (std::size_t idx = 0; idx < m_compiled_submodels.size(); ++idx) {
        auto& comp_model_desc = m_compiled_submodels[idx];
        if (!comp_model_desc.compiled_model) {
            continue;
        }
        if (ov::npuw::util::starts_with(submodel_device(idx), "NPU")) {
            return "NPU";
        }
    }

    return "CPU";
}

std::string ov::npuw::CompiledModel::funcall_mem_device(const std::size_t idx) const {
    // Force globally set device if set
    const std::string& device_alloc = m_cfg.get<::intel_npu::NPUW_WEIGHTS_BANK_ALLOC>();
    if (!device_alloc.empty()) {
        return device_alloc;
    }

    return submodel_device(idx);
}

void ov::npuw::CompiledModel::remove_long_output_names(const std::shared_ptr<ov::Model>& model) {
    NPUW_ASSERT(model.get() != nullptr);
    for (auto node : model->get_ordered_ops()) {
        for (auto&& output : node->outputs()) {
            const auto& tensor_names = output.get_tensor().get_names();
            if (tensor_names.size() > 32) {
                LOG_VERB(model->get_friendly_name() << " output " << output << " exceeds the name limit, removing...");
                output.get_tensor().set_names({});
            }
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
            LOG_VERB("Added input tensor name for " << model->get_friendly_name());
        }
        in_tensor_idx++;
    }
    for (auto& output : model->outputs()) {
        const auto& tensor_names = output.get_tensor().get_names();
        if (tensor_names.empty()) {
            output.get_tensor().set_names({"npuw_out_tensor_" + std::to_string(out_tensor_idx)});
            LOG_VERB("Added output tensor name for " << model->get_friendly_name());
        }
        out_tensor_idx++;
    }
}

void ov::npuw::CompiledModel::report_io() const {
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

bool ov::npuw::CompiledModel::compile_for_success(std::size_t id, const std::vector<std::string>& devices) {
    if (m_compiled_submodels[id].replaced_by && m_compiled_submodels[id].replaced_by != id) {
        LOG_BLOCK();
        LOG_INFO("Skip compilation for Subgraph[" << id
                                                  << "] "
                                                     "as it was already compiled (function)");
        return true;
    }

    auto& desc = m_compiled_submodels[id];
    if (devices.empty()) {
        return false;
    }

    // Apply NPU workarounds: pre-filter devices that would trigger unrecoverable failures.
    // These checks apply only to the main model compilation (not MoE/pyramid/HFA sub-models).
    std::vector<std::string> main_candidates;
    for (const auto& device : devices) {
        if (npuw::util::starts_with(device, "NPU")) {
            if (desc.model->inputs().empty()) {
                LOG_INFO("Avoid compilation for " << device << " as the model should be constant-folded");
                dump_on_fail(id, device, "Avoided due to workaround");
                continue;
            }
            if (ov::npuw::attn::get_compiled_dynamic(desc.pipeline.context) != nullptr) {
                bool has_dynamic = false;
                for (const auto& input : desc.model->inputs()) {
                    if (input.get_partial_shape().is_dynamic()) {
                        has_dynamic = true;
                        break;
                    }
                }
                if (has_dynamic) {
                    LOG_INFO("Avoid compilation for " << device << " as attention model has dynamic shapes");
                    dump_on_fail(id, device, "Avoided due to dynamic shapes");
                    continue;
                }
            }
        }
        main_candidates.push_back(device);
    }
    if (main_candidates.empty()) {
        return false;
    }

    // Factory builder: returns a failsafe::CompiledModel::Factory for the given model+suffix.
    // Captures the model as weak_ptr so that detach_memory() can truly release it when
    // no runtime fallback is expected. If the model has been cleared and failover fires
    // anyway, the lock() fails and the factory throws, which failsafe propagates correctly.
    auto make_factory = [this](const std::shared_ptr<ov::Model>& model,
                               const std::string& profile_suffix) -> ov::npuw::failsafe::CompiledModel::Factory {
        std::weak_ptr<ov::Model> weak_model = model;
        return [this, weak_model, profile_suffix](const std::string& device) -> ov::SoPtr<ov::ICompiledModel> {
            auto locked_model = weak_model.lock();
            OPENVINO_ASSERT(locked_model, "Failsafe factory: ov::Model was released before recompilation");
            ov::SoPtr<ov::ICompiledModel> compiled;
            // FIXME: Concurrent write access to the profile map, if compiled in parallel
            m_profile["compile/" + device + profile_suffix].record([&]() {
                compiled = compile_submodel(locked_model, device);
            });
            return compiled;
        };
    };

    auto make_failsafe = [&](const std::shared_ptr<ov::Model>& model,
                             const std::string& profile_suffix,
                             const std::vector<std::string>& devs) -> ov::SoPtr<ov::ICompiledModel> {
        if (!m_cfg.get<::intel_npu::NPUW_FALLBACK_EXEC>()) {
            std::exception_ptr last_failure;
            auto factory = make_factory(model, profile_suffix);
            for (const auto& device : devs) {
                try {
                    auto compiled = factory(device);
                    OPENVINO_ASSERT(compiled._ptr != nullptr,
                                    "Failsafe factory returned null compiled model for device ",
                                    device);
                    return compiled;
                } catch (...) {
                    last_failure = std::current_exception();
                }
            }
            if (last_failure) {
                std::rethrow_exception(last_failure);
            }
            OPENVINO_THROW("No candidate devices available for compilation");
        }

        return ov::npuw::failsafe::CompiledModel::create(model,
                                                         get_plugin(),
                                                         devs,
                                                         make_factory(model, profile_suffix));
    };

    auto make_wrapped = [&](const std::shared_ptr<ov::Model>& model,
                            const std::string& profile_suffix,
                            const std::vector<std::string>& devs) -> ov::SoPtr<ov::ICompiledModel> {
        auto main_cm = make_failsafe(model, profile_suffix, devs);
        if (m_acc_check) {
            const auto exec_devs = main_cm->get_property(ov::execution_devices.name()).as<std::vector<std::string>>();
            const std::string actual_device = exec_devs.empty() ? "" : exec_devs.front();
            if (actual_device != m_ref_device) {
                LOG_INFO("Wrapping with AccuracyChecked (main: " << actual_device << ", ref: " << m_ref_device << ").");
                auto ref_cm = compile_submodel(model, m_ref_device);
                return ov::npuw::accuracy_checked::CompiledModel::create(model,
                                                                         get_plugin(),
                                                                         std::move(main_cm),
                                                                         std::move(ref_cm),
                                                                         m_acc_check);
            }
        }
        return main_cm;
    };

    if (desc.pipeline.compile_executor) {
        ov::npuw::v1::subgraphs::CompileContext compile_context{desc.model, desc.compiled_model, devices, make_wrapped};
        desc.pipeline.compile_executor(compile_context);
    } else {
        desc.compiled_model = make_wrapped(desc.model, "", main_candidates);
    }

    if (auto* pyramid_attn = ov::npuw::attn::get_compiled_pyramid(desc.pipeline.context)) {
        LOG_INFO("Compiling pyramid attention submodels for Subgraph[" << id << "]...");
        LOG_BLOCK();

        const auto& pyramid_attn_models = pyramid_attn->_models_to_compile;
        const size_t total_models = pyramid_attn_models.size();
        const size_t models_to_compile = total_models > 0 ? total_models - 1 : 0;

        std::vector<ov::SoPtr<ov::ICompiledModel>> compiled_models(total_models);

        // Check if device supports strided I/O for non-last pyramid models.
        // The last model reuses the already-compiled main subgraph model (compiled without
        // enable_strides_for), so strided I/O only applies to the explicitly compiled models.
        bool support_strides_for = false;
        std::string npu_device_str;
        std::string saved_strides;
        for (const auto& device : devices) {
            if (ov::npuw::util::starts_with(device, "NPU") && models_to_compile > 0 && pyramid_attn->num_models() > 0) {
                const auto supported_properties =
                    get_npuw_plugin()->get_core()->get_property(device, ov::supported_properties);
                support_strides_for = std::find(supported_properties.begin(),
                                                supported_properties.end(),
                                                ov::intel_npu::enable_strides_for.name()) != supported_properties.end();
                if (support_strides_for) {
                    pyramid_attn->_can_use_tensor_view = true;
                    const auto& first_model = pyramid_attn_models[0];
                    npu_device_str = device;
                    const auto& strides_key = ov::intel_npu::enable_strides_for.name();
                    const ov::Any existing_any =
                        ov::npuw::util::at::_(m_meta_devices[npu_device_str]).at_or(strides_key, std::string{});
                    saved_strides = existing_any.as<std::string>();
                    std::string strided_inputs = saved_strides;
                    pyramid_attn->collect_strided_input_names(*first_model, strided_inputs);
                    m_meta_devices[npu_device_str][strides_key] = strided_inputs;
                    LOG_INFO("Enabled using tensor view for device: " << device
                                                                      << " for pyramid inputs: " << strided_inputs);
                }  // if(support_strides_for)
            }  // if(npu)
        }  // for(devices)

        auto compile_one = [&](size_t model_id) {
            compiled_models[model_id] =
                make_wrapped(pyramid_attn_models[model_id], "/pyramid_" + std::to_string(model_id), devices);
        };
        const bool par_opt = m_cfg.get<::intel_npu::NPUW_PARALLEL_COMPILE>();
        if (par_opt && models_to_compile > 0) {
            ov::parallel_for(models_to_compile, compile_one);
        } else {
            for (size_t model_id = 0; model_id < models_to_compile; ++model_id) {
                compile_one(model_id);
            }
        }

        if (total_models > 0) {
            OPENVINO_ASSERT(desc.compiled_model, "Original compiled model should exist");
            compiled_models[total_models - 1] = desc.compiled_model;
        }

        pyramid_attn->set_compiled_models(std::move(compiled_models));
        LOG_INFO("Pyramid attention compilation complete for Subgraph[" << id << "]");

        if (support_strides_for && !npu_device_str.empty()) {
            const auto& strides_key = ov::intel_npu::enable_strides_for.name();
            if (saved_strides.empty()) {
                m_meta_devices[npu_device_str].erase(strides_key);
            } else {
                m_meta_devices[npu_device_str][strides_key] = saved_strides;
            }
        }
    }  // if (pyramid_attn)

    if (auto* hfa = ov::npuw::attn::get_compiled_hfa(desc.pipeline.context)) {
        LOG_INFO("Compiling host flash attention tile models for Subgraph[" << id << "]...");
        LOG_BLOCK();
        
        std::string strided_inputs_intermediate{};
        std::string npu_stride_key{};

        if (!hfa->_tile_model_to_compile) {
            LOG_WARN("Host flash attention tile model is null, skipping compilation");
        } else {
            bool supports_strides_for = false;
            std::string npu_device_str;
            std::string saved_strides;
            for (const auto& device : devices) {
                if (!ov::npuw::util::starts_with(device, "NPU")) {
                    continue;
                }

                const auto supported_properties =
                    get_npuw_plugin()->get_core()->get_property(device, ov::supported_properties);
                const bool support_strides_for =
                    std::find(supported_properties.begin(),
                              supported_properties.end(),
                              ov::intel_npu::enable_strides_for.name()) != supported_properties.end();

                const bool m_use_npu_local_pipelining{m_cfg.get<::intel_npu::NPUW_CONTROLFLOW_EN>()};

                if (!support_strides_for || !m_use_npu_local_pipelining) {
                //if (!support_strides_for) {
                    break;
                }

                hfa->_can_use_tensor_view = true;
                npu_device_str = device;
                const auto& strides_key = ov::intel_npu::enable_strides_for.name();
                const ov::Any existing_any =
                    ov::npuw::util::at::_(m_meta_devices[device]).at_or(strides_key, std::string{});
                saved_strides = existing_any.as<std::string>();
                std::string strided_inputs = saved_strides;
                if (!strided_inputs.empty()) {
                    strided_inputs += ",";
                }

                

                if (!m_use_npu_local_pipelining) {                    
                    strided_inputs += std::string(hfa_tile_input_id_to_string(HFATileInputId::K_TILE)) + "," +
                                      std::string(hfa_tile_input_id_to_string(HFATileInputId::V_TILE));
                } else {
                    strided_inputs += std::string(hfa_tile_input_id_to_string(HFATileInputId::MASK_TILE));
                    strided_inputs_intermediate = strided_inputs;
                    strided_inputs_intermediate += ("," +std::string(hfa_tile_input_id_to_string(HFATileInputId::K_TILE)) +
                                                   "," +
                                                   std::string(hfa_tile_input_id_to_string(HFATileInputId::V_TILE)));
                }

                npu_stride_key = strides_key;
                m_meta_devices[device][strides_key] = strided_inputs;
                supports_strides_for = true;
                LOG_INFO("Enabled using tensor view for device: " << device << " for inputs: " << strided_inputs);
            }

            hfa->set_compiled_final_tile_model_strided(
                make_wrapped(hfa->_final_tile_model_to_compile, "/hfa_tile_strided", devices));                        

            if (npu_stride_key.length()) {
                for (const auto& device : devices) {
                    m_meta_devices[device][npu_stride_key] = strided_inputs_intermediate;
                }
            }            
            
            hfa->set_compiled_tile_model(make_wrapped(hfa->_tile_model_to_compile, "/hfa_tile", devices));

            hfa->set_compiled_final_tile_model(desc.compiled_model);
            LOG_INFO("Host flash attention compilation complete for Subgraph[" << id << "]");

            if (supports_strides_for && !npu_device_str.empty()) {
                const auto& strides_key = ov::intel_npu::enable_strides_for.name();
                if (saved_strides.empty()) {
                    m_meta_devices[npu_device_str].erase(strides_key);
                } else {
                    m_meta_devices[npu_device_str][strides_key] = saved_strides;
                }
            }
        }
    }  //  if (hfa)

    return true;
}

ov::SoPtr<ov::ICompiledModel> ov::npuw::CompiledModel::compile_submodel(const std::shared_ptr<ov::Model>& submodel,
                                                                        const std::string& device) {
    auto plugin = get_npuw_plugin();
    auto core = plugin->get_core();

    // set exclusive_async_requests in case when model is split
    // NOTE(dm): Not sure if it is required for the NPUW plugin, but likely it is
    // Make a device config COPY here!
    auto device_config = m_meta_devices[device];

    if (ov::npuw::util::starts_with(device, "NPU") && m_cfg.get<::intel_npu::NPUW_UNFOLD_IREQS>()) {
        device_config["NPU_RUN_INFERENCES_SEQUENTIALLY"] = "YES";
    }

    const auto& cache_dir = m_cfg.get<::intel_npu::NPUW_CACHE_DIR>();
    if (!cache_dir.empty()) {
        LOG_INFO("NPUW will try to utilize CACHE_DIR for " << submodel->get_friendly_name() << " submodel.");
        device_config.insert(ov::cache_dir(cache_dir));
    }

    if (m_compiled_submodels.size() > 1) {
        auto supported_internal_properties = core->get_property(device, ov::internal::supported_properties);
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
    const std::size_t end_idx = m_compiled_submodels.size();
    const std::size_t real_idx = m_compiled_submodels[id].replaced_by.value_or(id);

    if (ov::npuw::util::is_set(id, dof_opt, real_idx, end_idx)) {
        ov::npuw::dump_failure(m_compiled_submodels[id].model, device_to_try, extra);
    }
}

std::string ov::npuw::CompiledModel::format_subgraph_name(std::size_t id, const std::string& funcall) const {
    std::string name = m_name + "_" + ov::npuw::util::fmt(id, m_compiled_submodels.size());
    if (!funcall.empty()) {
        name += "_" + funcall;
    }
    return name;
}

void ov::npuw::CompiledModel::dump_subgraph_model(std::size_t id,
                                                  const std::string& funcall,
                                                  const std::string& dump_sub_opt) {
    const std::size_t end_sub_idx = m_compiled_submodels.size();
    const std::size_t real_id = m_compiled_submodels[id].replaced_by.value_or(id);

    if (!ov::npuw::util::is_set(id, dump_sub_opt, real_id, end_sub_idx)) {
        return;
    }

    LOG_INFO("Dumping Subgraph[" << id << "]");
    LOG_BLOCK();
    if (real_id != id) {
        LOG_INFO("NOTE: Dumping Subgraph[" << real_id << "]"
                                           << " as it is a function body for Subgraph[" << id << "]");
    }

    // const std::string dump_dir = m_cfg.get<::intel_npu::NPUW_DUMP_SUBS_DIR>();
    const std::string dump_dir = "C:\\Work\\gbaugh\\models\\dump\\ir\\";

    // Dump MoE expert models if present
    if (const auto* moe_experts = ov::npuw::moe::get_compiled_experts(m_compiled_submodels[id].pipeline.context)) {
        LOG_INFO("NOTE: Subgraph[" << id << "] has MoE experts mechanism.");
        const auto& moe_models = moe_experts->_models_to_compile;

        if (moe_models.empty()) {
            LOG_WARN("MoE experts models are empty (already compiled and cleared)");
        } else {
            for (const auto& entry : moe_models) {
                size_t chunk_size = entry.first;
                const auto& moe_model = entry.second;

                std::string moe_model_file_name =
                    format_subgraph_name(id, funcall) + "_moe_chunk_" + std::to_string(chunk_size) + ".xml";
                std::string moe_model_dump_path = ov::util::path_join({dump_dir, moe_model_file_name}).string();
                ov::save_model(moe_model, moe_model_dump_path);
                LOG_INFO("Wrote " << moe_model_dump_path);
            }
        }
        return;  // MoE experts don't have a single model to dump
    }

    // Dump MoE downstream model if present (the shape-reduced model passed to compile)
    if (const auto* moe_downstream =
            ov::npuw::moe::get_compiled_downstream(m_compiled_submodels[id].pipeline.context)) {
        LOG_INFO("NOTE: Subgraph[" << id << "] has MoE downstream mechanism.");
        if (moe_downstream->_model_to_compile) {
            std::string downstream_model_name = format_subgraph_name(id, funcall) + "_moe_downstream.xml";
            std::string downstream_model_dump_path = ov::util::path_join({dump_dir, downstream_model_name}).string();
            ov::save_model(moe_downstream->_model_to_compile, downstream_model_dump_path);
            LOG_INFO("Wrote " << downstream_model_dump_path);
        } else {
            LOG_WARN("MoE downstream model already compiled and cleared, cannot dump");
        }
        return;
    }

    const auto model_to_dump = m_compiled_submodels[real_id].model;
    if (!model_to_dump) {
        LOG_WARN("Model is null, cannot dump Subgraph[" << id << "]");
        return;
    }

    const std::string file_name = format_subgraph_name(id, funcall) + ".xml";
    const std::string model_dump_path = ov::util::path_join({dump_dir, file_name}).string();
    ov::save_model(model_to_dump, model_dump_path);
    LOG_INFO("Wrote " << model_dump_path);

    // Dump pyramid attention models if present
    if (const auto* pyramid_attn = ov::npuw::attn::get_compiled_pyramid(m_compiled_submodels[id].pipeline.context)) {
        LOG_INFO("NOTE: Subgraph[" << id << "] has a pyramid attention mechanism.");
        const auto& pyramid_attention_models = pyramid_attn->_models_to_compile;
        for (std::size_t idx = 0; idx < pyramid_attention_models.size(); ++idx) {
            std::string pyramid_attention_model_name = format_subgraph_name(id, funcall) + "_pyramid_" +
                                                       ov::npuw::util::fmt(idx, pyramid_attention_models.size()) +
                                                       ".xml";
            std::string pyramid_attention_model_dump_path =
                ov::util::path_join({dump_dir, pyramid_attention_model_name}).string();
            ov::save_model(pyramid_attention_models[idx], pyramid_attention_model_dump_path);
            LOG_INFO("Wrote " << pyramid_attention_model_dump_path);
        }
    }

    // Dump host flash attention models if present
    if (const auto* hfa = ov::npuw::attn::get_compiled_hfa(m_compiled_submodels[id].pipeline.context)) {
        LOG_INFO("NOTE: Subgraph[" << id << "] has a host flash attention mechanism.");
        const auto& hfa_tile_model = hfa->_tile_model_to_compile;
        std::string hfa_tile_model_name = format_subgraph_name(id, funcall) + "_hfa_tile.xml";
        std::string hfa_tile_model_dump_path = ov::util::path_join({dump_dir, hfa_tile_model_name}).string();
        ov::save_model(hfa_tile_model, hfa_tile_model_dump_path);
        LOG_INFO("Wrote " << hfa_tile_model_dump_path);

        const auto& hfa_final_tile_model = hfa->_final_tile_model_to_compile;
        std::string hfa_final_tile_model_name = format_subgraph_name(id, funcall) + "_hfa_final_tile.xml";
        std::string hfa_final_tile_model_dump_path =
            ov::util::path_join({dump_dir, hfa_final_tile_model_name}).string();
        ov::save_model(hfa_final_tile_model, hfa_final_tile_model_dump_path);
    }
}

void ov::npuw::CompiledModel::dump_subgraph_composition(const std::vector<ov::npuw::Subgraph>& orderedSubgraphs) const {
    LOG_INFO("Dumping subgraph composition for " << m_name << "...");
    LOG_BLOCK();

    const std::string dump_dir = m_cfg.get<::intel_npu::NPUW_DUMP_SUBS_DIR>();
    const std::string sg_file = "npuw_" + m_name + ".xml.sg";
    const std::string sg_path = ov::util::path_join({dump_dir, sg_file}).string();

    // Collect unique subgraphs via replaced_by()
    std::map<std::size_t, std::size_t> real_id_counts;
    for (size_t id = 0; id < orderedSubgraphs.size(); id++) {
        if (orderedSubgraphs[id]._optimized_out) {
            continue;
        }
        const std::size_t real_id = m_compiled_submodels[id].replaced_by.value_or(id);
        real_id_counts[real_id]++;
    }

    std::vector<std::pair<std::string, size_t>> subgraph_info;
    std::vector<std::string> base_subgraphs;
    std::vector<std::string> attn_subgraphs;
    std::vector<std::string> moe_subgraphs;

    for (const auto& [real_id, count] : real_id_counts) {
        const std::string& funcall = orderedSubgraphs[real_id]._funcall;
        std::string base_name = format_subgraph_name(real_id, funcall);
        subgraph_info.emplace_back(base_name, count);

        if (const auto* moe_experts =
                ov::npuw::moe::get_compiled_experts(m_compiled_submodels[real_id].pipeline.context)) {
            const auto& moe_models = moe_experts->_models_to_compile;
            if (!moe_models.empty()) {
                for (const auto& [chunk_size, model] : moe_models) {
                    moe_subgraphs.push_back(base_name + "_moe_chunk_" + std::to_string(chunk_size) + ".xml");
                }
            } else {
                base_subgraphs.push_back(base_name + ".xml");
            }
        } else if (ov::npuw::moe::get_compiled_downstream(m_compiled_submodels[real_id].pipeline.context) != nullptr) {
            base_subgraphs.push_back(base_name + ".xml");
            moe_subgraphs.push_back(base_name + "_moe_downstream.xml");
        } else {
            base_subgraphs.push_back(base_name + ".xml");

            if (const auto* pyramid_attn =
                    ov::npuw::attn::get_compiled_pyramid(m_compiled_submodels[real_id].pipeline.context)) {
                const auto& pyramid_models = pyramid_attn->_models_to_compile;
                for (std::size_t idx = 0; idx < pyramid_models.size(); ++idx) {
                    attn_subgraphs.push_back(base_name + "_pyramid_" + ov::npuw::util::fmt(idx, pyramid_models.size()) +
                                             ".xml");
                }
            }
            if (ov::npuw::attn::get_compiled_hfa(m_compiled_submodels[real_id].pipeline.context) != nullptr) {
                attn_subgraphs.push_back(base_name + "_hfa_tile.xml");
                attn_subgraphs.push_back(base_name + "_hfa_final_tile.xml");
            }
        }
    }

    std::ofstream json_file(sg_path);
    if (!json_file.is_open()) {
        const auto dir_path = ov::util::make_path(dump_dir);
        if (!ov::util::directory_exists(dir_path)) {
            LOG_ERROR("Failed to open file for writing: " << sg_path << ". Directory does not exist: " << dump_dir);
        } else {
            LOG_ERROR("Failed to open file for writing: " << sg_path << ". Check file permissions or disk space.");
        }
        return;
    }

    json_file << "{\n";
    json_file << "  \"pipeline_name\": \"npuw_" << m_name << "\",\n";
    json_file << "  \"repeated_numbers\": {\n";
    json_file << "    \"total_subgraphs\": " << subgraph_info.size();
    for (const auto& [name, count] : subgraph_info) {
        json_file << ",\n    \"" << name << "\": " << count;
    }
    json_file << "\n  },\n";
    json_file << "  \"subgraphs\": [\n";
    for (size_t i = 0; i < base_subgraphs.size(); i++) {
        json_file << "    \"" << base_subgraphs[i] << "\"";
        if (i < base_subgraphs.size() - 1) {
            json_file << ",";
        }
        json_file << "\n";
    }
    json_file << "  ]";

    // Add attention subgraphs if any exist
    if (!attn_subgraphs.empty()) {
        json_file << ",\n  \"attn_subgraphs\": [\n";
        for (size_t i = 0; i < attn_subgraphs.size(); i++) {
            json_file << "    \"" << attn_subgraphs[i] << "\"";
            if (i < attn_subgraphs.size() - 1) {
                json_file << ",";
            }
            json_file << "\n";
        }
        json_file << "  ]";
    }

    // Add MoE subgraphs if any exist
    if (!moe_subgraphs.empty()) {
        json_file << ",\n  \"moe_subgraphs\": [\n";
        for (size_t i = 0; i < moe_subgraphs.size(); i++) {
            json_file << "    \"" << moe_subgraphs[i] << "\"";
            if (i < moe_subgraphs.size() - 1) {
                json_file << ",";
            }
            json_file << "\n";
        }
        json_file << "  ]";
    }

    json_file << "\n}\n";

    json_file.close();
    LOG_INFO("Wrote " << sg_path);
}

std::shared_ptr<ov::npuw::IBaseInferRequest> ov::npuw::CompiledModel::create_base_infer_request() const {
    // Synchronous infer request implementation may vary based on the
    // selected strategy
    auto* non_const_this = const_cast<ov::npuw::CompiledModel*>(this);  // because of const in API
    auto non_const_this_sptr = std::static_pointer_cast<ov::npuw::CompiledModel>(non_const_this->shared_from_this());

    auto no_spatial_unpack = [&]() {
        const auto num_submodels = m_compiled_submodels.size();
        for (std::size_t idx = 0u; idx < num_submodels; idx++) {
            const auto& comp_model_desc = m_compiled_submodels[idx];
            if (!comp_model_desc.replaced_by.has_value() || comp_model_desc.forced_to_fcall) {
                // not a funcall, do nothing, or a subgraph that was forced to funcall
                // (a 1-call function) - skip
                continue;
            }
            const auto real_idx = comp_model_desc.replaced_by.value();
            if (m_compiled_submodels[real_idx].spatial) {
                LOG_WARN("Subgraph[" << idx << "] is a call to spatial function, unfold can't be done");
                return false;  // Spatial graph
            }
            if (unpack_required(idx)) {
                LOG_WARN("Subgraph[" << idx << "] requires unpack, unfold can't be done");
                return false;  // Unpack required
            }
        }

        return true;  // no spatial & subgraphs requiring unpack found
    };

    // UnfoldInferRequest caches direct references into compiled submodels; it is not safe
    // to use when runtime device failover is in play (failsafe wrapper may swap the inner
    // compiled model mid-inference).  Safe when either: only one device is configured,
    // or the user has explicitly disabled runtime fallback.
    auto no_failsafe_concern = [&]() {
        return m_dev_list.size() == 1 || !m_cfg.get<::intel_npu::NPUW_FALLBACK_EXEC>();
    };

    auto no_subgraph_behavior_concern = [&]() {
        for (std::size_t idx = 0u; idx < m_compiled_submodels.size(); idx++) {
            if (m_compiled_submodels[idx].pipeline.runtime_behavior.has_value()) {
                LOG_WARN("Subgraph[" << idx << "] has a runtime behavior, unfold can't be done");
                return false;
            }
        }
        return true;
    };

    std::shared_ptr<ov::npuw::IBaseInferRequest> result;
    if (m_compiled_pipeline_model) {
        result = std::make_shared<ov::npuw::PipelinedInferRequest>(non_const_this_sptr);
    } else if (m_cfg.get<::intel_npu::NPUW_UNFOLD_IREQS>() && no_spatial_unpack() && no_failsafe_concern() &&
        no_subgraph_behavior_concern()) {
        result = std::make_shared<ov::npuw::UnfoldInferRequest>(non_const_this_sptr);
    } else {
        result = std::make_shared<ov::npuw::JustInferRequest>(non_const_this_sptr);
    }
    NPUW_ASSERT(result);
    return result;
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::CompiledModel::create_sync_infer_request() const {
    return create_base_infer_request();
}

std::shared_ptr<ov::IAsyncInferRequest> ov::npuw::CompiledModel ::wrap_async_infer_request(
    std::shared_ptr<ov::npuw::IBaseInferRequest> internal_request) const {
    return std::make_shared<ov::IAsyncInferRequest>(internal_request, get_task_executor(), get_callback_executor());
}

std::shared_ptr<ov::IAsyncInferRequest> ov::npuw::CompiledModel::create_infer_request() const {
    return wrap_async_infer_request(create_base_infer_request());
}

void ov::npuw::CompiledModel::set_property(const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ov::Model> ov::npuw::CompiledModel::get_runtime_model() const {
    // NOTE(dm): See hetero plugin implementation if need to bring this method back
    // (that code should work as-is)
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ov::IPlugin> ov::npuw::CompiledModel::get_npuw_plugin() const {
    auto plugin = get_plugin();
    OPENVINO_ASSERT(plugin);
    return plugin;
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

std::string ov::npuw::CompiledModel::submodel_device(const std::size_t idx) const {
    if (!m_compiled_pipeline_model) {
        std::size_t real_idx = m_compiled_submodels[idx].replaced_by.value_or(idx);
        const auto& comp_subm_desc = m_compiled_submodels[real_idx];

        if (!comp_subm_desc.compiled_model) {
            return "";
        }

        const auto exec_devs =
            comp_subm_desc.compiled_model->get_property(ov::execution_devices.name()).as<std::vector<std::string>>();
        return exec_devs.empty() ? "" : exec_devs.front();
    } else {
        const auto exec_devs =
            m_compiled_pipeline_model->get_property(ov::execution_devices.name()).as<std::vector<std::string>>();
        return exec_devs.empty() ? "" : exec_devs.front();
    }
}

bool ov::npuw::CompiledModel::unpack_required(const std::size_t idx) const {
    auto& comp_model_desc = m_compiled_submodels.at(idx);
    for (std::size_t cidx = 0u; cidx < comp_model_desc.closure.get().closure.size(); cidx++) {
        if (unpack_required(idx, cidx)) {
            return true;
        }
    }
    return false;
}

bool ov::npuw::CompiledModel::unpack_required(const std::size_t idx, const std::size_t cidx) const {
    if (is_gather_closure(idx, cidx)) {
        return false;
    }

    auto& comp_model_desc = m_compiled_submodels.at(idx);
    const auto real_idx = comp_model_desc.replaced_by.value();
    auto& func_desc = m_compiled_submodels.at(real_idx);

    auto& closure = comp_model_desc.closure.get().closure.at(cidx);
    const auto closure_param_id = comp_model_desc.param_base + cidx;
    const auto port_type{func_desc.input_port_type[closure_param_id]};

    // auto& iport = func_desc.compiled_model->inputs()[closure_param_id];
    // return (closure.get_element_type() != iport.get_element_type());
    return (closure.get_element_type() != port_type);
}

bool ov::npuw::CompiledModel::is_gather_closure(const std::size_t idx, const std::size_t cidx) const {
    auto& comp_model_desc = m_compiled_submodels.at(idx);
    const auto real_idx = comp_model_desc.replaced_by.value();
    auto& func_desc = m_compiled_submodels.at(real_idx);

    const auto closure_param_id = comp_model_desc.param_base + cidx;

    if (func_desc.host_gather.dst_idx != -1 &&
        static_cast<uint64_t>(func_desc.host_gather.dst_idx) == closure_param_id) {
        return true;
    }
    return false;
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
    m_prop_to_opt = {{ov::supported_properties.name(),
                      {ov::PropertyMutability::RO,
                       [&](const ::intel_npu::Config&) -> std::vector<PropertyName>& {
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
                       [&](const ::intel_npu::Config&) -> std::string& {
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
                          BIND(npuw::partitioning::online::keep_blocks, NPUW_ONLINE_KEEP_BLOCKS),
                          BIND(npuw::partitioning::online::keep_block_size, NPUW_ONLINE_KEEP_BLOCK_SIZE),
                          BIND(npuw::partitioning::online::avoid, NPUW_ONLINE_AVOID),
                          BIND(npuw::partitioning::online::isolate, NPUW_ONLINE_ISOLATE),
                          BIND(npuw::partitioning::online::nofold, NPUW_ONLINE_NO_FOLD),
                          BIND(npuw::partitioning::online::dump_plan, NPUW_ONLINE_DUMP_PLAN),
                          BIND(npuw::partitioning::plan, NPUW_PLAN),
                          BIND(npuw::partitioning::fold, NPUW_FOLD),
                          BIND(npuw::partitioning::fold_only, NPUW_FOLD_ONLY),
                          BIND(npuw::partitioning::cwai, NPUW_CWAI),
                          BIND(npuw::partitioning::dyn_quant, NPUW_DQ),
                          BIND(npuw::partitioning::dyn_quant_full, NPUW_DQ_FULL),
                          BIND(npuw::partitioning::par_matmul_merge_dims, NPUW_PMM),
                          BIND(npuw::partitioning::matmul_gate_preserve_constants, NPUW_MM_GATED),
                          BIND(npuw::partitioning::slice_out, NPUW_SLICE_OUT),
                          BIND(npuw::partitioning::spatial, NPUW_SPATIAL),
                          BIND(npuw::partitioning::spatial_nway, NPUW_SPATIAL_NWAY),
                          BIND(npuw::partitioning::spatial_dyn, NPUW_SPATIAL_DYN),
                          BIND(npuw::partitioning::host_gather, NPUW_HOST_GATHER),
                          BIND(npuw::partitioning::funcall_for_all, NPUW_FUNCALL_FOR_ALL),
                          BIND(npuw::partitioning::f16_interconnect, NPUW_F16IC),
                          BIND(npuw::partitioning::dcoff_type, NPUW_DCOFF_TYPE),
                          BIND(npuw::partitioning::dcoff_with_scale, NPUW_DCOFF_SCALE),
                          BIND(npuw::partitioning::attn_hfa_fused, NPUW_ATTN_HFA_FUSED),
                          BIND(npuw::parallel_compilation, NPUW_PARALLEL_COMPILE),
                          BIND(npuw::ensure_compatibility, NPUW_ENSURE_COMPATIBILITY),
                          BIND(npuw::funcall_async, NPUW_FUNCALL_ASYNC),
                          BIND(npuw::controlflow_enabled, NPUW_CONTROLFLOW_EN),
                          BIND(npuw::unfold_ireqs, NPUW_UNFOLD_IREQS),
                          BIND(npuw::weights_bank, NPUW_WEIGHTS_BANK),
                          BIND(npuw::weights_bank_alloc, NPUW_WEIGHTS_BANK_ALLOC),
                          BIND(npuw::cache_dir, NPUW_CACHE_DIR),
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
