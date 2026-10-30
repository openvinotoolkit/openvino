// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/plugin.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "common_utils/parallel_mem_streambuf.hpp"
#include "common_utils/prefetch_streambuf.hpp"
#include "intel_gpu/graph/serialization/layout_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/utils.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/transformations_pipeline.hpp"
#include "plugin/ops/moe_offload_constant.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/device_query.hpp"
#include "intel_gpu/runtime/dispatch_probe.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/profiling.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/core/weights_prefetch.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/performance_heuristics.hpp"
#include "openvino/runtime/plugin_config.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"
#include "transformations/common_optimizations/dimension_tracking.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/runtime/tp/collective_contract.hpp"

#ifdef OV_GPU_MLIR_BACKEND_LINKED
#    include "transformations/mlir/interface/gpu_runtime.hpp"
#endif

#ifdef ENABLE_TP
#    include <optional>

#    include "ze/tp/collective_backend.hpp"
#    include "ze/tp/rank_communicator.hpp"
#    include "ze/ze_engine.hpp"
#endif

// Undef DEVICE_TYPE macro which can be defined somewhere in windows headers as DWORD and conflict with our metric
#ifdef DEVICE_TYPE
#undef DEVICE_TYPE
#endif

using ms = std::chrono::duration<double, std::ratio<1, 1000>>;
using Time = std::chrono::high_resolution_clock;
thread_local const size_t* g_current_tensor_base = nullptr;

namespace ov::intel_gpu {

namespace {

ov::internal::WeightSharingCtxPtr extract_weight_sharing_context(ov::AnyMap& config) {
    const auto* const property_name = ov::internal::model_sharing_context.name();
    const auto property_it = config.find(property_name);
    if (property_it == config.end()) {
        return nullptr;
    }

    auto ctx = property_it->second.as<ov::internal::WeightSharingCtxPtr>();
    config.erase(property_it);
    return ctx;
}

}  // namespace

#define FACTORY_DECLARATION(op_version, op_name) \
    void __register ## _ ## op_name ## _ ## op_version();

#define FACTORY_CALL(op_version, op_name) \
    __register ## _ ## op_name ## _ ## op_version();

#define REGISTER_FACTORY(op_version, op_name) FACTORY_DECLARATION(op_version, op_name)
#include "intel_gpu/plugin/primitives_list.hpp"
#undef REGISTER_FACTORY

void Plugin::register_primitives() {
#define REGISTER_FACTORY(op_version, op_name) FACTORY_CALL(op_version, op_name)
#include "intel_gpu/plugin/primitives_list.hpp"
#undef REGISTER_FACTORY
}

std::string Plugin::get_device_id_from_config(const ov::AnyMap& config) {
    std::string id;
    if (config.find(ov::device::id.name()) != config.end()) {
        id = config.at(ov::device::id.name()).as<std::string>();
    }
    return id;
}

std::string Plugin::get_device_id(const ov::AnyMap& config) const {
    std::string id = m_default_device_id;
    if (config.find(ov::device::id.name()) != config.end()) {
        id = config.at(ov::device::id.name()).as<std::string>();
    }
    return id;
}

bool Plugin::is_weightless_cache_attributes_set(const std::shared_ptr<const ov::Model>& model) {
    const auto& type_info = ov::WeightlessCacheAttribute::get_type_info_static();

    for (const auto& node : model->get_ordered_ops()) {
        if (ov::op::util::is_constant(node)) {
            auto& rtInfo = node->get_rt_info();
            const auto& it = rtInfo.find(type_info);

            if (it != rtInfo.end()) {
                return true;
            }
        }
    }

    return false;
}

void Plugin::set_weightless_cache_attributes(const std::shared_ptr<const ov::Model>& model) {
    uint32_t offset = 0;
    const auto& type_info = ov::WeightlessCacheAttribute::get_type_info_static();

    for (const auto& node : model->get_ordered_ops()) {
        if (ov::op::util::is_constant(node)) {
            auto& rtInfo = node->get_rt_info();

            // Offset behaves as a unique key for each constant. Size = 1 is used as dummy.
            rtInfo[type_info] = ov::WeightlessCacheAttribute(1, offset++, node->get_element_type());
        }
    }
}

void Plugin::create_weightless_cache_attributes(const std::shared_ptr<const ov::Model>& model, ExecutionConfig& config) {
    uint32_t offset = 0;

    std::shared_ptr<GpuWeightlessCacheMap>cache_attr_map = std::make_shared<GpuWeightlessCacheMap>();

    for (const auto& node : model->get_ordered_ops()) {
        if (ov::op::util::is_constant(node)) {
            // Offset behaves as a unique key for each constant. Size = 1 is used as dummy.
            cache_attr_map->emplace(node->get_instance_id(), ov::WeightlessCacheAttribute(1, offset++, node->get_element_type()));
        }
    }

    config.set_property(ov::intel_gpu::weightless_attr(cache_attr_map));
}

void Plugin::transform_model(std::shared_ptr<ov::Model>& model, const ExecutionConfig& config, const std::shared_ptr<RemoteContextImpl>& context) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::transform_model");
    TransformationsPipeline transformations(config, context);

    auto start = Time::now();
    transformations.apply(model);
    GPU_DEBUG_LOG << "Transformations time: " << std::chrono::duration_cast<ms>(Time::now() - start).count() << " ms" << std::endl;
}

std::shared_ptr<ov::Model> Plugin::clone_and_transform_model(const std::shared_ptr<const ov::Model>& model,
                                                             ExecutionConfig& config,
                                                             const std::shared_ptr<RemoteContextImpl>& context) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::clone_and_transform_model");
    GPU_DEBUG_DEFINE_MEM_LOGGER("Plugin::clone_and_transform_model");

    auto cloned_model = model->clone();
    OPENVINO_ASSERT(cloned_model != nullptr, "[GPU] Failed to clone model!");

    // Here we create a copy of the config to finalize it and ensure that transformation pipe can use correct options values
    // This is manily needed to correctly update lower level properties when higher level option is set by user
    // For example, transformation use inference_precision hint which may be updated by execution_mode property.
    // Update itself will happen on finalization stage, so we must call it to have correct passes flow.
    // The reason why we can't do finalization once and then just run all graph transformations is that
    // part of the tranformations may actually impact some properties. For example, LSTMSequence op presense
    // impacts value of use_onednn property. But in order to understand if there's an op of this type we have to run
    // common optimizations which may do subgraph fusion to LSTMSequence op. So basically, final value of use_onednn
    // property can be computed for transformed model only.
    auto config_copy = config.clone();
    config_copy.finalize(context.get(), model.get());

    // Propagate auto-detected kv_cache_precision from the pre-transformation finalize
    // to the original config. This is needed because model transformations may convert
    // 4-bit weight constants (u4/i4) to higher precision, making the detection fail
    // during the second finalize on the transformed model.
    if (config.get_user_properties().find(ov::hint::kv_cache_precision.name()) == config.get_user_properties().end()) {
        auto detected_kv_prec = config_copy.get_kv_cache_precision();
        if (detected_kv_prec != ov::element::dynamic) {
            config.set_user_property({ov::hint::kv_cache_precision(detected_kv_prec)}, OptionVisibility::RELEASE);
        }
    }

    std::string dump_path = GPU_DEBUG_VALUE_OR(config_copy.get_dump_graphs_path(), "");
    GPU_DEBUG_IF(!dump_path.empty()) {
        auto path_base = ov::util::make_path(dump_path) / (cloned_model->get_name() + ".svg");
        ov::pass::VisualizeTree(path_base).run_on_model(cloned_model);
    }

    // Set weightless cache attribute only for non IR (e.g. onnxruntime) models
    // This is a temporary solution. A common way of handling weightless caching will be defined later.
    if (config_copy.get_enable_weightless()) {
        const auto weight_path = ov::util::make_path(config_copy.get_weights_path());

        if (weight_path.extension() != ".bin" && !is_weightless_cache_attributes_set(cloned_model)) {
            set_weightless_cache_attributes(cloned_model);
        }
    }
    transform_model(cloned_model, config_copy, context);

    // Transformations for some reason may drop output tensor names, so here we copy those from the original model
    auto new_results = cloned_model->get_results();
    auto old_results = model->get_results();
    OPENVINO_ASSERT(new_results.size() == old_results.size(), "[GPU] Unexpected outputs count change in transformed model",
                                                              "Before: ", old_results.size(), " After: ", new_results.size());
    for (size_t i = 0; i < model->get_results().size(); i++) {
        auto new_res = new_results[i];
        auto old_res = old_results[i];

        new_res->output(0).set_names(old_res->output(0).get_names());
        new_res->set_friendly_name(old_res->get_friendly_name());
    }

    GPU_DEBUG_IF(!dump_path.empty()) {
        auto path_base = std::filesystem::path(dump_path) / (cloned_model->get_name() + "_" + "transformed_func.svg");
        ov::pass::VisualizeTree(path_base).run_on_model(cloned_model);
    }
    return cloned_model;
}

// weak map to hold singleton default contexts
// Weak singleton map is used to share contexts between multiple plugin(or Core) instances.
// It is needed to ensure that context is released before plugin is unloaded.
// As the actual ownership is in plugin class, the ownership is released when plugin is destructed.
std::map<std::string, std::weak_ptr<RemoteContextImpl>> weak_singleton_default_contexts;
std::mutex singleton_default_contexts_mutex;

std::map<std::string, RemoteContextImpl::Ptr> Plugin::get_default_contexts() const {
    std::call_once(m_default_contexts_once, [this]() {
        std::lock_guard<std::mutex> lock(singleton_default_contexts_mutex);
        for (const auto& device : m_device_map) {
            const auto device_name = get_device_name() + "." + device.first;

            // If already initialized, use existing one
            if (weak_singleton_default_contexts.find(device.first) != weak_singleton_default_contexts.end()) {
                if (auto ctx = weak_singleton_default_contexts[device.first].lock()) {
                    m_default_contexts[device.first] = ctx;
                    continue;
                }
            }
            // If context is not created yet or expired, create new one
            const auto initialize = false;
            auto ctx = std::make_shared<RemoteContextImpl>(device_name, std::vector<cldnn::device::ptr>{device.second}, initialize);
            weak_singleton_default_contexts[device.first] = ctx;
            m_default_contexts[device.first] = ctx;
        }
    });
    return m_default_contexts;
}

Plugin::Plugin() {
#ifdef OV_GPU_MLIR_BACKEND_LINKED
    ov::intel_gpu::mlir::register_mlir_gpu_runtime();
#endif
    set_device_name("GPU");
    register_primitives();

    cldnn::device_query device_query;
    m_device_map = device_query.get_available_devices();

    // Set default configs for each device
    for (const auto& device : m_device_map) {
        m_configs_map.insert({device.first, ExecutionConfig(ov::device::id(device.first))});
    }

    // Set common info for compiled_model_runtime_properties
    const auto& ov_version = ov::get_openvino_version();
    m_compiled_model_runtime_properties["OV_VERSION"] = ov_version.buildNumber;
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model, const ov::AnyMap& orig_config) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::compile_model");
    std::string device_id = get_device_id(orig_config);
    auto context = get_default_context(device_id);
    GPU_DEBUG_SET_ACTIVE_LUID(context->get_device().get_info().luid);

    OPENVINO_ASSERT(m_configs_map.find(device_id) != m_configs_map.end(), "[GPU] compile_model: Couldn't find config for GPU with id ", device_id);

    ExecutionConfig config = m_configs_map.at(device_id);
    auto user_config = orig_config;
    auto weight_sharing_context = extract_weight_sharing_context(user_config);
    config.set_user_property(user_config, OptionVisibility::RELEASE);

    auto transformed_model = clone_and_transform_model(model, config, context);

    config.finalize(context.get(), transformed_model.get());

    if (config.get_offload_ratio() > 0) {
        validate_model_resident_constant_memory(*transformed_model,
                                                config.get_offload_ratio(),
                                                context->get_device().get_info().max_global_mem_size);
    }

    {
        OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::compile_model::CreateCompiledModel");
        return std::make_shared<CompiledModel>(transformed_model, shared_from_this(), context, config, weight_sharing_context);
    }
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& orig_config,
                                                          const ov::SoPtr<ov::IRemoteContext>& context) const {
    auto context_impl = get_context_impl(context);
    context_impl->initialize();

    auto device_id = ov::DeviceIDParser{context_impl->get_device_name()}.get_device_id();
    GPU_DEBUG_SET_ACTIVE_LUID(context_impl->get_device().get_info().luid);

    OPENVINO_ASSERT(m_configs_map.find(device_id) != m_configs_map.end(), "[GPU] compile_model: Couldn't find config for GPU with id ", device_id);

    ExecutionConfig config = m_configs_map.at(device_id);
    auto user_config = orig_config;
    auto weight_sharing_context = extract_weight_sharing_context(user_config);
    config.set_user_property(user_config, OptionVisibility::RELEASE);
#ifdef ENABLE_TP
    if (const auto& communicator = context_impl->get_tp_communicator()) {
        config.set_property(ov::intel_gpu::tp_collective_backend(communicator->backend()));
    }
#endif

    auto transformed_model = clone_and_transform_model(model, config, context_impl);

    config.finalize(context_impl.get(), transformed_model.get());

    if (config.get_offload_ratio() > 0) {
        validate_model_resident_constant_memory(*transformed_model,
                                                config.get_offload_ratio(),
                                                context_impl->get_device().get_info().max_global_mem_size);
    }


    return std::make_shared<CompiledModel>(transformed_model, shared_from_this(), context_impl, config, weight_sharing_context);
}

ov::SoPtr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap& remote_properties) const {
    if (remote_properties.empty()) {
        return get_default_context(m_default_device_id);
    }
#ifdef ENABLE_TP
    if (remote_properties.count(ov::tp::internal::device_group.name()) != 0) {
        return create_tp_context(remote_properties);
    }
#else
    OPENVINO_ASSERT(remote_properties.count(ov::tp::internal::device_group.name()) == 0,
                    "[GPU] tensor-parallel contexts need a GPU plugin built with ENABLE_TP on the Level Zero runtime");
#endif
    return std::make_shared<RemoteContextImpl>(get_default_contexts(), remote_properties);
}

#ifdef ENABLE_TP
namespace {

/// Removes `property` from `params` and returns its value, if it was there.
template <typename T, ov::PropertyMutability M>
std::optional<T> take(ov::AnyMap& params, const ov::Property<T, M>& property) {
    auto it = params.find(property.name());
    if (it == params.end()) {
        return std::nullopt;
    }
    auto value = it->second.template as<T>();
    params.erase(it);
    return value;
}

cldnn::ze::tp::CollectiveOptions collective_options(const ExecutionConfig& config) {
    cldnn::ze::tp::CollectiveOptions options;
    options.enable_halving = config.get_tp_enable_halving();
    options.halving_max_bytes = config.get_tp_halving_max_bytes();
    options.force_sync_collective = GPU_DEBUG_VALUE_OR(config.get_tp_force_sync_collective(), false);
    options.skip_collective = GPU_DEBUG_VALUE_OR(config.get_tp_skip_collective(), false);
    options.verbosity = GPU_DEBUG_VALUE_OR(config.get_tp_verbose(), ov::log::Level::NO);

    const auto profiling = GPU_DEBUG_VALUE_OR(config.get_tp_profiling(), TPProfilingMode::NONE);
    options.profiling_host = profiling == TPProfilingMode::HOST || profiling == TPProfilingMode::ALL;
    options.profiling_device = profiling == TPProfilingMode::DEVICE || profiling == TPProfilingMode::ALL;
    if (profiling != TPProfilingMode::NONE) {
        const auto period = static_cast<std::size_t>(GPU_DEBUG_VALUE_OR(config.get_tp_dump_period(), uint64_t{0}));
        options.dump_period = period != 0 ? period : 1;
    }
    return options;
}

}  // namespace

cldnn::ze::tp::DeviceGroup Plugin::describe_tp_device_group(const std::vector<std::string>& names,
                                                            std::vector<std::string>& ids) const {
    cldnn::ze::tp::DeviceGroup group;
    group.names = names;
    ids.clear();
    for (const auto& name : names) {
        ov::DeviceIDParser parser(name);
        if (parser.get_device_name() != get_device_name()) {
            group.problem = "device " + name + " of the group is not a " + get_device_name() + " device";
            return group;
        }
        auto id = parser.get_device_id().empty() ? m_default_device_id : parser.get_device_id();
        if (m_device_map.count(id) == 0) {
            group.problem = "device " + name + " of the group does not exist";
            return group;
        }
        if (std::find(ids.begin(), ids.end(), id) != ids.end()) {
            group.problem = "device " + name + " appears in the group more than once";
            return group;
        }
        ids.push_back(std::move(id));
    }
    for (size_t r = 0; r < ids.size(); ++r) {
        const auto& engine = get_default_context(ids[r])->get_engine();
        if (engine.runtime_type() != cldnn::runtime_types::ze) {
            group.problem = "device " + names[r] + " does not run on Level Zero";
            return group;
        }
        const auto& ze_engine = cldnn::downcast<const cldnn::ze::ze_engine>(engine);
        const auto driver = ze_engine.get_driver().handle();
        if (r != 0 && driver != group.driver) {
            group.problem = "all devices of a group must share a Level Zero driver, " + names[r] +
                            " belongs to a different one than " + names[0];
            return group;
        }
        group.driver = driver;
        group.devices.push_back(ze_engine.get_device().handle());
    }
    return group;
}

ov::Any Plugin::query_tp_collectives(const std::string& name, const ov::AnyMap& arguments) const {
    const auto group_it = arguments.find(ov::tp::internal::device_group.name());
    OPENVINO_ASSERT(group_it != arguments.end(),
                    "[GPU][TP] ", name, " needs the device group as argument ", ov::tp::internal::device_group.name());
    std::vector<std::string> ids;
    const auto group = describe_tp_device_group(group_it->second.as<std::vector<std::string>>(), ids);
    if (name == ov::tp::internal::collective_backends.name()) {
        return decltype(ov::tp::internal::collective_backends)::value_type{cldnn::ze::tp::supported_backends(group)};
    }
    const auto backend_it = arguments.find(ov::tp::internal::collective_backend.name());
    const auto backend =
        backend_it == arguments.end() ? std::string{ov::tp::auto_backend} : backend_it->second.as<std::string>();
    return decltype(ov::tp::internal::collective_support)::value_type{cldnn::ze::tp::backend_support(group, backend)};
}

std::shared_ptr<RemoteContextImpl> Plugin::create_tp_context(const ov::AnyMap& params) const {
    ov::AnyMap rest = params;
    const auto group_names = take(rest, ov::tp::internal::device_group).value();
    const auto groups = take(rest, ov::tp::internal::groups);
    OPENVINO_ASSERT(groups.has_value(), "[GPU][TP] ", ov::tp::internal::groups.name(), " is required with ",
                    ov::tp::internal::device_group.name());
    OPENVINO_ASSERT(groups->size() == 1 && groups->count(0) == 1,
                    "[GPU][TP] only collective group 0 is supported, got ", groups->size(), " group(s)");
    const auto& group = groups->at(0);
    OPENVINO_ASSERT(group.exchange != nullptr, "[GPU][TP] collective group 0 has no exchange endpoint");

    const auto channels = take(rest, ov::tp::internal::channels);
    OPENVINO_ASSERT(!channels.has_value() || *channels == 1,
                    "[GPU][TP] only one channel per TP context is supported, got ", channels.value_or(0));
    const auto comm_buffer_size = take(rest, ov::tp::internal::comm_buffer_size).value_or(0);
    const auto timeout_ms = take(rest, ov::tp::internal::communication_timeout_ms).value_or(5000);
    OPENVINO_ASSERT(timeout_ms >= 0, "[GPU][TP] ", ov::tp::internal::communication_timeout_ms.name(),
                    " must not be negative, got ", timeout_ms);
    const auto own_id = take(rest, ov::device::id).value_or(m_default_device_id);
    OPENVINO_ASSERT(m_configs_map.count(own_id) != 0, "[GPU][TP] unknown device id ", own_id);

    // What remains tunes the collectives; anything else is a mistake.
    for (const auto& entry : rest) {
        OPENVINO_ASSERT(entry.first.rfind("GPU_TP_", 0) == 0,
                        "[GPU][TP] unsupported parameter of a TP context: ", entry.first);
    }
    ExecutionConfig config = m_configs_map.at(own_id);
    try {
        config.set_user_property(rest, OptionVisibility::ANY);
    } catch (const ov::Exception& e) {
        OPENVINO_THROW("[GPU][TP] the options of the TP context cannot be applied; the GPU_TP_* debug options exist "
                       "only in builds with ENABLE_GPU_DEBUG_CAPS: ", e.what());
    }

    const auto world_size = group.exchange->world_size();
    const auto rank = group.exchange->rank();
    OPENVINO_ASSERT(group_names.size() == static_cast<size_t>(world_size),
                    "[GPU][TP] the device group lists ", group_names.size(), " devices for ", world_size, " ranks");
    OPENVINO_ASSERT(rank >= 0 && rank < world_size, "[GPU][TP] invalid rank ", rank, " of ", world_size);

    std::vector<std::string> ids;
    const auto device_group = describe_tp_device_group(group_names, ids);
    // Resolving the name is the binding support check: an unusable group or name fails here.
    const auto backend = cldnn::ze::tp::resolve_backend(device_group, group.backend);
    OPENVINO_ASSERT(ids[static_cast<size_t>(rank)] == own_id,
                    "[GPU][TP] rank ", rank, " runs on ", group_names[static_cast<size_t>(rank)],
                    " by the group, but its context is created on ", get_device_name(), ".", own_id);

    auto shared = cldnn::ze::tp::acquire_shared_context(device_group.driver, device_group.devices);

    // Applies the environment (OV_GPU_TP_*) on top of what was set.
    config.finalize(get_default_context(own_id).get(), nullptr);
    auto communicator = std::make_shared<cldnn::ze::tp::RankCommunicator>(group.exchange,
                                                                           backend,
                                                                           std::move(shared),
                                                                           std::chrono::milliseconds{timeout_ms},
                                                                           collective_options(config),
                                                                           comm_buffer_size);
    return std::make_shared<RemoteContextImpl>(get_default_contexts(), std::move(communicator));
}
#endif  // ENABLE_TP

std::shared_ptr<RemoteContextImpl> Plugin::get_default_context(const std::string& device_id, bool initialize) const {
    auto contexts = get_default_contexts();
    OPENVINO_ASSERT(contexts.count(device_id), "[GPU] Context was not initialized for ", device_id, " device");

    auto context = contexts.at(device_id);
    if (initialize) {
        context->initialize();
    }

    return context;
}

ov::SoPtr<ov::IRemoteContext> Plugin::get_default_context(const AnyMap& params) const {
    std::string device_id = m_default_device_id;

    if (params.find(ov::device::id.name()) != params.end()) {
        device_id = params.at(ov::device::id.name()).as<std::string>();
    }

    return get_default_context(device_id);
}

void Plugin::apply_device_id_map(const std::map<std::string, std::string>& id_map) {
    // Core owns the numbering when several libraries share a device name; adopt its ids for every
    // device we enumerated, so nothing downstream (contexts, configs, metrics) sees our own.
    std::map<std::string, cldnn::device::ptr> renamed_devices;
    std::map<std::string, ExecutionConfig> renamed_configs;
    for (const auto& [internal_id, device] : m_device_map) {
        const auto it = id_map.find(internal_id);
        OPENVINO_ASSERT(it != id_map.end(), "[GPU] Core did not assign an id for enumerated device \"", internal_id, "\", so it could not be addressed.");
        renamed_devices[it->second] = device;
        renamed_configs.emplace(it->second, ExecutionConfig(ov::device::id(it->second)));
    }
    m_device_map = std::move(renamed_devices);
    m_configs_map = std::move(renamed_configs);
    // The assigned ids need not contain "0": another member may hold the device Core numbered 0.
    m_default_device_id = m_device_map.empty() ? std::string("0") : m_device_map.begin()->first;
}

void Plugin::set_property(const ov::AnyMap &config) {
    auto update_config = [](ExecutionConfig& config, const ov::AnyMap& user_config) {
        config.set_user_property(user_config, OptionVisibility::RELEASE);
        // Check that custom layers config can be loaded
        if (user_config.find(ov::intel_gpu::config_file.name()) != user_config.end()) {
            CustomLayerMap custom_layers;
            auto custom_layers_config = user_config.at(ov::intel_gpu::config_file.name()).as<std::string>();
            CustomLayer::LoadFromFile(custom_layers_config, custom_layers, custom_layers_config.empty());
        }
    };

    auto rest = config;
    if (auto id_map = rest.extract(ov::internal::device_id_map.name()); !id_map.empty()) {
        apply_device_id_map(id_map.mapped().as<std::map<std::string, std::string>>());
        if (!rest.empty()) {
            set_property(rest);
        }
        return;
    }

    if (config.find(ov::internal::config_device_id.name()) != config.end()) {
        std::string device_id = config.at(ov::internal::config_device_id.name()).as<std::string>();
        auto config_for_device = config;
        config_for_device.erase(ov::internal::config_device_id.name());
        update_config(m_configs_map.at(device_id), config_for_device);
    } else {
        std::string device_id = get_device_id_from_config(config);
        if (!device_id.empty()) {
            m_default_device_id = device_id;
            update_config(m_configs_map.at(device_id), config);
        } else {
            for (auto& conf : m_configs_map) {
                update_config(conf.second, config);
            }
        }
    }
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model, const ov::AnyMap& orig_config) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::query_model");
    ov::SupportedOpsMap res;
    std::string device_id = get_device_id(orig_config);

    auto ctx = get_default_context(device_id);

    ExecutionConfig config = m_configs_map.at(device_id);
    config.set_user_property(orig_config, OptionVisibility::RELEASE);
    config.finalize(ctx.get(), model.get());

    ProgramBuilder prog(ctx->get_engine(), config);

    float query_model_ratio = config.get_query_model_ratio();

    auto supported = ov::get_supported_nodes(
        model,
        [&config, &ctx](std::shared_ptr<ov::Model>& model) {
            std::map<std::string, ov::PartialShape> shapes;
            std::map<std::string, std::pair<int64_t, int64_t>> batch_dim;
            transform_model(model, config, ctx);
        },
        [&prog](std::shared_ptr<ov::Node> node) {
            return prog.is_op_supported(node);
        },
        query_model_ratio);

    for (auto&& op_name : supported) {
        res.emplace(op_name, ctx->get_device_name());
    }

    return res;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model, const ov::AnyMap& config) const {
    std::string device_id = get_device_id(config);
    auto context = get_default_context(device_id);
    GPU_DEBUG_SET_ACTIVE_LUID(context->get_device().get_info().luid);
    return import_model(model, { context, nullptr }, config);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& orig_config) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::ImportNetwork");

    auto context_impl = get_context_impl(context);
    context_impl->initialize();

    auto device_id = ov::DeviceIDParser{context_impl->get_device_name()}.get_device_id();
    GPU_DEBUG_SET_ACTIVE_LUID(context_impl->get_device().get_info().luid);

    // check ov::loaded_from_cache property and erase it due to not needed any more.
    auto _orig_config = orig_config;
    const auto& it = _orig_config.find(ov::loaded_from_cache.name());
    bool loaded_from_cache = false;
    if (it != _orig_config.end()) {
        loaded_from_cache = it->second.as<bool>();
        _orig_config.erase(it);
    }

    if (auto blob_it = _orig_config.find(ov::hint::compiled_blob.name()); blob_it != _orig_config.end()) {
        _orig_config.erase(blob_it);
    }

    auto weight_sharing_context = extract_weight_sharing_context(_orig_config);

    ExecutionConfig config = m_configs_map.at(device_id);
    config.set_user_property(_orig_config, OptionVisibility::RELEASE);
    config.apply_weightless_options();
#ifdef ENABLE_TP
    if (const auto& communicator = context_impl->get_tp_communicator()) {
        config.set_property(ov::intel_gpu::tp_collective_backend(communicator->backend()));
    }
#endif

    ov::CacheMode cache_mode = config.get_cache_mode();
    ov::EncryptionCallbacks encryption_callbacks = config.get_cache_encryption_callbacks();

    std::unique_ptr<cldnn::BinaryInputBuffer> ib_ptr =
        encryption_callbacks.decrypt
            ? std::make_unique<cldnn::EncryptedBinaryInputBuffer>(model, context_impl->get_engine(), encryption_callbacks.decrypt)
            : std::make_unique<cldnn::BinaryInputBuffer>(model, context_impl->get_engine(), g_current_tensor_base);
    auto& ib = *ib_ptr;

    ov::CacheMode loaded_cache_mode = ov::CacheMode::OPTIMIZE_SPEED;
    ib >> cldnn::make_data(&loaded_cache_mode, sizeof(ov::CacheMode));

    if (loaded_cache_mode != cache_mode) {
        return nullptr;
    }

    if (config.get_enable_weightless()) {
        const auto& weights_path = ov::util::make_path(config.get_weights_path());

        if (weights_path.extension() != ".bin") {
            // This is non IR case, e.g. onnxruntime.
            // This may not be required. Constant nodes should have the information already.
            // This is a temporary solution. A more robust solution will be implemented in future.

            // If some app modifies ov::Model before compile_model(), and
            // the constants are changed, and such modification is not done before import_model(),
            // weightless caching will not produce correct result.
            if (const auto& orig_model = config.get_model(); orig_model != nullptr) {
                if (!is_weightless_cache_attributes_set(orig_model)) {
                    create_weightless_cache_attributes(orig_model, config);
                }
            } else {
                return nullptr;
            }
        }
    }

    return std::make_shared<CompiledModel>(ib, shared_from_this(), context_impl, config, loaded_from_cache, weight_sharing_context);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model,
                                                         const ov::AnyMap& config) const{
    std::string device_id = get_device_id(config);
    auto context = get_default_context(device_id);
    return import_model(model, { context, nullptr }, config);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& config) const{
    // Store tensor base in thread-local storage
    g_current_tensor_base = static_cast<const size_t*>(model.data());
    std::unique_ptr<std::streambuf> buf;
#ifdef _WIN32
    // TODO: CVS-198510 enable on Windows once prefetch shows a compile-time gain there.
    const bool use_prefetch = false;
#else
    const bool use_prefetch = static_cast<bool>(ov::get_memory_hints(model));
#endif
    if (!use_prefetch) {
        // Memory without hints (e.g. user provided), keep the pread path for user file mappings.
        buf = std::make_unique<ov::intel_gpu::ParallelMemStreamBuf>(model.data(), model.get_byte_size());
    } else {
        buf = std::make_unique<ov::intel_gpu::PrefetchStreamBuf>(model);
    }
    std::istream stream(buf.get());
    auto compiled_model = import_model(stream, context, config);

    // Clear thread-local storage
    g_current_tensor_base = nullptr;

    // Ensure the tensor stays alive as long as the compiled model
    auto* gpu_model = dynamic_cast<CompiledModel*>(compiled_model.get());
    if (gpu_model) {
        gpu_model->set_backing_tensor(std::make_shared<ov::Tensor>(model));
    }

    return compiled_model;
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& options) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::get_property");

    // The metrics below don't depend on the device ID, so we should handle those
    // earler than querying actual ID to avoid exceptions when no devices are found
    if (name == ov::supported_properties) {
        return decltype(ov::supported_properties)::value_type {get_supported_properties()};
    }
    if (ov::internal::supported_properties == name) {
        return decltype(ov::internal::supported_properties)::value_type{get_supported_internal_properties()};
    }
    if (name == ov::available_devices) {
        std::vector<std::string> available_devices = { };
        for (const auto& dev : m_device_map) {
            available_devices.push_back(dev.first);
        }
        return decltype(ov::available_devices)::value_type {available_devices};
    }
    if (name == ov::internal::caching_properties) {
        return decltype(ov::internal::caching_properties)::value_type(get_caching_properties());
    }
    if (name == ov::tp::internal::collective_backends.name() || name == ov::tp::internal::collective_support.name()) {
#ifdef ENABLE_TP
        return query_tp_collectives(name, options);
#else
        if (name == ov::tp::internal::collective_backends.name()) {
            return decltype(ov::tp::internal::collective_backends)::value_type{};
        }
        return decltype(ov::tp::internal::collective_support)::value_type{
            "the GPU plugin is built without tensor-parallel collectives (ENABLE_TP on the Level Zero runtime)"};
#endif
    }

    ov::AnyMap actual_runtime_info;
    auto prepare_actual_runtime_info = [&]() {
        // Items can be inserted here if needed.
        // Note: driver version used to be here, but it was moved to cache load logic. See
        // https://github.com/openvinotoolkit/openvino/pull/29195
    };
    // Below properties depend on the device ID.
    if (name == ov::internal::compiled_model_runtime_properties.name()) {
        prepare_actual_runtime_info();
        auto model_runtime_info = m_compiled_model_runtime_properties;
        // Set specified device info for compiled_model_runtime_properties
        model_runtime_info.insert(actual_runtime_info.begin(), actual_runtime_info.end());
        auto model_format = ov::Any(model_runtime_info);
        return decltype(ov::internal::compiled_model_runtime_properties)::value_type(
            std::move(model_format.as<std::string>()));
    }
    if (name == ov::internal::compiled_model_runtime_properties_supported.name()) {
        ov::Any res = true;
        prepare_actual_runtime_info();
        auto it = options.find(ov::internal::compiled_model_runtime_properties.name());
        if (it == options.end()) {
            res = false;
            return res;
        }
        ov::AnyMap input_map = it->second.as<ov::AnyMap>();
        // Check common info of compiled_model_runtime_properties
        for (const auto& item : m_compiled_model_runtime_properties) {
            auto it = input_map.find(item.first);
            if (it == input_map.end() || it->second.as<std::string>() != item.second.as<std::string>()) {
                res = false;
                return res;
            }
        }
        // Check specified device info of compiled_model_runtime_properties
        for (const auto& it : actual_runtime_info) {
            auto item = input_map.find(it.first);
            if (item == input_map.end() || item->second.as<std::string>() != it.second.as<std::string>()) {
                res = false;
                break;
            }
        }
        return res;
    }

    OPENVINO_ASSERT(!m_device_map.empty(), "[GPU] Can't get ", name, " property as no supported devices found or an error happened during devices query.\n"
                                           "[GPU] Please check OpenVINO documentation for GPU drivers setup guide.\n");

    if (is_metric(name)) {
        return get_metric(name, options);
    }

    std::string device_id = m_default_device_id;
    if (options.find(ov::device::id.name()) != options.end()) {
        device_id = options.find(ov::device::id.name())->second.as<std::string>();
    }
    OPENVINO_ASSERT(m_configs_map.find(device_id) != m_configs_map.end(), "[GPU] get_property: Couldn't find config for GPU with id ", device_id);

    const auto& c = m_configs_map.at(device_id);
    return c.get_property(name, OptionVisibility::RELEASE);
}

auto StringRightTrim = [](std::string string, std::string substring, bool case_sensitive = true) {
    auto ret_str = string;
    if (!case_sensitive) {
        std::transform(string.begin(), string.end(), string.begin(), ::tolower);
        std::transform(substring.begin(), substring.end(), substring.begin(), ::tolower);
    }
    auto erase_position = string.rfind(substring);
    if (erase_position != std::string::npos) {
        // if space exists before substring remove it also
        if (std::isspace(string.at(erase_position - 1))) {
            erase_position--;
        }
        return ret_str.substr(0, erase_position);
    }
    return ret_str;
};

bool Plugin::is_metric(const std::string& name) {
    auto all_properties = get_supported_properties();
    auto internal_properties = get_supported_internal_properties();
    auto caching_properties = get_caching_properties();
    all_properties.insert(all_properties.end(), internal_properties.begin(), internal_properties.end());
    all_properties.insert(all_properties.end(), caching_properties.begin(), caching_properties.end());
    auto it = std::find(all_properties.begin(), all_properties.end(), name);
    OPENVINO_ASSERT(it != all_properties.end(), "[GPU] Property ", name, " is not in a list of supported properties");

    return !it->is_mutable();
}

ov::Any Plugin::get_metric(const std::string& name, const ov::AnyMap& options) const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Plugin::get_metric");
    auto device_id = get_property(ov::device::id.name(), options).as<std::string>();

    auto iter = m_device_map.find(std::to_string(cldnn::device_query::device_id));
    if (iter == m_device_map.end()) {
        iter = m_device_map.find(device_id);
    }
    if (iter == m_device_map.end()) {
        iter = m_device_map.begin();
    }
    auto device = iter->second;
    auto device_info = device->get_info();

    if (name == ov::intel_gpu::device_total_mem_size) {
        return decltype(ov::intel_gpu::device_total_mem_size)::value_type {device_info.max_global_mem_size};
    }
    if (name == ov::intel_gpu::device_max_alloc_mem_size) {
        return decltype(ov::intel_gpu::device_max_alloc_mem_size)::value_type {device_info.max_alloc_mem_size};
    }
    if (name == ov::device::type) {
        auto dev_type = device_info.dev_type == cldnn::device_type::discrete_gpu ? ov::device::Type::DISCRETE : ov::device::Type::INTEGRATED;
        return decltype(ov::device::type)::value_type {dev_type};
    }
    if (name == ov::device::gops) {
        std::map<element::Type, float> gops;
        gops[element::i8] = device->get_gops(cldnn::data_types::i8);
        gops[element::u8] = device->get_gops(cldnn::data_types::u8);
        gops[element::f16] = device->get_gops(cldnn::data_types::f16);
        gops[element::f32] = device->get_gops(cldnn::data_types::f32);
        return decltype(ov::device::gops)::value_type {gops};
    }
    if (name == ov::intel_gpu::cacheline_size) {
        return static_cast<decltype(ov::intel_gpu::cacheline_size)::value_type>(device_info.cacheline_size.value_or(0));
    }
    if (name == ov::intel_gpu::execution_units_count) {
        return static_cast<decltype(ov::intel_gpu::execution_units_count)::value_type>(device_info.execution_units_count);
    }
    if (name == ov::intel_gpu::uarch_version) {
        std::stringstream s;
        if (device_info.gfx_ver.major == 0 && device_info.gfx_ver.minor == 0 && device_info.gfx_ver.revision == 0) {
            s << "unknown";
        } else {
            s << static_cast<int>(device_info.gfx_ver.major) << "."
              << static_cast<int>(device_info.gfx_ver.minor) << "."
              << static_cast<int>(device_info.gfx_ver.revision);
        }
        return decltype(ov::intel_gpu::uarch_version)::value_type {s.str()};
    }
    if (name == ov::optimal_batch_size) {
        return decltype(ov::optimal_batch_size)::value_type {get_optimal_batch_size(options)};
    }
    if (name == ov::device::uuid) {
        return decltype(ov::device::uuid)::value_type {device_info.uuid};
    }
    if (name == ov::device::luid) {
        return decltype(ov::device::luid)::value_type {device_info.luid};
    }
    if (name == ov::device::full_name) {
        auto deviceName = StringRightTrim(device_info.dev_name, "NEO", false);
        deviceName += std::string(" (") + (device_info.dev_type == cldnn::device_type::discrete_gpu ? "dGPU" : "iGPU") + ")";
        return decltype(ov::device::full_name)::value_type {deviceName};
    }
    if (name == ov::device::capabilities) {
        return decltype(ov::device::capabilities)::value_type {get_device_capabilities(device_info)};
    }
    if (name == ov::range_for_async_infer_requests) {
        std::tuple<unsigned int, unsigned int, unsigned int> range = std::make_tuple(1, 2, 1);
        return decltype(ov::range_for_async_infer_requests)::value_type {range};
    }
    if (name == ov::range_for_streams) {
        std::tuple<unsigned int, unsigned int> range = std::make_tuple(1, device_info.num_ccs == 1 ? 2 : device_info.num_ccs);
        return decltype(ov::range_for_streams)::value_type {range};
    }
    if (name == ov::intel_gpu::memory_statistics) {
        const auto& ctx = get_default_context(device_id, false);
        if (!ctx->is_initialized()) {
            decltype(ov::intel_gpu::memory_statistics)::value_type res;
            auto add_zero_mem_usage = [&res](auto alloc_type) {
                std::ostringstream oss;
                oss << alloc_type;
                res[oss.str()] = 0;
            };
            add_zero_mem_usage(cldnn::allocation_type::unknown);
            add_zero_mem_usage(cldnn::allocation_type::cl_mem);
            add_zero_mem_usage(cldnn::allocation_type::usm_host);
            add_zero_mem_usage(cldnn::allocation_type::usm_shared);
            add_zero_mem_usage(cldnn::allocation_type::usm_device);
            return res;
        }
        return decltype(ov::intel_gpu::memory_statistics)::value_type{ctx->get_engine().get_memory_statistics()};
    }
    if (name == ov::max_batch_size) {
        return decltype(ov::max_batch_size)::value_type {get_max_batch_size(options)};
    }
    if (name == ov::intel_gpu::runtime_type) {
        return decltype(ov::intel_gpu::runtime_type)::value_type {std::string(cldnn::get_runtime_cache_tag())};
    }
    if (name == ov::intel_gpu::driver_version) {
        return decltype(ov::intel_gpu::driver_version)::value_type {device_info.driver_version};
    }
    if (name == ov::intel_gpu::device_id) {
        std::stringstream s;
        s << "0x" << std::hex << device_info.device_id;
        return decltype(ov::intel_gpu::device_id)::value_type {s.str()};
    }
    if (name == ov::device::architecture) {
        std::stringstream s;
        s << "GPU: vendor=0x" << std::hex << device_info.vendor_id << std::dec << " arch=";
        if (device_info.gfx_ver.major == 0 && device_info.gfx_ver.minor == 0) {
            s << device_info.dev_name;
        } else {
            s << "v" << static_cast<int>(device_info.gfx_ver.major)
              << "." << static_cast<int>(device_info.gfx_ver.minor)
              << "." << static_cast<int>(device_info.gfx_ver.revision);
        }
        return decltype(ov::device::architecture)::value_type {s.str()};
    }
    if (name == ov::device::pci_info) {
        ov::device::PCIInfo info;
        info.domain = device_info.pci_info.pci_domain;
        info.bus = device_info.pci_info.pci_bus;
        info.device = device_info.pci_info.pci_device;
        info.function = device_info.pci_info.pci_function;
        return decltype(ov::device::pci_info)::value_type {info};
    }
    if (name == ov::internal::cache_header_alignment) {
        return decltype(ov::internal::cache_header_alignment)::value_type{
            static_cast<decltype(ov::internal::cache_header_alignment)::value_type>(ov::util::get_system_page_size())};
    }
    if (name == ov::compatibility_check) {
        if (auto it = options.find(ov::runtime_requirements.name()); it != options.end()) {
            const auto& requirements = it->second.as<std::string>();
            if (!requirements.empty()) {
                // Same compatibility policy as the import_model() guard (single source of truth).
                return CompiledModel::is_runtime_requirements_compatible(requirements, device_info)
                           ? ov::CompatibilityCheck::SUPPORTED
                           : ov::CompatibilityCheck::UNSUPPORTED;
            }
        }
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }
    OPENVINO_THROW("Unsupported metric key ", name);
}

std::vector<ov::PropertyName> Plugin::get_caching_properties() {
    static const std::vector<ov::PropertyName> caching_properties = {
        ov::PropertyName{ov::device::architecture.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::execution_units_count.name(), PropertyMutability::RO},
        ov::PropertyName{ov::hint::inference_precision.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::execution_mode.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::performance_mode.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::dynamic_quantization_group_size.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::activations_scale_factor.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::kv_cache_precision.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::runtime_type.name(), PropertyMutability::RO},
    };

    return caching_properties;
}

std::vector<ov::PropertyName> Plugin::get_supported_properties() {
    static const std::vector<ov::PropertyName> supported_properties = {
        // Metrics
        ov::PropertyName{ov::supported_properties.name(), PropertyMutability::RO},
        ov::PropertyName{ov::available_devices.name(), PropertyMutability::RO},
        ov::PropertyName{ov::range_for_async_infer_requests.name(), PropertyMutability::RO},
        ov::PropertyName{ov::range_for_streams.name(), PropertyMutability::RO},
        ov::PropertyName{ov::optimal_batch_size.name(), PropertyMutability::RO},
        ov::PropertyName{ov::max_batch_size.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::architecture.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::full_name.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::uuid.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::luid.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::type.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::gops.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::capabilities.name(), PropertyMutability::RO},
        ov::PropertyName{ov::device::pci_info.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::device_id.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::device_total_mem_size.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::device_max_alloc_mem_size.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::uarch_version.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::cacheline_size.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::execution_units_count.name(), PropertyMutability::RO},
        ov::PropertyName{ov::intel_gpu::memory_statistics.name(), PropertyMutability::RO},
        ov::PropertyName{ov::compatibility_check.name(), PropertyMutability::RO},

        // Configs
        ov::PropertyName{ov::enable_profiling.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::model_priority.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::host_task_priority.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::queue_priority.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::queue_throttle.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::enable_sdpa_optimization.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::enable_lora_operation.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::hint::enable_large_allocations.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::enable_loop_unrolling.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::disable_winograd_convolution.name(), PropertyMutability::RW},
        ov::PropertyName{ov::cache_dir.name(), PropertyMutability::RW},
        ov::PropertyName{ov::cache_mode.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::performance_mode.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::execution_mode.name(), PropertyMutability::RW},
        ov::PropertyName{ov::compilation_num_threads.name(), PropertyMutability::RW},
        ov::PropertyName{ov::num_streams.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::num_requests.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::inference_precision.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::enable_cpu_pinning.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::enable_cpu_reservation.name(), PropertyMutability::RW},
        ov::PropertyName{ov::device::id.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::dynamic_quantization_group_size.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::activations_scale_factor.name(), PropertyMutability::RW},
        ov::PropertyName{ov::weights_path.name(), PropertyMutability::RW},
        ov::PropertyName{ov::cache_encryption_callbacks.name(), PropertyMutability::WO},
        ov::PropertyName{ov::hint::kv_cache_precision.name(), PropertyMutability::RW},
        ov::PropertyName{ov::hint::model.name(), PropertyMutability::WO},
        ov::PropertyName{ov::intel_gpu::offload_ratio.name(), PropertyMutability::RW},
        ov::PropertyName{ov::intel_gpu::config_file.name(), PropertyMutability::RW},
    };

    return supported_properties;
}

std::vector<ov::PropertyName> Plugin::get_supported_internal_properties() {
    static const std::vector<ov::PropertyName> supported_internal_properties = {
        ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::internal::config_device_id.name(), ov::PropertyMutability::WO},
        ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW},
        ov::PropertyName{ov::internal::compiled_model_runtime_properties.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::internal::compiled_model_runtime_properties_supported.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::internal::query_model_ratio.name(), PropertyMutability::RW},
        ov::PropertyName{ov::internal::caching_with_mmap.name(), PropertyMutability::RO},
        ov::PropertyName{ov::internal::cache_header_alignment.name(), PropertyMutability::RO},
        ov::PropertyName{ov::internal::device_id_map.name(), ov::PropertyMutability::WO},
        ov::PropertyName{ov::tp::internal::collective_backends.name(), ov::PropertyMutability::RO},
        ov::PropertyName{ov::tp::internal::collective_support.name(), ov::PropertyMutability::RO}};
    return supported_internal_properties;
}

std::vector<std::string> Plugin::get_device_capabilities(const cldnn::device_info& info) {
    std::vector<std::string> capabilities;

    capabilities.emplace_back(ov::device::capability::FP32);
    capabilities.emplace_back(ov::device::capability::BIN);
    if (info.supports_fp16) {
        capabilities.emplace_back(ov::device::capability::FP16);
    }
    if (info.supports_bf16) {
        capabilities.emplace_back(ov::device::capability::BF16);
    }
    if (info.supports_imad || info.supports_immad) {
        capabilities.emplace_back(ov::device::capability::INT8);
    }
    if (info.supports_immad) {
        capabilities.emplace_back(ov::intel_gpu::capability::HW_MATMUL);
    }
    if (info.supports_usm) {
        capabilities.emplace_back(ov::intel_gpu::capability::USM_MEMORY);
    }
    capabilities.emplace_back(ov::device::capability::EXPORT_IMPORT);

    return capabilities;
}

uint32_t Plugin::get_max_batch_size(const ov::AnyMap& options) const {
    auto device_id = get_property(ov::device::id.name(), options).as<std::string>();
    auto context = get_default_contexts().at(device_id);
    GPU_DEBUG_SET_ACTIVE_LUID(context->get_device().get_info().luid);
    const auto& device_info = context->get_device().get_info();
    auto config = m_configs_map.at(device_id);
    config.set_property(ov::intel_gpu::partial_build_program(true));
    uint32_t n_streams = static_cast<uint32_t>(config.get_num_streams());
    uint64_t occupied_device_mem = 0;
    auto statistic_result = get_metric(ov::intel_gpu::memory_statistics.name(), options).as<std::map<std::string, uint64_t>>();
    std::ostringstream usm_device_oss;
    usm_device_oss << cldnn::allocation_type::usm_device;
    auto occupied_usm_dev = statistic_result.find(usm_device_oss.str());
    if (occupied_usm_dev != statistic_result.end()) {
        occupied_device_mem = occupied_usm_dev->second;
    }

    int64_t available_device_mem = device_info.max_global_mem_size - occupied_device_mem;
    GPU_DEBUG_LOG << "[GPU_MAX_BATCH_SIZE] available memory is " << available_device_mem
                  << " (occupied: " << occupied_device_mem << ")" << std::endl;

    int64_t max_batch_size = 1;

    if (options.find(ov::hint::model.name()) == options.end()) {
        GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] MODELS_PTR is not set: return 1" << std::endl;
        return static_cast<uint32_t>(max_batch_size);
    }

    const uint32_t default_streams_for_tput = 2;
    if (options.count(ov::num_streams.name()) > 0) {
        auto streams = options.at(ov::num_streams.name()).as<ov::streams::Num>();
        if (streams == ov::streams::AUTO) {
            n_streams = std::max(default_streams_for_tput, device_info.num_ccs);
        } else {
            n_streams = static_cast<uint32_t>(streams.num);
        }
    }

    GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] n_streams : " << n_streams << std::endl;

    auto available_device_mem_it = options.find(ov::intel_gpu::hint::available_device_mem.name());
    if (available_device_mem_it != options.end()) {
        if (available_device_mem_it->second.is<int64_t>()) {
            available_device_mem = std::min(available_device_mem, available_device_mem_it->second.as<int64_t>());
            GPU_DEBUG_LOG << "[GPU_MAX_BATCH_SIZE] available memory is reset by user " << available_device_mem << std::endl;
        } else {
            OPENVINO_THROW("[GPU_MAX_BATCH_SIZE] bad casting: ov::intel_gpu::hint::available_device_mem should be int64_t type");
        }
        if (available_device_mem < 0) {
            OPENVINO_THROW("[GPU_MAX_BATCH_SIZE] ov::intel_gpu::hint::available_device_mem value should be greater than 0 for max batch size calculation");
        }
    }

    std::shared_ptr<const ov::Model> model;
    auto model_param = options.find(ov::hint::model.name())->second;
    if (model_param.is<std::shared_ptr<const ov::Model>>()) {
        model = model_param.as<std::shared_ptr<const ov::Model>>();
    } else {
        OPENVINO_THROW("[GPU_MAX_BATCH_SIZE] ov::hint::model should be std::shared_ptr<ov::Model> type");
    }

    // Initialize the context before use
    context->initialize();

    config.finalize(context.get(), model.get());

    size_t base_batch_size = 16; // empirically decided for DG1

    auto& engine = context->get_engine();

    std::shared_ptr<ProgramBuilder> program;

    auto cloned_model = model->clone();

    try {
        std::set<std::pair<size_t, size_t>> batched_inputs;

        auto tmp_model = cloned_model->clone();
        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ov::pass::FindBatch>(true, false);
        m.run_passes(tmp_model);
        const auto& params = tmp_model->get_parameters();
        for (size_t input_id = 0; input_id < params.size(); input_id++) {
            const auto& input = params[input_id];
            const auto& shape = input->get_partial_shape();
            // currently no plugin support batched execution for dynamic networks
            if (shape.is_dynamic()) {
                GPU_DEBUG_LOG << "[MAX_BATCH_SIZE] does not support dynamic networks" << std::endl;
                return static_cast<uint32_t>(max_batch_size);
            }

            if (shape.size()) {
                for (size_t s = 0; s < shape.size(); s++) {
                    if (const auto& symbol = shape[s].get_symbol()) {
                        batched_inputs.insert(std::make_pair(input_id, s));
                        GPU_DEBUG_LOG << "[MAX_BATCH_SIZE] detected batched input " << input->get_friendly_name()
                                      << " with index " << symbol
                                      << "[" << s << "]" << std::endl;
                    }
                }
            }
        }

        if (batched_inputs.empty()) {
            GPU_DEBUG_LOG << "[MAX_BATCH_SIZE] MAX_BATCH_SIZE supports only networks with inputs/outputs featuring batched dim." << std::endl;
            return static_cast<uint32_t>(max_batch_size);
        }

        try {
            std::map<size_t, ov::PartialShape> shapes;
            const auto& params = cloned_model->get_parameters();
            for (size_t input_id = 0; input_id < params.size(); input_id++) {
                const auto& param = params[input_id];
                shapes[input_id] = param->get_output_partial_shape(0);
            }
            for (const auto& input : batched_inputs) {
                shapes[input.first][input.second] = base_batch_size;
            }
            cloned_model->reshape(shapes);
        } catch (...) {
            GPU_DEBUG_INFO << "[MAX_BATCH_SIZE] Error at reshape to " << base_batch_size << std::endl;
            return static_cast<uint32_t>(max_batch_size);
        }

        TransformationsPipeline transformations(config, context);
        transformations.apply(cloned_model);
        program = std::make_shared<ProgramBuilder>(cloned_model, engine, config);
        std::pair<int64_t, int64_t> device_memory_usage = program->get_compiled_program()->get_estimated_device_mem_usage();
        if (device_memory_usage.first == static_cast<int64_t>(-1L) && device_memory_usage.second == static_cast<int64_t>(-1L)) {
            return static_cast<uint32_t>(max_batch_size);
        }
        int64_t mem_for_general = std::max<int64_t>(1, available_device_mem - device_memory_usage.first);
        int64_t mem_per_batch = std::max<int64_t>(1, device_memory_usage.second / static_cast<int64_t>(base_batch_size));
        max_batch_size = mem_for_general / (mem_per_batch * static_cast<int64_t>(n_streams));
        GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] Base batch size: " << base_batch_size  << std::endl;
        GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] Const mem usage: " << device_memory_usage.first  << std::endl;
        GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] General mem usage: " << device_memory_usage.second  << std::endl;
    } catch (std::exception& e) {
        GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] Failed in reshape or build program " << e.what() << std::endl;
    }

    return static_cast<uint32_t>(max_batch_size);
}

uint32_t Plugin::get_optimal_batch_size(const ov::AnyMap& options) const {
    auto device_id = get_property(ov::device::id.name(), options).as<std::string>();
    auto context = get_default_contexts().at(device_id);
    GPU_DEBUG_SET_ACTIVE_LUID(context->get_device().get_info().luid);
    const auto& device_info = context->get_device().get_info();

    auto closest_pow_of_2 = [] (double x) {
        int lower_power = static_cast<int>(floor(std::log(x) / std::log(2)));
        double lower_value = pow(2, lower_power);        // Current power of 2
        double upper_value = pow(2, lower_power + 1);   // Next power of 2

        // Determine the threshold (70% of the range between lower and upper values)
        // If x is within the upper 30% of the range, return the upper power of 2.
        double threshold = 0.7 * (upper_value - lower_value);

        // Compare x with the threshold and return the appropriate power of 2
        if (x - lower_value > threshold) {
            return upper_value;  // Return the next power of 2
        }
        return lower_value;  // Return the current power of 2

    };

    auto model_param = options.find(ov::hint::model.name());
    if (model_param == options.end()) {
        GPU_DEBUG_INFO << "[OPTIMAL_BATCH_SIZE] ov::hint::model is not set: return 1" << std::endl;
        return static_cast<uint32_t>(1);
    }
    std::shared_ptr<const ov::Model> model;
    try {
        model = model_param->second.as<std::shared_ptr<const ov::Model>>();
    } catch (...) {
        OPENVINO_THROW("[OPTIMAL_BATCH_SIZE] ov::hint::model should be std::shared_ptr<const ov::Model> type");
    }
    GPU_DEBUG_INFO << "DEVICE_INFO:"
                   << "gfx_version.major, " << device_info.gfx_ver.major
                   << "gfx_version.minor " << std::to_string(device_info.gfx_ver.minor)
                   << "Cache size " << std::to_string(device_info.max_global_cache_size) << std::endl;

    // Initialize the context before use
    context->initialize();

    float L3_cache_size = static_cast<float>(device_info.max_global_cache_size);
    auto config = m_configs_map.at(device_id);
    auto cloned_model = clone_and_transform_model(model, config, context);
    ov::MemBandwidthPressure memPressure = ov::mem_bandwidth_pressure_tolerance(cloned_model, L3_cache_size);
    uint32_t batch = 1;
    if (memPressure.max_mem_tolerance != ov::MemBandwidthPressure::UNKNOWN) {
        batch = static_cast<uint32_t>(std::max(1.0, 16 * closest_pow_of_2(memPressure.max_mem_tolerance)));
    }
    ov::AnyMap options_for_max_batch;
    options_for_max_batch[ov::hint::model.name()] = model;
    options_for_max_batch[ov::num_streams.name()] = ov::streams::AUTO;
    auto max_batch_size = get_metric(ov::max_batch_size.name(), options_for_max_batch).as<uint32_t>();
    uint32_t closest = static_cast<uint32_t>(closest_pow_of_2(max_batch_size));
    batch = std::min(closest, batch);
    batch = std::min(256U, batch);  // batch 256 is a max
    GPU_DEBUG_INFO << memPressure.max_mem_tolerance << std::endl;
    GPU_DEBUG_INFO << "MAX_BATCH: " << max_batch_size << std::endl;
    GPU_DEBUG_INFO << "ACTUAL OPTIMAL BATCH: " << batch << std::endl;

    return batch;
}

}  // namespace ov::intel_gpu

namespace {

void enumerate_dispatch_devices(std::vector<ov::EnumeratedDevice>& out) {
    out.clear();  // The probe reports, it does not append: repeated calls must be idempotent.
    try {
        // Read once per process: the debug override cannot change while the library is loaded.
        static const std::string forced = ov::util::getenv_string("OV_GPU_RUNTIME");

        for (const auto& d : cldnn::lightweight_enumerate()) {
            ov::EnumeratedDevice e;
            e.internal_id = d.map_id;
            e.fingerprint = cldnn::make_fingerprint(d.info);
            e.score = cldnn::probe_score(cldnn::get_default_runtime_type(), d.info, forced);
            out.push_back(std::move(e));
        }
    } catch (...) {
        out.clear();  // Any failure -> "I serve nothing"; the probe never throws.
    }
}

}  // namespace

static const ov::Version version = { CI_BUILD_NUMBER, "Intel GPU plugin" };
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::intel_gpu::Plugin, version)

// Real implementation of the plugin-ABI enumeration probe (the GPU plugin participates
// in device-name dispatch); mirrors the stub every other plugin exports.
OV_DEFINE_PLUGIN_ENUMERATE_FUNCTION(enumerate_dispatch_devices)
