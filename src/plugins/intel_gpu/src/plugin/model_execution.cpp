// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <optional>
#include <set>
#include <sstream>
#include <utility>

#include "common_utils/parallel_mem_streambuf.hpp"
#include "common_utils/prefetch_streambuf.hpp"
#include "intel_gpu/graph/serialization/layout_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/utils.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/custom_layer.hpp"
#include "intel_gpu/plugin/plugin.hpp"
#include "intel_gpu/plugin/transformations_pipeline.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/internal_properties.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "openvino/core/deprecated.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/performance_heuristics.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"
#include "plugin/ops/moe_offload_constant.hpp"
#include "transformations/common_optimizations/dimension_tracking.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"
#include "transformations/utils/utils.hpp"

namespace {

using ms = std::chrono::duration<double, std::ratio<1, 1000>>;
using Time = std::chrono::high_resolution_clock;
thread_local const size_t* g_current_tensor_base = nullptr;

}  // namespace

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

#define FACTORY_DECLARATION(op_version, op_name) void __register##_##op_name##_##op_version();

#define FACTORY_CALL(op_version, op_name) __register##_##op_name##_##op_version();

#define REGISTER_FACTORY(op_version, op_name) FACTORY_DECLARATION(op_version, op_name)
#include "intel_gpu/plugin/primitives_list.hpp"
#undef REGISTER_FACTORY

void Plugin::register_primitives() {
#define REGISTER_FACTORY(op_version, op_name) FACTORY_CALL(op_version, op_name)
#include "intel_gpu/plugin/primitives_list.hpp"
#undef REGISTER_FACTORY
}

bool Plugin::supports_model_execution() {
    return true;
}

bool Plugin::supports_runtime_property(const std::string& /*name*/) {
    return true;
}

void Plugin::validate_custom_layer_config(const ov::AnyMap& user_config) {
    // Check that custom layers config can be loaded
    if (user_config.find(ov::intel_gpu::config_file.name()) != user_config.end()) {
        CustomLayerMap custom_layers;
        auto custom_layers_config = user_config.at(ov::intel_gpu::config_file.name()).as<std::string>();
        CustomLayer::LoadFromFile(custom_layers_config, custom_layers, custom_layers_config.empty());
    }
}

ov::Any Plugin::get_model_metric(const std::string& name, const ov::AnyMap& options, const cldnn::device_info& device_info) const {
    if (name == ov::range_for_async_infer_requests) {
        return decltype(ov::range_for_async_infer_requests)::value_type{1, 2, 1};
    }
    if (name == ov::optimal_batch_size) {
        return decltype(ov::optimal_batch_size)::value_type{get_optimal_batch_size(options)};
    }
    if (name == ov::max_batch_size) {
        return decltype(ov::max_batch_size)::value_type{get_max_batch_size(options)};
    }
    if (name == ov::compatibility_check) {
        if (auto it = options.find(ov::runtime_requirements.name()); it != options.end()) {
            const auto& requirements = it->second.as<std::string>();
            if (!requirements.empty()) {
                // Same compatibility policy as the import_model() guard (single source of truth).
                return CompiledModel::is_runtime_requirements_compatible(requirements, device_info) ? ov::CompatibilityCheck::SUPPORTED
                                                                                                    : ov::CompatibilityCheck::UNSUPPORTED;
            }
        }
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }
    return {};
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

    std::shared_ptr<GpuWeightlessCacheMap> cache_attr_map = std::make_shared<GpuWeightlessCacheMap>();

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
    OPENVINO_ASSERT(new_results.size() == old_results.size(),
                    "[GPU] Unexpected outputs count change in transformed model",
                    "Before: ",
                    old_results.size(),
                    " After: ",
                    new_results.size());
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
        validate_model_resident_constant_memory(*transformed_model, config.get_offload_ratio(), context->get_device().get_info().max_global_mem_size);
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

    auto transformed_model = clone_and_transform_model(model, config, context_impl);

    config.finalize(context_impl.get(), transformed_model.get());

    if (config.get_offload_ratio() > 0) {
        validate_model_resident_constant_memory(*transformed_model, config.get_offload_ratio(), context_impl->get_device().get_info().max_global_mem_size);
    }

    return std::make_shared<CompiledModel>(transformed_model, shared_from_this(), context_impl, config, weight_sharing_context);
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
    return import_model(model, {context, nullptr}, config);
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

    ov::CacheMode cache_mode = config.get_cache_mode();
    ov::EncryptionCallbacks encryption_callbacks = config.get_cache_encryption_callbacks();

    std::unique_ptr<cldnn::BinaryInputBuffer> ib_ptr =
        encryption_callbacks.decrypt ? std::make_unique<cldnn::EncryptedBinaryInputBuffer>(model, context_impl->get_engine(), encryption_callbacks.decrypt)
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

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model, const ov::AnyMap& config) const {
    std::string device_id = get_device_id(config);
    auto context = get_default_context(device_id);
    return import_model(model, {context, nullptr}, config);
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(const ov::Tensor& model,
                                                         const ov::SoPtr<ov::IRemoteContext>& context,
                                                         const ov::AnyMap& config) const {
    // Store tensor base in thread-local storage
    g_current_tensor_base = static_cast<const size_t*>(model.data());
    std::unique_ptr<std::streambuf> buf;
    const bool use_prefetch = ov::intel_gpu::PrefetchStreamBuf::is_supported(model);
    if (!use_prefetch) {
        // Memory without hints (e.g. user provided) or not a byte blob, keep the pread path for user file mappings.
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
    GPU_DEBUG_LOG << "[GPU_MAX_BATCH_SIZE] available memory is " << available_device_mem << " (occupied: " << occupied_device_mem << ")" << std::endl;

    int64_t max_batch_size = 1;

    if (options.find(ov::hint::model.name()) == options.end()) {
        GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] MODELS_PTR is not set: return 1" << std::endl;
        return static_cast<uint32_t>(max_batch_size);
    }

    const uint32_t default_streams_for_tput = 2;
    if (options.contains(ov::num_streams.name())) {
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

    size_t base_batch_size = 16;  // empirically decided for DG1

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
                        GPU_DEBUG_LOG << "[MAX_BATCH_SIZE] detected batched input " << input->get_friendly_name() << " with index " << symbol << "[" << s << "]"
                                      << std::endl;
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
        GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] Base batch size: " << base_batch_size << std::endl;
        GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] Const mem usage: " << device_memory_usage.first << std::endl;
        GPU_DEBUG_INFO << "[GPU_MAX_BATCH_SIZE] General mem usage: " << device_memory_usage.second << std::endl;
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

    auto closest_pow_of_2 = [](double x) {
        int lower_power = static_cast<int>(floor(std::log(x) / std::log(2)));
        double lower_value = pow(2, lower_power);      // Current power of 2
        double upper_value = pow(2, lower_power + 1);  // Next power of 2

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
                   << "gfx_version.major, " << device_info.gfx_ver.major << "gfx_version.minor " << std::to_string(device_info.gfx_ver.minor) << "Cache size "
                   << std::to_string(device_info.max_global_cache_size) << std::endl;

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
