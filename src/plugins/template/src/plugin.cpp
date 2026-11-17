// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <memory>

#include "attach_cache_manager_to_paged_attention.hpp"
#include "hsm_common.hpp"
#include "hsm_reader_handler.hpp"
#include "itt.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/hsm_reader.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "remote_context.hpp"
#include "template/properties.hpp"
#include "transformations/common_optimizations/common_optimizations.hpp"
#include "transformations/control_flow/unroll_if.hpp"
#include "transformations/fp16_compression/convert_compression_only_to_legacy.hpp"
#include "transformations/fp16_compression/mark_decompression_convert_constant_folding.hpp"
#include "transformations/op_conversions/convert_avgpool_downgrade.hpp"
#include "transformations/op_conversions/convert_maxpool_downgrade.hpp"
#include "transformations/op_conversions/convert_reduce_to_pooling.hpp"
#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"
#include "transformations/paged_attention/convert_pagedattn_inputs.hpp"

namespace {
static constexpr const char* wait_executor_name = "TemplateWaitExecutor";
static constexpr const char* stream_executor_name = "TemplateStreamsExecutor";
static constexpr const char* template_exclusive_executor = "TemplateExecutor";

using ov::template_plugin::Plugin, ov::template_plugin::Configuration, ov::template_plugin::HsmReaderHandler;

// Properties that affect the compiled model; they form the cache ID and are persisted in the HSM blob.
std::vector<ov::PropertyName> caching_properties() {
    return {ov::device::architecture,
            ov::hint::performance_mode,
            ov::hint::execution_mode,
            ov::hint::inference_precision};
}

uint64_t get_blob_data_size(std::istream& model) {
    uint64_t size = 0;
    model.read(reinterpret_cast<char*>(&size), sizeof(size));
    return size;
}

std::string get_model_str(std::istream& model) {
    const auto model_size = get_blob_data_size(model);
    std::string xml;
    xml.resize(model_size);
    model.read(xml.data(), model_size);
    return xml;
}

ov::Tensor read_weights(std::istream& model, const size_t weights_size) {
    ov::Tensor weights(ov::element::from<uint8_t>(), ov::Shape{weights_size});
    model.read(reinterpret_cast<char*>(weights.data()), weights_size);
    return weights;
}

ov::Tensor get_model_weights(std::istream& model) {
    const auto weights_size = get_blob_data_size(model);
    return weights_size != 0 ? read_weights(model, weights_size) : ov::Tensor();
}

ov::Tensor get_model_weights(const ov::AnyMap& properties) {
    if (auto weights_path = properties.find(ov::weights_path.name()); weights_path != properties.end()) {
        const auto w_path = ov::util::make_path(weights_path->second.as<std::string>());
        return ov::read_tensor_data(w_path);
    } else {
        return {};
    }
}

// Removes import-only hints from properties; returns the ov::loaded_from_cache value.
bool consume_cache_import_hints(ov::AnyMap& properties) {
    bool loaded_from_cache = false;
    if (auto node = properties.extract(ov::loaded_from_cache.name())) {
        loaded_from_cache = node.mapped().as<bool>();
    }
    properties.erase(ov::hint::compiled_blob.name());
    return loaded_from_cache;
}

std::shared_ptr<ov::Model> get_ov_model_from_blob(const ov::template_plugin::Plugin& plugin,
                                                  const ov::Tensor& weights,
                                                  size_t offset,
                                                  const ov::AnyMap& properties) {
    if (auto blob_it = properties.find(ov::hint::compiled_blob.name()); blob_it != properties.end()) {
        if (const auto& blob = blob_it->second.as<ov::Tensor>(); blob) {
            ov::SharedStreamBuffer shared_buffer(blob.data(), blob.get_byte_size());
            std::istream blob_stream(&shared_buffer);
            blob_stream.seekg(offset, std::ios::beg);
            const auto model = get_model_str(blob_stream);
            try {
                return weights ? plugin.get_core()->read_model(model, weights) : nullptr;
            } catch (...) {
            }
        }
    }
    return nullptr;
};

// ! [plugin:read_hsm_blob]
std::shared_ptr<ov::AlignedBuffer> to_aligned_buffer(ov::Tensor tensor) {
    if (!tensor) {
        return nullptr;
    }
    // const data(): tensor may be read-only (e.g. mmap'd weights) - never written through, only wrapped.
    const auto* data = reinterpret_cast<const std::byte*>(static_cast<const ov::Tensor&>(tensor).data());
    return std::make_shared<ov::SharedBuffer<ov::Tensor>>(const_cast<std::byte*>(data), tensor.get_byte_size(), tensor);
}

std::pair<std::shared_ptr<ov::Model>, Configuration> read_hsm_blob(
    const std::shared_ptr<ov::ICore>& core,
    std::string_view runtime_requirements,
    const Configuration& default_cfg,
    const ov::runtime::hsm::Reader& reader,
    const ov::AnyMap& properties,
    ov::Tensor properties_weights_hint,
    std::shared_ptr<ov::AlignedBuffer> source_blob = {}) {
    HsmReaderHandler handler;
    handler.m_source_blob = source_blob;
    reader.read_sections({&handler});
    handler.check_mandatory_sections();

    // Opaque and optional, like NPU's own ov::runtime_requirements: absent means "nothing to check", not
    // an import failure - only a present-but-mismatched descriptor rejects the blob.
    if (const auto& requirements = handler.m_runtime_requirements) {
        OPENVINO_ASSERT(requirements->find(runtime_requirements) == 0,
                        "TEMPLATE: unsupported blob runtime requirements: ",
                        *requirements);
    }

    auto config = Configuration{properties, Configuration{handler.m_compiled_options, default_cfg, false}};

    const bool weightless = handler.m_blob_info->flags & ov::template_plugin::hsm::BlobInfo::weightless;
    auto weights = handler.m_weights;
    if (!weights) {
        weights = to_aligned_buffer(properties_weights_hint);
    }
    // Last resort: the path the model was originally compiled with, restored from the blob's own compiled
    // options - properties_weights_hint only covers what the caller already knew from @p properties.
    if (!weights && !config.weights_path.empty() && ov::util::file_size(config.weights_path) > 0) {
        weights = to_aligned_buffer(ov::read_tensor_data(config.weights_path));
    }
    OPENVINO_ASSERT(weightless || weights,
                    "TEMPLATE: missing mandatory weights section (model is not weightless and no external "
                    "weights were provided)");

    return {core->read_model(handler.m_model, weights), std::move(config)};
}
// ! [plugin:read_hsm_blob]
}  // namespace

// ! [plugin:ctor]
ov::template_plugin::Plugin::Plugin() {
    // TODO: fill with actual device name, backend engine
    set_device_name("TEMPLATE");

    // create backend which performs inference using openvino reference implementations
    m_backend = ov::runtime::Backend::create();

    // create default stream executor with a given name
    m_waitExecutor = get_executor_manager()->get_idle_cpu_streams_executor({wait_executor_name});
}
// ! [plugin:ctor]

// ! [plugin:dtor]
ov::template_plugin::Plugin::~Plugin() {
    // Plugin should remove executors from executor cache to avoid threads number growth in the whole application
    get_executor_manager()->clear(stream_executor_name);
    get_executor_manager()->clear(wait_executor_name);
}
// ! [plugin:dtor]

// ! [plugin:create_context]
ov::SoPtr<ov::IRemoteContext> ov::template_plugin::Plugin::create_context(const ov::AnyMap& remote_properties) const {
    return std::make_shared<ov::template_plugin::RemoteContext>();
}
// ! [plugin:create_context]

// ! [plugin:get_default_context]
ov::SoPtr<ov::IRemoteContext> ov::template_plugin::Plugin::get_default_context(
    const ov::AnyMap& remote_properties) const {
    return std::make_shared<ov::template_plugin::RemoteContext>();
}
// ! [plugin:get_default_context]

// ! [plugin:transform_model]
void transform_model(const std::shared_ptr<ov::Model>& model) {
    // Perform common optimizations and device-specific transformations
    ov::pass::Manager passManager("Plugin:Template");

    // Defaults are overwritten below from the actual PA node's input types (inputs 3/4 for
    // cache, input 0 for query). f32 is only used if there is no PA node in the model.
    ov::pass::ConvertPagedAttnInputs::KVCacheConfig cacheConfig;
    cacheConfig.keyCachePrecision = ov::element::f32;
    cacheConfig.valueCachePrecision = ov::element::f32;
    cacheConfig.inferencePrecision = ov::element::f32;
    for (const auto& op : model->get_ops()) {
        if (auto pa = std::dynamic_pointer_cast<ov::op::PagedAttentionExtension>(op)) {
            const auto key_et = pa->get_input_element_type(3);
            const auto val_et = pa->get_input_element_type(4);
            if (!key_et.is_dynamic())
                cacheConfig.keyCachePrecision = key_et;
            if (!val_et.is_dynamic())
                cacheConfig.valueCachePrecision = val_et;
            const auto qry_et = pa->get_input_element_type(0);
            if (!qry_et.is_dynamic())
                cacheConfig.inferencePrecision = qry_et;
            break;
        }
    }
    auto noop_shape = [](const ov::element::Type&, const bool, const size_t, int64_t&, int64_t&) {};
    passManager.register_pass<ov::pass::ConvertPagedAttnInputs>(cacheConfig, noop_shape);

    // Example: register CommonOptimizations transformation from transformations library
    passManager.register_pass<ov::pass::CommonOptimizations>();
    // Disable some transformations
    passManager.get_pass_config()->disable<ov::pass::UnrollIf>();
    passManager.get_pass_config()->disable<ov::pass::ConvertMaxPool14ToMaxPool8>();
    passManager.get_pass_config()->disable<ov::pass::ConvertAvgPool14ToAvgPool1>();
    // This transformation changes output name
    passManager.get_pass_config()->disable<ov::pass::ConvertReduceSumToPooling>();
    // Register any other transformations
    // ..

    const auto& pass_config = passManager.get_pass_config();

    // Allow FP16 Converts to be folded and FP16 constants to be upgraded to FP32 data type
    pass_config->disable<ov::pass::DisableDecompressionConvertConstantFolding>();
    pass_config->disable<ov::pass::ConvertCompressedOnlyToLegacy>();

    // Disabled SDPA transformation, since there is ref SDPA op.
    pass_config->disable<ov::pass::ScaledDotProductAttentionDecomposition>();

    // Attach cache manager to PA nodes for KV cache block management
    passManager.register_pass<ov::pass::AttachCacheManagerToPagedAttention>();

    // After `run_passes`, we have the transformed function, where operations match device operations,
    // and we can create device backend-dependent graph
    passManager.run_passes(model);
}
// ! [plugin:transform_model]

// ! [plugin:compile_model]
std::shared_ptr<ov::ICompiledModel> ov::template_plugin::Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties) const {
    return compile_model(model, properties, {});
}
// ! [plugin:compile_model]

// ! [plugin:compile_model_with_remote]
std::shared_ptr<ov::ICompiledModel> ov::template_plugin::Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::SoPtr<ov::IRemoteContext>& context) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::compile_model");

    Configuration fullConfig;
    {
        auto _properties = properties;
        // remove not supported properties which are consumed by compile_model
        _properties.erase(ov::loaded_from_cache.name());
        _properties.erase(ov::hint::compiled_blob.name());
        fullConfig = Configuration{_properties, m_cfg};
    }

    fullConfig.streams_executor_config = ov::threading::IStreamsExecutor::Config{stream_executor_name,
                                                                                 fullConfig.streams,
                                                                                 fullConfig.threads_per_stream};
    auto streamsExecutorConfig =
        ov::threading::IStreamsExecutor::Config::make_default_multi_threaded(fullConfig.streams_executor_config);
    fullConfig.streams = streamsExecutorConfig.get_streams();
    fullConfig.threads = streamsExecutorConfig.get_threads();
    fullConfig.threads_per_stream = streamsExecutorConfig.get_threads_per_stream();

    return std::make_shared<CompiledModel>(
        model->clone(),
        shared_from_this(),
        context,
        fullConfig.exclusive_async_requests
            ? get_executor_manager()->get_executor(template_exclusive_executor)
            : get_executor_manager()->get_idle_cpu_streams_executor(streamsExecutorConfig),
        fullConfig,
        false);
}
// ! [plugin:compile_model_with_remote]

// ! [plugin:import_model]
std::shared_ptr<ov::ICompiledModel> ov::template_plugin::Plugin::import_model(std::istream& model,
                                                                              const ov::AnyMap& properties) const {
    return import_model(model, {}, properties);
}
// ! [plugin:import_model]

// ! [plugin:import_model_with_remote]
std::shared_ptr<ov::ICompiledModel> ov::template_plugin::Plugin::import_model(
    std::istream& model,
    const ov::SoPtr<ov::IRemoteContext>& context,
    const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::import_model");

    auto import_properties = properties;
    const auto loaded_from_cache = consume_cache_import_hints(import_properties);

    auto weights = get_model_weights(properties);
    auto ov_model = get_ov_model_from_blob(*this, weights, model.tellg(), properties);

    Configuration config;
    const auto hsm_start = model.tellg();
    if (ov_model) {
        // Already fully imported from the hint above - no need to parse `model` at all.
        config = Configuration{import_properties, m_cfg};
    } else if (auto hsm_reader = ov::runtime::hsm::Reader::open(model)) {
        std::tie(ov_model, config) =
            read_hsm_blob(get_core(), get_runtime_requirements(), m_cfg, *hsm_reader, import_properties, weights);
    } else {
        // hsm reader failed on open (not valid hsm blob) - fall back to legacy path and restore stream possition
        // when legacy import removed this should be import error
        model.clear();
        model.seekg(hsm_start);

        config = Configuration{import_properties, m_cfg};
        // read XML content
        const auto model_xml = get_model_str(model);

        // read blob content
        if (!weights) {
            weights = get_model_weights(model);
        }

        ov_model = get_core()->read_model(model_xml, weights);
    }

    const auto task_executor = get_executor_manager()->get_idle_cpu_streams_executor(
        ov::threading::IStreamsExecutor::Config::make_default_multi_threaded(config.streams_executor_config));
    return std::make_shared<CompiledModel>(ov_model,
                                           shared_from_this(),
                                           context,
                                           task_executor,
                                           config,
                                           loaded_from_cache);
}
// ! [plugin:import_model_with_remote]

std::shared_ptr<ov::ICompiledModel> ov::template_plugin::Plugin::import_model(const ov::Tensor& model,
                                                                              const ov::AnyMap& properties) const {
    return import_model(model, {}, properties);
}

std::shared_ptr<ov::ICompiledModel> ov::template_plugin::Plugin::import_model(
    const ov::Tensor& model,
    const ov::SoPtr<ov::IRemoteContext>& context,
    const ov::AnyMap& properties) const {
    if (auto reader =
            ov::runtime::hsm::Reader::open(static_cast<const std::byte*>(model.data()), model.get_byte_size())) {
        auto import_properties = properties;
        const auto loaded_from_cache = consume_cache_import_hints(import_properties);
        const auto [imported_model, imported_config] = read_hsm_blob(get_core(),
                                                                     get_runtime_requirements(),
                                                                     m_cfg,
                                                                     *reader,
                                                                     import_properties,
                                                                     get_model_weights(properties),
                                                                     to_aligned_buffer(model));

        const auto executor = get_executor_manager()->get_idle_cpu_streams_executor(
            ov::threading::IStreamsExecutor::Config::make_default_multi_threaded(
                imported_config.streams_executor_config));
        return std::make_shared<CompiledModel>(imported_model,
                                               shared_from_this(),
                                               context,
                                               executor,
                                               imported_config,
                                               loaded_from_cache);
    } else {
        // legacy format
        ov::SharedStreamBuffer buffer{model.data(), model.get_byte_size()};
        std::istream stream{&buffer};
        return import_model(stream, properties);
    }
}

// ! [plugin:query_model]
ov::SupportedOpsMap ov::template_plugin::Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                                             const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::TemplatePlugin, "Plugin::query_model");

    Configuration fullConfig{properties, m_cfg, false};

    OPENVINO_ASSERT(model, "OpenVINO Model is empty!");

    auto supported = ov::get_supported_nodes(
        model,
        [&](std::shared_ptr<ov::Model>& model) {
            // skip transformations in case of user config
            if (fullConfig.disable_transformations)
                return;
            // 1. It is needed to apply all transformations as it is done in compile_model
            transform_model(model);
        },
        [&](std::shared_ptr<ov::Node> node) {
            // 2. Сheck whether node is supported
            ov::OpSet op_super_set;
#define _OPENVINO_OP_REG(NAME, NAMESPACE) op_super_set.insert<NAMESPACE::NAME>();
        // clang-format off
#include "openvino/opsets/opset1_tbl.hpp"
#include "openvino/opsets/opset2_tbl.hpp"
#include "openvino/opsets/opset3_tbl.hpp"
#include "openvino/opsets/opset4_tbl.hpp"
#include "openvino/opsets/opset5_tbl.hpp"
#include "openvino/opsets/opset6_tbl.hpp"
#include "openvino/opsets/opset7_tbl.hpp"
#include "openvino/opsets/opset8_tbl.hpp"
#include "openvino/opsets/opset9_tbl.hpp"
#include "openvino/opsets/opset10_tbl.hpp"
#include "openvino/opsets/opset11_tbl.hpp"
#include "openvino/opsets/opset12_tbl.hpp"
#include "openvino/opsets/opset13_tbl.hpp"
#include "openvino/opsets/opset14_tbl.hpp"
#include "openvino/opsets/opset15_tbl.hpp"
#include "openvino/opsets/opset16_tbl.hpp"
#include "openvino/opsets/opset17_tbl.hpp"
        // clang-format on
#undef _OPENVINO_OP_REG
            return op_super_set.contains_type(node->get_type_info());
        });

    // 3. Produce the result
    ov::SupportedOpsMap res;
    for (auto&& layerName : supported) {
        res.emplace(layerName, get_device_name() + "." + std::to_string(m_cfg.device_id));
    }

    return res;
}
// ! [plugin:query_model]

// ! [plugin:set_property]
void ov::template_plugin::Plugin::set_property(const ov::AnyMap& properties) {
    m_cfg = Configuration{properties, m_cfg};
}
// ! [plugin:set_property]

// ! [plugin:get_property]
ov::Any ov::template_plugin::Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::available_devices,
                                                    ov::supported_properties,
                                                    ov::device::full_name,
                                                    ov::device::architecture,
                                                    ov::device::capabilities,
                                                    ov::device::type,
                                                    ov::range_for_async_infer_requests,
                                                    ov::execution_devices,
                                                    ov::compatibility_check};
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{
            ov::device::id,
            ov::enable_profiling,
            ov::hint::performance_mode,
            ov::hint::num_requests,
            ov::hint::inference_precision,
            ov::hint::execution_mode,
            ov::num_streams,
            ov::template_plugin::disable_transformations,
            ov::log::level,
            ov::hint::model_priority,
            ov::hint::enable_hyper_threading,
            ov::hint::enable_cpu_pinning,
            ov::hint::scheduling_core_type,
            ov::compilation_num_threads,
            ov::inference_num_threads,
            ov::weights_path,
            ov::cache_mode,
        };
        return rw_properties;
    };
    if (ov::supported_properties == name) {
        auto ro_properties = default_ro_properties();
        auto rw_properties = default_rw_properties();

        std::vector<ov::PropertyName> supported_properties;
        supported_properties.reserve(ro_properties.size() + rw_properties.size());
        supported_properties.insert(supported_properties.end(), ro_properties.begin(), ro_properties.end());
        supported_properties.insert(supported_properties.end(), rw_properties.begin(), rw_properties.end());
        return supported_properties;
    } else if (ov::internal::supported_properties == name) {
        auto internal_properties = decltype(ov::internal::supported_properties)::value_type{
            ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::inference_num_threads.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::internal::threads_per_stream.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::internal::compiled_model_runtime_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::internal::cache_header_alignment.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::internal::emit_hsm_format.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::template_plugin::export_hsm_format.name(), ov::PropertyMutability::RW},
        };
        // Only the HSM reader imports from a mapped blob without copying; the legacy path would copy it.
        if (m_cfg.export_hsm_format) {
            internal_properties.emplace_back(ov::internal::caching_with_mmap.name(), ov::PropertyMutability::RO);
        }
        return internal_properties;
    } else if (ov::available_devices == name) {
        // TODO: fill list of available devices
        return decltype(ov::available_devices)::value_type{{""}};
    } else if (ov::device::full_name == name) {
        return decltype(ov::device::full_name)::value_type{"Template Device Full Name"};
    } else if (ov::device::architecture == name) {
        // TODO: return device architecture for device specified by DEVICE_ID config
        return decltype(ov::device::architecture)::value_type{get_device_name()};
    } else if (ov::device::type == name) {
        return decltype(ov::device::type)::value_type{ov::device::Type::INTEGRATED};
    } else if (ov::internal::caching_properties == name) {
        return decltype(ov::internal::caching_properties)::value_type{caching_properties()};
    } else if (ov::device::capabilities == name) {
        // TODO: fill actual list of supported capabilities: e.g. Template device supports only FP32 and EXPORT_IMPORT
        return decltype(ov::device::capabilities)::value_type{ov::device::capability::FP32,
                                                              ov::device::capability::EXPORT_IMPORT};
    } else if (ov::execution_devices == name) {
        return decltype(ov::execution_devices)::value_type{get_device_name()};
    } else if (ov::range_for_async_infer_requests == name) {
        return decltype(ov::range_for_async_infer_requests)::value_type{1, 1, 1};
    } else if (ov::compatibility_check == name) {
        if (auto it = arguments.find(ov::runtime_requirements.name()); it != arguments.end()) {
            const auto& requirements = it->second.as<std::string>();
            if (!requirements.empty()) {
                if (const auto pos = requirements.find(get_runtime_requirements()); pos == 0) {
                    return ov::CompatibilityCheck::SUPPORTED;
                } else {
                    return ov::CompatibilityCheck::UNSUPPORTED;
                }
            }
        }
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    } else {
        return m_cfg.Get(name);
    }
}
// ! [plugin:get_property]

// ! [plugin:create_plugin_engine]
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_template_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::template_plugin::Plugin, version)
// ! [plugin:create_plugin_engine]

// ! [plugin:enumerate_dispatch_devices]
// This plugin does not participate in device-name dispatch; export the probe as a stub.
OV_DEFINE_PLUGIN_ENUMERATE_STUB()
// ! [plugin:enumerate_dispatch_devices]

// ! [plugin:get_runtime_requirements]
std::string_view ov::template_plugin::Plugin::get_runtime_requirements() const {
    return "TEMPLATE_PLUGIN_CAPABILITIES_v1";
}
// ! [plugin:get_runtime_requirements]
