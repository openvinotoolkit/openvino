// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <fstream>
#include <memory>

#include "itt.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/pass/manager.hpp"
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

namespace {
static constexpr const char* wait_executor_name = "TemplateWaitExecutor";
static constexpr const char* stream_executor_name = "TemplateStreamsExecutor";
static constexpr const char* template_exclusive_executor = "TemplateExecutor";

uint64_t get_blob_data_size(std::istream& model) {
    uint64_t size = 0;
    model.read(reinterpret_cast<char*>(&size), sizeof(size));
    return size;
}

std::string get_model_str(std::istream& model) {
    const auto model_size = std::min<uint64_t>(model.rdbuf()->in_avail(), get_blob_data_size(model));
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
    const auto weights_size = std::min<uint64_t>(model.rdbuf()->in_avail(), get_blob_data_size(model));
    return weights_size != 0 ? read_weights(model, weights_size) : ov::Tensor();
}

ov::Tensor get_model_weights(const ov::AnyMap& properties) {
    ov::Tensor weights;
    if (auto weights_path = properties.find(ov::weights_path.name()); weights_path != properties.end()) {
        const auto w_path = std::filesystem::path(weights_path->second.as<std::string>());
        if (auto size = ov::util::file_size(w_path); size > 0) {
            auto f_weights = std::ifstream(w_path, std::ios::binary | std::ios::in);
            weights = read_weights(f_weights, size);
        }
    }
    return weights;
}

std::shared_ptr<ov::Model> get_ov_model_from_blob(const ov::template_plugin::Plugin& plugin,
                                                  ov::Tensor& weights,
                                                  size_t offset,
                                                  const ov::AnyMap& properties) {
    if (auto blob_it = properties.find(ov::hint::compiled_blob.name()); blob_it != properties.end()) {
        if (auto&& blob = blob_it->second.as<ov::Tensor>(); blob) {
            ov::SharedStreamBuffer shared_buffer(reinterpret_cast<char*>(blob.data()), blob.get_byte_size());
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

    // check ov::loaded_from_cache property and erase it due to not needed any more.
    auto _properties = properties;
    const auto& it = _properties.find(ov::loaded_from_cache.name());
    bool loaded_from_cache = false;
    if (it != _properties.end()) {
        loaded_from_cache = it->second.as<bool>();
        _properties.erase(it);
    }
    _properties.erase(ov::hint::compiled_blob.name());

    auto fullConfig = Configuration{_properties, m_cfg};
    fullConfig.streams_executor_config = ov::threading::IStreamsExecutor::Config{stream_executor_name,
                                                                                 fullConfig.streams,
                                                                                 fullConfig.threads_per_stream};
    auto weights = get_model_weights(properties);
    if (!weights) {
        if (auto model_hint = properties.find(ov::hint::model.name()); model_hint != properties.end()) {
            if (auto m = model_hint->second.as<std::shared_ptr<ov::Model>>()) {
                if (m->has_rt_info("__weights_path")) {
                    AnyMap rt_info;
                    auto p = m->get_rt_info<std::string>("__weights_path");
                    rt_info[ov::weights_path.name()] = m->get_rt_info<ov::Any>("__weights_path");
                    weights = get_model_weights(rt_info);
                }
            }
        }
    }
    auto ov_model = get_ov_model_from_blob(*this, weights, model.tellg(), properties);
    if (!ov_model) {
        // read XML content
        std::string xmlString = get_model_str(model);

        // read blob content
        if (!weights) {
            weights = get_model_weights(model);
        }

        ov_model = get_core()->read_model(xmlString, weights);
    }
    auto streamsExecutorConfig =
        ov::threading::IStreamsExecutor::Config::make_default_multi_threaded(fullConfig.streams_executor_config);
    fullConfig.streams = streamsExecutorConfig.get_streams();
    fullConfig.threads = streamsExecutorConfig.get_threads();
    fullConfig.threads_per_stream = streamsExecutorConfig.get_threads_per_stream();
    auto compiled_model =
        std::make_shared<CompiledModel>(ov_model,
                                        shared_from_this(),
                                        context,
                                        get_executor_manager()->get_idle_cpu_streams_executor(streamsExecutorConfig),
                                        fullConfig,
                                        loaded_from_cache);
    return compiled_model;
}
// ! [plugin:import_model_with_remote]

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
                                                    ov::execution_devices};
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
        return decltype(ov::internal::supported_properties)::value_type{
            ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::internal::exclusive_async_requests.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::inference_num_threads.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::internal::threads_per_stream.name(), ov::PropertyMutability::RW},
            ov::PropertyName{ov::internal::compiled_model_runtime_properties.name(), ov::PropertyMutability::RO}};
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
        return decltype(ov::internal::caching_properties)::value_type{ov::device::architecture};
    } else if (ov::device::capabilities == name) {
        // TODO: fill actual list of supported capabilities: e.g. Template device supports only FP32 and EXPORT_IMPORT
        return decltype(ov::device::capabilities)::value_type{ov::device::capability::FP32,
                                                              ov::device::capability::EXPORT_IMPORT};
    } else if (ov::execution_devices == name) {
        return decltype(ov::execution_devices)::value_type{get_device_name()};
    } else if (ov::range_for_async_infer_requests == name) {
        return decltype(ov::range_for_async_infer_requests)::value_type{1, 1, 1};
    } else {
        return m_cfg.Get(name);
    }
}
// ! [plugin:get_property]

// ! [plugin:create_plugin_engine]
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_template_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::template_plugin::Plugin, version)
// ! [plugin:create_plugin_engine]
