// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>

#include "async_infer_request.hpp"
#include "blob_serialization.hpp"
#include "graph_debug_dump.hpp"
#include "hsm_writer_handler.hpp"
#include "itt.hpp"
#include "op/device_subgraph.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/hsm_deferred_writer.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/xml_parse_utils.hpp"
#include "perf_log.hpp"
#include "properties.hpp"

namespace {

ov::SoPtr<ov::ICompiledModel> import_submodel_blob(const std::shared_ptr<ov::ICore>& core,
                                                   const std::string& device,
                                                   const ov::AnyMap& load_config,
                                                   const ov::hetero::HsmLocatedSection& located,
                                                   const ov::hetero::HsmBlobSource& source) {
    if (source.stream != nullptr) {
        auto& stream = *source.stream;
        stream.clear();
        stream.seekg(source.start + static_cast<std::streamoff>(located.entry.offset));
        OPENVINO_ASSERT(stream, "Failed to seek to a HETERO compiled submodel section");
        ov::hetero::BoundedStreamBuffer buffer{stream, located.entry.size};
        std::istream payload{&buffer};
        auto compiled_model = core->import_model(payload, device, load_config);
        // The sub-device may leave the shared stream in a failed state; other sections are read next.
        stream.clear();
        return compiled_model;
    }

    const auto view = located.section.view();
    OPENVINO_ASSERT(!view.empty(), "HETERO compiled submodel section is empty");
    if (source.tensor.get_element_type() == ov::element::u8 && source.tensor.get_shape().size() == 1) {
        try {
            // A view of the mapped container lets the sub-device keep its weights zero-copy.
            return core->import_model(ov::hetero::read_hsm_tensor(located, source), device, load_config);
        } catch (const ov::Exception&) {
            // The plugin wrapper reports a missing tensor import as a plain exception; a broken blob fails again below.
        }
    }
    ov::SharedStreamBuffer buffer{view.data(), view.size()};
    std::istream payload{&buffer};
    return core->import_model(payload, device, load_config);
}

class CountingOutputBuffer : public std::streambuf {
public:
    std::uint64_t size() const {
        return _size;
    }

protected:
    std::streamsize xsputn(const char*, std::streamsize count) override {
        if (count <= 0) {
            return 0;
        }

        const auto bytes = static_cast<std::uint64_t>(count);
        OPENVINO_ASSERT(_size <= std::numeric_limits<std::uint64_t>::max() - bytes,
                        "HETERO compiled blob payload size is too large");
        _size += bytes;
        return count;
    }

    int_type overflow(int_type ch) override {
        if (traits_type::eq_int_type(ch, traits_type::eof())) {
            return traits_type::not_eof(ch);
        }

        OPENVINO_ASSERT(_size < std::numeric_limits<std::uint64_t>::max(),
                        "HETERO compiled blob payload size is too large");
        ++_size;
        return ch;
    }

private:
    std::uint64_t _size = 0;
};

template <typename ExportPayload>
void write_framed_payload_two_pass(std::ostream& modelStream,
                                   char payloadType,
                                   ExportPayload exportPayload,
                                   const char* fieldName) {
    CountingOutputBuffer countingBuffer;
    std::ostream countingStream(&countingBuffer);
    exportPayload(countingStream);
    OPENVINO_ASSERT(countingStream, "Failed to export HETERO compiled blob ", fieldName);

    const auto payloadSize = countingBuffer.size();
    modelStream.write(&payloadType, sizeof(payloadType));
    OPENVINO_ASSERT(modelStream, "Failed to write HETERO compiled blob payload type");
    modelStream.write(reinterpret_cast<const char*>(&payloadSize), sizeof(payloadSize));
    OPENVINO_ASSERT(modelStream, "Failed to write HETERO compiled blob payload size");

    try {
        exportPayload(modelStream);
    } catch (const ov::NotImplemented&) {
        OPENVINO_THROW("Cannot continue HETERO compiled blob export after starting framed payload ",
                       fieldName,
                       ". Child plugin reported NotImplemented during streaming pass");
    }
    OPENVINO_ASSERT(modelStream, "Failed to write HETERO compiled blob ", fieldName);
}

void write_ir_framed_payload(std::ostream& modelStream, const std::shared_ptr<ov::Model>& model) {
    write_framed_payload_two_pass(
        modelStream,
        ov::hetero::IR_PAYLOAD,
        [&](std::ostream& payloadStream) {
            ov::hetero::write_ir_payload(payloadStream, model);
        },
        "IR payload");
}

template <typename ExportPayload>
void write_framed_payload_via_seekable_stream(std::ostream& modelStream,
                                              char payloadType,
                                              ExportPayload exportPayload,
                                              const char* fieldName) {
    std::stringstream payloadStream(std::ios::in | std::ios::out | std::ios::binary);
    exportPayload(payloadStream);
    OPENVINO_ASSERT(payloadStream, "Failed to export HETERO compiled blob ", fieldName);

    payloadStream.clear();
    payloadStream.seekp(0, std::ios::end);
    const auto payloadEndPos = payloadStream.tellp();
    OPENVINO_ASSERT(payloadEndPos != std::streampos(-1),
                    "Failed to determine HETERO compiled blob ",
                    fieldName,
                    " size");

    const auto payloadSize = static_cast<std::uint64_t>(payloadEndPos);
    payloadStream.clear();
    payloadStream.seekg(0, std::ios::beg);
    OPENVINO_ASSERT(payloadStream, "Failed to rewind HETERO compiled blob ", fieldName);

    ov::hetero::write_framed_payload(modelStream, payloadType, payloadStream, payloadSize);
}

ov::hetero::Configuration hetero_read_config(const pugi::xml_node& hetero_node, const ov::hetero::Configuration& base) {
    using namespace ov::util::pugixml;

    ov::AnyMap properties;
    auto hetero_configs_node = hetero_node.child("hetero_config");
    FOREACH_CHILD (hetero_config_node, hetero_configs_node, "config") {
        properties.emplace(get_str_attr(hetero_config_node, "key"), get_str_attr(hetero_config_node, "value"));
    }
    return ov::hetero::Configuration(properties, base);
}

ov::hetero::SubgraphsMappingInfo hetero_read_mapping(const pugi::xml_node& hetero_node) {
    using namespace ov::util::pugixml;

    ov::hetero::SubgraphsMappingInfo mapping_info;
    auto inputs_map_node = hetero_node.child("inputs_to_submodels_inputs");
    FOREACH_CHILD (xml_node, inputs_map_node, "pair") {
        mapping_info._inputs_to_submodels_inputs.emplace_back(get_uint64_attr(xml_node, "submodel_idx"),
                                                              get_uint64_attr(xml_node, "node_idx"));
    }
    auto outputs_map_node = hetero_node.child("outputs_to_submodels_outputs");
    FOREACH_CHILD (xml_node, outputs_map_node, "pair") {
        mapping_info._outputs_to_submodels_outputs.emplace_back(get_uint64_attr(xml_node, "submodel_idx"),
                                                                get_uint64_attr(xml_node, "node_idx"));
    }
    auto submodels_input_to_prev_output_node = hetero_node.child("submodels_input_to_prev_output");
    FOREACH_CHILD (xml_node, submodels_input_to_prev_output_node, "record") {
        std::pair<uint64_t, uint64_t> in_pair = {get_uint64_attr(xml_node, "in_submodel_idx"),
                                                 get_uint64_attr(xml_node, "in_node_idx")};
        std::pair<uint64_t, uint64_t> out_pair = {get_uint64_attr(xml_node, "out_submodel_idx"),
                                                  get_uint64_attr(xml_node, "out_node_idx")};
        mapping_info._submodels_input_to_prev_output.emplace(in_pair, out_pair);
    }
    return mapping_info;
}

// Metadata shared by the legacy and the HSM blob: everything but the name and the legacy format version.
template <typename Submodels>
void hetero_write_metadata(pugi::xml_node& hetero_node,
                           const ov::hetero::SubgraphsMappingInfo& mapping_info,
                           const Submodels& submodels,
                           const ov::hetero::Configuration& cfg) {
    auto inputs_map_node = hetero_node.append_child("inputs_to_submodels_inputs");
    for (const auto& it : mapping_info._inputs_to_submodels_inputs) {
        auto xml_node = inputs_map_node.append_child("pair");
        xml_node.append_attribute("submodel_idx").set_value(std::to_string(it.first).c_str());
        xml_node.append_attribute("node_idx").set_value(std::to_string(it.second).c_str());
    }
    auto outputs_map_node = hetero_node.append_child("outputs_to_submodels_outputs");
    for (const auto& it : mapping_info._outputs_to_submodels_outputs) {
        auto xml_node = outputs_map_node.append_child("pair");
        xml_node.append_attribute("submodel_idx").set_value(std::to_string(it.first).c_str());
        xml_node.append_attribute("node_idx").set_value(std::to_string(it.second).c_str());
    }

    auto submodels_input_to_prev_output_node = hetero_node.append_child("submodels_input_to_prev_output");
    for (const auto& it : mapping_info._submodels_input_to_prev_output) {
        auto xml_node = submodels_input_to_prev_output_node.append_child("record");
        xml_node.append_attribute("in_submodel_idx").set_value(std::to_string(it.first.first).c_str());
        xml_node.append_attribute("in_node_idx").set_value(std::to_string(it.first.second).c_str());
        xml_node.append_attribute("out_submodel_idx").set_value(std::to_string(it.second.first).c_str());
        xml_node.append_attribute("out_node_idx").set_value(std::to_string(it.second.second).c_str());
    }

    auto subnetworksNode = hetero_node.append_child("compiled_submodels");
    for (const auto& submodel : submodels) {
        auto sub_network_node = subnetworksNode.append_child("compiled_submodel");
        sub_network_node.append_attribute("device").set_value(submodel.device.c_str());
    }

    auto heteroConfigsNode = hetero_node.append_child("hetero_config");
    for (const auto& config : cfg.get_hetero_properties()) {
        auto hetero_config_node = heteroConfigsNode.append_child("config");
        hetero_config_node.append_attribute("key").set_value(config.first.c_str());
        hetero_config_node.append_attribute("value").set_value(config.second.as<std::string>().c_str());
    }
}

}  // namespace

ov::hetero::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                         const std::vector<ov::hetero::SubmodelInfo>& submodels,
                                         const SubgraphsMappingInfo& mapping_info,
                                         const std::shared_ptr<const ov::IPlugin>& plugin,
                                         ov::hetero::RemoteContext::Ptr context,
                                         const Configuration& cfg)
    : ov::ICompiledModel(model, plugin, context),
      m_cfg(cfg),
      m_name(model->get_friendly_name()),
      m_loaded_from_cache(false),
      m_mapping_info(mapping_info) {
    try {
        compile_model(submodels);
    } catch (const std::exception& e) {
        OPENVINO_THROW("Standard exception from compilation library: ", e.what());
    } catch (...) {
        OPENVINO_THROW("Generic exception is thrown");
    }
}

void ov::hetero::CompiledModel::compile_model(const std::vector<ov::hetero::SubmodelInfo>& submodels) {
    using clock = std::chrono::steady_clock;
    const auto to_ms = [](clock::duration d) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
    };
    const bool perf_logging_enabled = perf_log_enabled(PerfLogLevel::Summary);
    clock::time_point t0{};
    clock::duration submodel_total_time{};
    if (perf_logging_enabled) {
        t0 = clock::now();
    }

    const bool add_exclusive = submodels.size() > 1;
    const auto& hetero_plugin = get_hetero_plugin();
    const auto& core = hetero_plugin->get_core();
    const auto& device_properties = m_cfg.get_device_properties();

    m_compiled_submodels.clear();
    m_compiled_submodels.reserve(submodels.size());

    size_t submodel_index = 0;
    for (const auto& [device, sub_model] : submodels) {
        clock::time_point t_submodel_start{};
        clock::time_point t_compile_start{};
        clock::time_point t_compile_end{};
        clock::time_point t_submodel_end{};
        if (perf_logging_enabled) {
            t_submodel_start = clock::now();
        }
        // get meta devices properties for the target device
        auto meta_devices = hetero_plugin->get_properties_per_device(device, device_properties);

        // disable caching for subgraphs, because the whole HETERO model is cached
        auto device_config = meta_devices.at(device);
        device_config[ov::cache_dir.name()] = "";

        // set exclusive_async_requests in case when model is split
        if (add_exclusive) {
            auto supported_internal_properties = core->get_property(device, ov::internal::supported_properties);
            if (std::find(supported_internal_properties.begin(),
                          supported_internal_properties.end(),
                          ov::internal::exclusive_async_requests) != supported_internal_properties.end()) {
                // adds property if it is not set yet
                device_config.insert(ov::internal::exclusive_async_requests(true));
            }
        }

        // compile the submodel and add to the compiled submodels list
        CompiledModelDesc desc;
        desc.device = device;
        desc.model = sub_model;
        if (perf_logging_enabled) {
            t_compile_start = clock::now();
        }
        desc.compiled_model = core->compile_model(sub_model, device, device_config);
        if (perf_logging_enabled) {
            t_compile_end = clock::now();
        }
        m_compiled_submodels.emplace_back(std::move(desc));

        if (perf_logging_enabled) {
            t_submodel_end = clock::now();
            submodel_total_time += t_submodel_end - t_submodel_start;
            HETERO_PERF_LOG_LEVEL(PerfLogLevel::Summary,
                                  "CompiledModel::compile_model submodel[",
                                  submodel_index,
                                  "] device=",
                                  device,
                                  " ops=",
                                  sub_model ? sub_model->get_ops().size() : 0,
                                  " total=",
                                  to_ms(t_submodel_end - t_submodel_start),
                                  " ms, core_compile=",
                                  to_ms(t_compile_end - t_compile_start),
                                  " ms");
        }
        ++submodel_index;
    }

    clock::time_point t_set_io_start{};
    clock::time_point t_set_io_end{};
    if (perf_logging_enabled) {
        t_set_io_start = clock::now();
    }
    set_inputs_and_outputs();
    if (perf_logging_enabled) {
        t_set_io_end = clock::now();
        HETERO_PERF_LOG_LEVEL(PerfLogLevel::Summary,
                              "CompiledModel::compile_model timing: total=",
                              to_ms(t_set_io_end - t0),
                              " ms, submodels=",
                              submodels.size(),
                              ", submodel_total=",
                              to_ms(submodel_total_time),
                              " ms, set_inputs_and_outputs=",
                              to_ms(t_set_io_end - t_set_io_start),
                              " ms");
    }
}

ov::hetero::CompiledModel::CompiledModel(std::istream& model,
                                         const std::shared_ptr<const ov::IPlugin>& plugin,
                                         const Configuration& cfg,
                                         const bool loaded_from_cache)
    : ov::ICompiledModel(nullptr, plugin),
      m_cfg(cfg),
      m_name(),
      m_loaded_from_cache(loaded_from_cache) {
    const auto start = model.tellg();
    if (auto reader = ov::runtime::hsm::Reader::open(model)) {
        import_container(*reader, HsmBlobSource{&model, start, {}});
        return;
    }
    // Not an HSM container: rewind and read the legacy layout. A non-seekable stream was not consumed.
    if (start != std::streampos(-1)) {
        model.clear();
        model.seekg(start);
    }

    std::string heteroXmlStr;
    std::getline(model, heteroXmlStr);

    pugi::xml_document heteroXmlDoc;
    pugi::xml_parse_result res = heteroXmlDoc.load_string(heteroXmlStr.c_str());

    if (res.status != pugi::status_ok)
        OPENVINO_THROW("Failed to read Hetero device xml header");

    using namespace ov::util::pugixml;

    pugi::xml_node heteroNode = heteroXmlDoc.document_element();
    m_name = get_str_attr(heteroNode, "name");
    const auto version_attr = heteroNode.attribute(HETERO_BLOB_FORMAT_VERSION_ATTR);
    std::uint32_t blob_format_version = 1;
    if (version_attr) {
        const std::string version_str = version_attr.value();
        if (version_str == "1") {
            blob_format_version = 1;
        } else if (version_str == std::to_string(HETERO_BLOB_FORMAT_VERSION)) {
            blob_format_version = HETERO_BLOB_FORMAT_VERSION;
        } else {
            OPENVINO_THROW("Unsupported HETERO compiled blob format version: ",
                           version_str,
                           ". Supported versions are 1 (legacy unframed) and ",
                           HETERO_BLOB_FORMAT_VERSION,
                           " (framed). ");
        }
    }
    const bool is_framed_blob = blob_format_version == HETERO_BLOB_FORMAT_VERSION;

    // clang-format off
    m_cfg = hetero_read_config(heteroNode, m_cfg);

    pugi::xml_node subnetworksNode = heteroNode.child("compiled_submodels");
    FOREACH_CHILD(subnetworkNode, subnetworksNode, "compiled_submodel") {
        auto device = get_str_attr(subnetworkNode, "device");
        auto loadConfig = get_load_config(device);

        ov::SoPtr<ov::ICompiledModel> compiled_model;
        std::shared_ptr<ov::Model> ov_model;

        const auto& core = plugin->get_core();
        if (is_framed_blob) {
            const auto payloadHeader = read_payload_header(model);

            if (payloadHeader.type == COMPILED_BLOB_PAYLOAD &&
                payloadHeader.size <= MAX_IN_MEMORY_COMPILED_PAYLOAD_SIZE) {
                OPENVINO_ASSERT(payloadHeader.size <=
                                    static_cast<std::uint64_t>(std::numeric_limits<std::string::size_type>::max()),
                                "HETERO compiled blob compiled submodel payload size is too large: ",
                                payloadHeader.size);
                std::string payload(static_cast<std::string::size_type>(payloadHeader.size), '\0');
                ov::hetero::read_payload_bytes(model, payload.data(), payloadHeader.size, "compiled submodel payload");
                std::istringstream payloadStream(payload);
                compiled_model = core->import_model(payloadStream, device, loadConfig);
            } else if (payloadHeader.type == IR_PAYLOAD) {
                read_ir_payload(model, core, device, loadConfig, ov_model, compiled_model, payloadHeader.size);
            } else if (payloadHeader.type == COMPILED_BLOB_PAYLOAD) {
                BoundedStreamBuffer buffer{model, payloadHeader.size};
                std::istream payloadStream{&buffer};
                compiled_model = core->import_model(payloadStream, device, loadConfig);
                buffer.consume_remaining_payload();
            } else {
                OPENVINO_THROW("Unsupported HETERO compiled submodel payload type: ", payloadHeader.type);
            }
        } else if (core->device_supports_model_caching(device)) {
            compiled_model = core->import_model(model, device, loadConfig);
        } else {
            read_ir_payload(model,
                            core,
                            device,
                            loadConfig,
                            ov_model,
                            compiled_model,
                            std::numeric_limits<std::uint64_t>::max());
        }

        m_compiled_submodels.emplace_back(ov::hetero::CompiledModel::CompiledModelDesc{
            device,
            ov_model,
            compiled_model,
        });
    }

    m_mapping_info = hetero_read_mapping(heteroNode);
    // clang-format on
    set_inputs_and_outputs();
}

ov::hetero::CompiledModel::CompiledModel(const ov::Tensor& model,
                                         const std::shared_ptr<const ov::IPlugin>& plugin,
                                         const Configuration& cfg,
                                         const bool loaded_from_cache)
    : ov::ICompiledModel(nullptr, plugin),
      m_cfg(cfg),
      m_name(),
      m_loaded_from_cache(loaded_from_cache) {
    auto reader = ov::runtime::hsm::Reader::open(static_cast<const std::byte*>(model.data()), model.get_byte_size());
    OPENVINO_ASSERT(reader, "[HETERO] Cannot import compiled blob: invalid HSM container.");
    import_container(*reader, HsmBlobSource{nullptr, {}, model});
}

ov::AnyMap ov::hetero::CompiledModel::get_load_config(const std::string& device) const {
    auto meta_devices = get_hetero_plugin()->get_properties_per_device(device, m_cfg.get_device_properties());
    assert(meta_devices.size() == 1);
    return meta_devices[device];
}

void ov::hetero::CompiledModel::import_container(const ov::runtime::hsm::Reader& reader, const HsmBlobSource& source) {
    using namespace ov::util::pugixml;

    // Before any other section is read: an incompatible blob must be rejected without touching its payload.
    check_hsm_runtime_requirements(reader);

    // Records the sections without reading them, so the structure is validated before any payload is touched.
    HsmReaderHandler handler;
    reader.read_sections({&handler});
    handler.check_structure();

    pugi::xml_document hetero_xml_doc;
    OPENVINO_ASSERT(
        hetero_xml_doc.load_string(read_hsm_string(handler.metadata().section).c_str()).status == pugi::status_ok,
        "Failed to read Hetero device xml header");
    const auto hetero_node = hetero_xml_doc.document_element();
    m_name = get_str_attr(hetero_node, "name");
    m_cfg = hetero_read_config(hetero_node, m_cfg);

    std::vector<std::string> devices;
    const auto sub_networks_node = hetero_node.child("compiled_submodels");
    FOREACH_CHILD (sub_network_node, sub_networks_node, "compiled_submodel") {
        devices.push_back(get_str_attr(sub_network_node, "device"));
    }
    OPENVINO_ASSERT(handler.submodel_count() == devices.size(),
                    "[HETERO] Cannot import compiled blob: expected sections of ",
                    devices.size(),
                    " submodels, found ",
                    handler.submodel_count());

    const auto& core = get_hetero_plugin()->get_core();
    for (size_t i = 0; i < devices.size(); ++i) {
        const auto& device = devices[i];
        const auto& sections = handler.submodel(i);
        const auto load_config = get_load_config(device);

        ov::SoPtr<ov::ICompiledModel> compiled_model;
        std::shared_ptr<ov::Model> ov_model;
        if (sections.blob) {
            compiled_model = import_submodel_blob(core, device, load_config, *sections.blob, source);
        } else {
            const auto xml = read_hsm_string(sections.ir_xml->section);
            const auto weights = sections.ir_weights ? read_hsm_tensor(*sections.ir_weights, source) : ov::Tensor{};
            ov_model = core->read_model(xml, weights);
            compiled_model = core->compile_model(ov_model, device, load_config);
        }
        m_compiled_submodels.emplace_back(CompiledModelDesc{device, ov_model, compiled_model});
    }

    m_mapping_info = hetero_read_mapping(hetero_node);
    set_inputs_and_outputs();
}

std::shared_ptr<ov::ISyncInferRequest> ov::hetero::CompiledModel::create_sync_infer_request() const {
    return std::make_shared<ov::hetero::InferRequest>(
        std::static_pointer_cast<const ov::hetero::CompiledModel>(shared_from_this()));
}

std::shared_ptr<ov::IAsyncInferRequest> ov::hetero::CompiledModel::create_infer_request() const {
    auto internal_request = create_sync_infer_request();
    auto async_infer_request = std::make_shared<ov::hetero::AsyncInferRequest>(
        std::static_pointer_cast<ov::hetero::InferRequest>(internal_request),
        get_task_executor(),
        get_callback_executor());

    return async_infer_request;
}

void ov::hetero::CompiledModel::set_property(const ov::AnyMap& properties) {
    OPENVINO_THROW_NOT_IMPLEMENTED("It's not possible to set property of an already compiled model. "
                                   "Set property to Core::compile_model during compilation");
}

std::shared_ptr<const ov::Model> ov::hetero::CompiledModel::get_runtime_model() const {
    std::vector<std::shared_ptr<ov::Model>> rt_models;
    std::vector<std::shared_ptr<void>> shared_objects;
    // Collect runtime subgraphs
    rt_models.reserve(m_compiled_submodels.size());
    shared_objects.reserve(m_compiled_submodels.size());
    for (auto& compiled_submodel : m_compiled_submodels) {
        rt_models.push_back(compiled_submodel.compiled_model->get_runtime_model()->clone());
        shared_objects.push_back(compiled_submodel.compiled_model._so);
    }
    ov::hetero::merge_submodels(rt_models, m_mapping_info._submodels_input_to_prev_output);
    auto& runtime_graph = rt_models[0];
    OPENVINO_ASSERT(runtime_graph->inputs().size() == inputs().size());
    auto merged_shared_object = std::make_shared<std::vector<std::shared_ptr<void>>>(std::move(shared_objects));
    set_model_shared_object(
        *runtime_graph,
        std::shared_ptr<void>(std::move(merged_shared_object), reinterpret_cast<void*>(merged_shared_object.get())));
    return runtime_graph;
}

std::shared_ptr<const ov::hetero::Plugin> ov::hetero::CompiledModel::get_hetero_plugin() const {
    auto plugin = get_plugin();
    OPENVINO_ASSERT(plugin);
    auto hetero_plugin = std::static_pointer_cast<const ov::hetero::Plugin>(plugin);
    OPENVINO_ASSERT(hetero_plugin);
    return hetero_plugin;
}

ov::Any ov::hetero::CompiledModel::get_property(const std::string& name) const {
    const auto& add_ro_properties = [](const std::string& name, std::vector<ov::PropertyName>& properties) {
        properties.emplace_back(ov::PropertyName{name, ov::PropertyMutability::RO});
    };
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::model_name,
                                                    ov::optimal_number_of_infer_requests,
                                                    ov::execution_devices,
                                                    ov::loaded_from_cache,
                                                    ov::hetero::number_of_submodels};
        return ro_properties;
    };

    if (ov::supported_properties == name) {
        auto supported_properties = default_ro_properties();
        add_ro_properties(ov::supported_properties.name(), supported_properties);
        add_ro_properties(ov::device::properties.name(), supported_properties);
        add_ro_properties(ov::device::priorities.name(), supported_properties);
        return decltype(ov::supported_properties)::value_type(std::move(supported_properties));
    } else if (ov::device::properties == name) {
        ov::AnyMap all_devices = {};
        for (const auto& comp_model_desc : m_compiled_submodels) {
            ov::AnyMap device_properties = {};
            if (all_devices.count(comp_model_desc.device) == 0) {
                auto device_supported_props =
                    comp_model_desc.compiled_model->get_property(ov::supported_properties.name());
                for (auto&& property_name : device_supported_props.as<std::vector<ov::PropertyName>>())
                    device_properties[property_name] = comp_model_desc.compiled_model->get_property(property_name);
                all_devices[comp_model_desc.device] = device_properties;
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
            value = std::max(value,
                             comp_model_desc.compiled_model->get_property(ov::optimal_number_of_infer_requests.name())
                                 .as<unsigned int>());
        }
        return decltype(ov::optimal_number_of_infer_requests)::value_type{value};
    } else if (ov::execution_devices == name) {
        std::vector<std::string> device_names;
        std::set<std::string> s;
        for (const auto& comp_model_desc : m_compiled_submodels) {
            if (s.count(comp_model_desc.device) != 0)
                continue;
            s.insert(comp_model_desc.device);
            device_names.push_back(comp_model_desc.device);
        }
        return decltype(ov::execution_devices)::value_type{std::move(device_names)};
    } else if (ov::hetero::number_of_submodels == name) {
        return decltype(ov::hetero::number_of_submodels)::value_type{
            (m_compiled_submodels.size() - get_hetero_plugin()->independent_submodel_size)};
    }
    return m_cfg.get(name);
}

const std::vector<ov::Output<const ov::Node>>& ov::hetero::CompiledModel::inputs() const {
    return m_compiled_inputs;
}

const std::vector<ov::Output<const ov::Node>>& ov::hetero::CompiledModel::outputs() const {
    return m_compiled_outputs;
}

void ov::hetero::CompiledModel::set_inputs_and_outputs() {
    // Restore inputs/outputs from compiled submodels
    m_compiled_inputs.reserve(m_mapping_info._inputs_to_submodels_inputs.size());
    for (const auto& it : m_mapping_info._inputs_to_submodels_inputs) {
        const auto& submodel_idx = it.first;
        OPENVINO_ASSERT(submodel_idx < m_compiled_submodels.size(),
                        "Model contains " + std::to_string(m_compiled_submodels.size()) +
                            " submodels. Index is out of range: " + std::to_string(submodel_idx));
        const auto& compiled_submodel = m_compiled_submodels[submodel_idx].compiled_model;
        const auto& input_idx = it.second;
        OPENVINO_ASSERT(input_idx < compiled_submodel->inputs().size(),
                        "Submodel " + std::to_string(submodel_idx) + " has " +
                            std::to_string(compiled_submodel->inputs().size()) +
                            " inputs. Index is out of range: " + std::to_string(input_idx));
        m_compiled_inputs.emplace_back(compiled_submodel->inputs()[input_idx]);
    }
    m_compiled_outputs.reserve(m_mapping_info._outputs_to_submodels_outputs.size());
    for (const auto& it : m_mapping_info._outputs_to_submodels_outputs) {
        const auto& submodel_idx = it.first;
        OPENVINO_ASSERT(submodel_idx < m_compiled_submodels.size(),
                        "Model contains " + std::to_string(m_compiled_submodels.size()) +
                            " submodels. Index is out of range: " + std::to_string(submodel_idx));
        const auto& compiled_submodel = m_compiled_submodels[submodel_idx].compiled_model;
        const auto& output_idx = it.second;
        OPENVINO_ASSERT(output_idx < compiled_submodel->outputs().size(),
                        "Submodel " + std::to_string(submodel_idx) + " has " +
                            std::to_string(compiled_submodel->outputs().size()) +
                            " outputs. Index is out of range: " + std::to_string(output_idx));
        m_compiled_outputs.emplace_back(compiled_submodel->outputs()[output_idx]);
    }
}

void ov::hetero::CompiledModel::export_container(std::ostream& model_stream) const {
    pugi::xml_document doc;
    auto hetero_node = doc.append_child("hetero");
    hetero_node.append_attribute("name").set_value(m_name.c_str());
    hetero_write_metadata(hetero_node, m_mapping_info, m_compiled_submodels, m_cfg);
    std::ostringstream metadata;
    doc.save(metadata, nullptr, pugi::format_raw);

    const auto& core = get_plugin()->get_core();
    const auto metadata_text = metadata.str();
    std::vector<bool> use_ir(m_compiled_submodels.size());
    for (size_t i = 0; i < m_compiled_submodels.size(); ++i) {
        OPENVINO_ASSERT(m_compiled_submodels[i].compiled_model);
        use_ir[i] = !core->device_supports_model_caching(m_compiled_submodels[i].device);
    }

    const auto write_container = [&](std::ostream& stream) {
        const auto container_start = stream.tellp();
        while (true) {
            std::vector<HsmSubmodelPayload> submodels(m_compiled_submodels.size());
            std::optional<size_t> exporting;
            for (size_t i = 0; i < submodels.size(); ++i) {
                const auto& comp_model_desc = m_compiled_submodels[i];
                if (use_ir[i]) {
                    submodels[i].ir_model = comp_model_desc.model;
                } else {
                    submodels[i].export_blob = [&comp_model_desc, &exporting, i](std::ostream& blob_stream) {
                        exporting = i;
                        comp_model_desc.compiled_model->export_model(blob_stream);
                        exporting.reset();
                    };
                }
            }

            HsmWriterHandler handler{stream, metadata_text, std::move(submodels)};
            try {
                auto writer = ov::runtime::hsm::DeferredWriter::open(stream);
                OPENVINO_ASSERT(writer, "[HETERO] Cannot export in HSM format: the output stream must be seekable.");
                writer->add_sections({&handler});
                const auto error = writer->finalize();
                OPENVINO_ASSERT(!error, "[HETERO] HSM export failed: ", error.message());
                return;
            } catch (const ov::NotImplemented& ex) {
                OPENVINO_ASSERT(exporting, "[HETERO] Unexpected NotImplemented during HSM export: ", ex.what());
                // E.g. the Auto batch plugin reports caching but cannot export: write the container again with IR.
                use_ir[*exporting] = true;
                stream.clear();
                stream.seekp(container_start);
            }
        }
    };

    if (is_output_stream_seekable(model_stream)) {
        write_container(model_stream);
    } else {
        std::stringstream staging_stream(std::ios::in | std::ios::out | std::ios::binary);
        write_container(staging_stream);
        model_stream << staging_stream.rdbuf();
        OPENVINO_ASSERT(model_stream, "Failed to write HETERO compiled blob");
    }
}

void ov::hetero::CompiledModel::export_model(std::ostream& model_stream) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "CompiledModel::export_model");

    if (m_cfg.export_hsm_format) {
        export_container(model_stream);
        return;
    }

    pugi::xml_document doc;
    auto heteroNode = doc.append_child("hetero");
    heteroNode.append_attribute("name").set_value(m_name.c_str());
    heteroNode.append_attribute(HETERO_BLOB_FORMAT_VERSION_ATTR).set_value(HETERO_BLOB_FORMAT_VERSION);
    hetero_write_metadata(heteroNode, m_mapping_info, m_compiled_submodels, m_cfg);

    doc.save(model_stream, nullptr, pugi::format_raw);
    doc.reset();
    model_stream << std::endl;

    const bool seekable_output = is_output_stream_seekable(model_stream);

    for (const auto& comp_model_desc : m_compiled_submodels) {
        OPENVINO_ASSERT(comp_model_desc.compiled_model);
        if (get_plugin()->get_core()->device_supports_model_caching(comp_model_desc.device)) {
            if (seekable_output) {
                const auto payloadFrame = start_framed_payload(model_stream, COMPILED_BLOB_PAYLOAD);
                FramedPayloadOutputBuffer payloadBuffer(model_stream);
                std::ostream payloadStream(&payloadBuffer);
                try {
                    // Batch plugin reports property of low level plugin
                    // If we use Batch plugin inside hetero, we won't be able to call export
                    // Auto batch plugin will throw NOT_IMPLEMENTED
                    comp_model_desc.compiled_model->export_model(payloadStream);
                    OPENVINO_ASSERT(payloadStream, "Failed to export HETERO compiled blob compiled submodel payload");
                    finish_framed_payload(model_stream, payloadFrame, payloadBuffer.written_size());
                    continue;
                } catch (ov::NotImplemented&) {
                    OPENVINO_ASSERT(
                        payloadBuffer.written_size() == 0,
                        "Cannot fall back to IR serialization after partially writing a compiled submodel blob");
                    model_stream.clear();
                    model_stream.seekp(payloadFrame.frame_start_pos);
                    OPENVINO_ASSERT(model_stream,
                                    "Failed to rewind HETERO compiled blob output stream for IR fallback");
                }

                const auto irPayloadFrame = start_framed_payload(model_stream, IR_PAYLOAD);
                write_ir_payload(model_stream, comp_model_desc.model);
                finish_framed_payload(model_stream, irPayloadFrame);
                continue;
            }

            try {
                write_framed_payload_via_seekable_stream(
                    model_stream,
                    COMPILED_BLOB_PAYLOAD,
                    [&](std::ostream& payloadStream) {
                        comp_model_desc.compiled_model->export_model(payloadStream);
                    },
                    "compiled submodel payload");
                continue;
            } catch (ov::NotImplemented&) {
            }

            write_ir_framed_payload(model_stream, comp_model_desc.model);
            continue;
        }

        if (seekable_output) {
            const auto payloadFrame = start_framed_payload(model_stream, IR_PAYLOAD);
            write_ir_payload(model_stream, comp_model_desc.model);
            finish_framed_payload(model_stream, payloadFrame);
            continue;
        }

        write_ir_framed_payload(model_stream, comp_model_desc.model);
    }
}
