// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include "infer_request.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tp_gpu/paged_attention_cache_controller.hpp"
#include "remote_context.hpp"
#include "tp_blob.hpp"

namespace ov {
namespace tp {

namespace {
ov::SoPtr<ov::IRemoteContext> make_context(const std::vector<ov::SoPtr<ov::ICompiledModel>>& rank_compiled,
                                           const std::vector<std::string>& device_names) {
    OPENVINO_ASSERT(!rank_compiled.empty(), "[TP] compiled model has no ranks");
    return {std::make_shared<RemoteContext>(device_names, rank_compiled.front()->get_context()), nullptr};
}

std::vector<ov::Output<const ov::Node>> make_inputs(const std::vector<tp_blob::Port>& ports) {
    std::vector<ov::Output<const ov::Node>> inputs;
    for (const auto& port : ports) {
        auto parameter = std::make_shared<ov::op::v0::Parameter>(port.type, port.shape);
        parameter->set_friendly_name(port.node_name);
        parameter->output(0).set_names(port.names);
        inputs.emplace_back(parameter->output(0));
    }
    return inputs;
}

std::vector<ov::Output<const ov::Node>> make_outputs(const std::vector<tp_blob::Port>& ports) {
    std::vector<ov::Output<const ov::Node>> outputs;
    for (const auto& port : ports) {
        auto producer = std::make_shared<ov::op::v0::Parameter>(port.type, port.shape);
        producer->output(0).set_names(port.names);
        auto result = std::make_shared<ov::op::v0::Result>(producer);
        result->set_friendly_name(port.node_name);
        result->output(0).set_names(port.names);
        outputs.emplace_back(result->output(0));
    }
    return outputs;
}
}  // namespace

CompiledModel::CompiledModel(const std::shared_ptr<const ov::Model>& model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             std::vector<ov::SoPtr<ov::ICompiledModel>>&& rank_compiled,
                             std::vector<std::string>&& device_names,
                             std::vector<std::string>&& sharded_state_ids,
                             bool loaded_from_cache,
                             const TPConfig& config,
                             const ImportedPorts& imported_ports)
    : ov::ICompiledModel(model, plugin, make_context(rank_compiled, device_names)),
      m_rank_compiled(std::move(rank_compiled)),
      m_device_names(std::move(device_names)),
      m_sharded_state_ids(std::move(sharded_state_ids)),
      m_imported_inputs(make_inputs(imported_ports.inputs)),
      m_imported_outputs(make_outputs(imported_ports.outputs)),
      m_loaded_from_cache(loaded_from_cache),
      m_config(config) {
    OPENVINO_ASSERT(!m_rank_compiled.empty(), "[TP] compiled model has no ranks");
    OPENVINO_ASSERT(m_rank_compiled.size() == m_device_names.size(),
                    "[TP] ", m_rank_compiled.size(), " rank models against ",
                    m_device_names.size(), " device names");
    OPENVINO_ASSERT(inputs().size() == m_rank_compiled.front()->inputs().size() &&
                        outputs().size() == m_rank_compiled.front()->outputs().size(),
                    "[TP] the model has ", inputs().size(), " inputs and ", outputs().size(),
                    " outputs, its ranks ", m_rank_compiled.front()->inputs().size(), " and ",
                    m_rank_compiled.front()->outputs().size());

    m_cache_controller = CacheController::create(m_rank_compiled);
}

const std::vector<ov::Output<const ov::Node>>& CompiledModel::inputs() const {
    const auto& own = ov::ICompiledModel::inputs();
    return own.empty() ? m_imported_inputs : own;
}

const std::vector<ov::Output<const ov::Node>>& CompiledModel::outputs() const {
    const auto& own = ov::ICompiledModel::outputs();
    return own.empty() ? m_imported_outputs : own;
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    return std::make_shared<InferRequest>(
        std::static_pointer_cast<const CompiledModel>(shared_from_this()));
}

void CompiledModel::export_model(std::ostream& stream) const {
    stream.write(tp_blob::magic, sizeof(tp_blob::magic));
    tp_blob::write_trivial<uint32_t>(stream, tp_blob::version);
    tp_blob::write_trivial<uint32_t>(stream, static_cast<uint32_t>(m_rank_compiled.size()));
    tp_blob::write_trivial<uint32_t>(stream, static_cast<uint32_t>(m_config.get_tp_strategy()));
    tp_blob::write_trivial<uint32_t>(stream, static_cast<uint32_t>(m_sharded_state_ids.size()));

    for (const auto& name : m_device_names) {
        tp_blob::write_string(stream, name);
    }

    for (const auto& id : m_sharded_state_ids) {
        tp_blob::write_string(stream, id);
    }

    for (const auto* ports : {&inputs(), &outputs()}) {
        tp_blob::write_trivial<uint32_t>(stream, static_cast<uint32_t>(ports->size()));
        for (const auto& port : *ports) {
            tp_blob::write_port(stream, port);
        }
    }

    for (const auto& rank : m_rank_compiled) {
        // The rank blob length is only known once the device plugin has written
        // it, so reserve the slot and patch it afterwards. Import needs the
        // length to find the next rank instead of trusting the rank reader to
        // stop exactly on the blob boundary.
        const auto length_pos = stream.tellp();
        OPENVINO_ASSERT(length_pos != std::ostream::pos_type(-1),
                        "[TP] export_model needs a seekable stream to record per-rank blob sizes");
        tp_blob::write_trivial<uint64_t>(stream, uint64_t{0});

        const auto blob_start = stream.tellp();
        rank->export_model(stream);
        const auto blob_end = stream.tellp();

        stream.seekp(length_pos);
        tp_blob::write_trivial<uint64_t>(stream, static_cast<uint64_t>(blob_end - blob_start));
        stream.seekp(blob_end);
        OPENVINO_ASSERT(stream.good(), "[TP] export_model failed while writing rank blobs");
    }
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    OPENVINO_ASSERT(!m_rank_compiled.empty(),
                    "[TP] No compiled models");
    return m_rank_compiled[0]->get_runtime_model();
}

void CompiledModel::set_property(const ov::AnyMap&) {
    OPENVINO_NOT_IMPLEMENTED;
}

void CompiledModel::release_memory() {
    // Whatever caches the rank models hold live in their own plugins;
    // TP wrapper owns nothing releasable of its own.
    for (const auto& rank : m_rank_compiled) {
        rank->release_memory();
    }
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    if (name == ov::supported_properties.name()) {
        std::vector<ov::PropertyName> properties {
            ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::loaded_from_cache.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::execution_devices.name(), ov::PropertyMutability::RO},
        };
        if (m_cache_controller) {
            properties.emplace_back(ov::tp_gpu::paged_attention_cache_controller.name(), ov::PropertyMutability::RO);
        }

        const auto rank_properties = m_rank_compiled.front()->get_property(name);
        for (const auto& property : rank_properties.as<std::vector<ov::PropertyName>>()) {
            const auto known = std::find(properties.begin(), properties.end(), property);
            if (known == properties.end()) {
                properties.emplace_back(property, ov::PropertyMutability::RO);
            }
        }
        return properties;
    }

    if (name == ov::loaded_from_cache.name()) {
        return m_loaded_from_cache;
    }

    if (name == ov::execution_devices.name()) {
        return m_device_names;
    }

    if (name == ov::tp::internal::world_size.name()) {
        return static_cast<int>(m_rank_compiled.size());
    }

    if (name == ov::internal::supported_properties.name()) {
        return std::vector<ov::PropertyName>{
            ov::PropertyName{ov::tp::internal::world_size.name(), ov::PropertyMutability::RO}};
    }

    if (name == ov::tp_gpu::paged_attention_cache_controller.name()) {
        // Absent on models that do not use a paged-attention cache: a caller
        // finding nothing here has to keep allocating the cache itself.
        OPENVINO_ASSERT(m_cache_controller,
                        "[TP] This model has no paged-attention cache to hand over");
        return std::static_pointer_cast<ov::tp_gpu::IPagedAttentionCacheController>(m_cache_controller);
    }

    return m_rank_compiled.front()->get_property(name);
}

}  // namespace tp
}  // namespace ov
