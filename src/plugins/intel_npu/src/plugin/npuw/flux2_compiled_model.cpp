// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "flux2_compiled_model.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>

#include "intel_npu/config/npuw.hpp"
#include "logging.hpp"
#include "openvino/core/version.hpp"
#include "openvino/runtime/properties.hpp"
#include "partitioning/patterns/fold_const.hpp"
#include "serialization.hpp"

namespace {

template <typename T>
auto cfg_get(const ov::AnyMap& properties) -> typename T::ValueType {
    const auto& opt_name = std::string(T::key());
    if (properties.count(opt_name)) {
        return properties.at(opt_name).as<typename T::ValueType>();
    }
    return T::defaultValue();
}

void merge_config_with(ov::AnyMap& lhs, const ov::AnyMap& rhs) {
    for (const auto& [key, value] : rhs) {
        if (auto it = lhs.find(key); it != lhs.end()) {
            it->second = value;
        } else {
            lhs.emplace(key, value);
        }
    }
}

// Flux.2-klein pipeline submodels. Each is compiled separately (with NPUW_FLUX2:YES)
// and needs its own NPUW configuration - see submodel_config().
enum class Flux2Submodel { TEXT_ENCODER, TRANSFORMER, VAE_DECODER, VAE_ENCODER, UNKNOWN };

const char* to_cstr(Flux2Submodel role) {
    switch (role) {
    case Flux2Submodel::TEXT_ENCODER:
        return "text_encoder";
    case Flux2Submodel::TRANSFORMER:
        return "transformer";
    case Flux2Submodel::VAE_DECODER:
        return "vae_decoder";
    case Flux2Submodel::VAE_ENCODER:
        return "vae_encoder";
    default:
        return "unknown";
    }
}

std::string to_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

// Per-submodel NPUW knobs for the Flux.2-klein pipeline. NPUW_F16IC:NO keeps the
// inter-subgraph interconnect in fp32 (Flux.2-klein exports IO in f32; lowering a
// consumer's input Parameter to f16 fails at runtime with a ParameterMismatch).
ov::AnyMap submodel_config(Flux2Submodel role) {
    switch (role) {
    case Flux2Submodel::TEXT_ENCODER:
        return {
            {std::string(::intel_npu::NPUW_F16IC::key()), "NO"},
        };
    case Flux2Submodel::TRANSFORMER:
        return {
            {std::string(::intel_npu::COMPILATION_MODE_PARAMS::key()), "compute-layers-with-higher-precision=MVN"},
        };
    case Flux2Submodel::VAE_DECODER:
    case Flux2Submodel::VAE_ENCODER:
        return {
            {std::string(::intel_npu::NPUW_ONLINE_PIPELINE::key()), "NONE"},
        };
    default:
        return {};
    }
}

Flux2Submodel detect_submodel(const std::shared_ptr<ov::Model>& model) {
    // 1) Explicit signal from the model's friendly name, if the exporter set it.
    const auto name = to_lower(model->get_friendly_name());
    if (name.find("vae_encoder") != std::string::npos) {
        return Flux2Submodel::VAE_ENCODER;
    }
    if (name.find("vae_decoder") != std::string::npos) {
        return Flux2Submodel::VAE_DECODER;
    }
    if (name.find("transformer") != std::string::npos) {
        return Flux2Submodel::TRANSFORMER;
    }
    if (name.find("text_encoder") != std::string::npos) {
        return Flux2Submodel::TEXT_ENCODER;
    }

    // 2) Fall back to the input signature.
    bool has_transformer_sig = false;
    bool has_input_ids = false;
    for (const auto& p : model->get_parameters()) {
        const auto& pname = p->get_friendly_name();
        if (pname.find("encoder_hidden_states") != std::string::npos || pname.find("timestep") != std::string::npos ||
            pname.find("txt_ids") != std::string::npos) {
            has_transformer_sig = true;
        }
        if (pname.find("input_ids") != std::string::npos) {
            has_input_ids = true;
        }
    }
    if (has_transformer_sig) {
        return Flux2Submodel::TRANSFORMER;
    }
    if (has_input_ids) {
        return Flux2Submodel::TEXT_ENCODER;
    }

    // 3) VAE: a 4D conv-style tensor. 3 input channels => encoder (RGB image), else decoder.
    for (const auto& p : model->get_parameters()) {
        const auto& shape = p->get_partial_shape();
        if (shape.rank().is_static() && shape.rank().get_length() == 4) {
            const auto& channels = shape[1];
            if (channels.is_static() && channels.get_length() == 3) {
                return Flux2Submodel::VAE_ENCODER;
            }
            return Flux2Submodel::VAE_DECODER;
        }
    }
    return Flux2Submodel::UNKNOWN;
}

// Build the effective config: NPU device base < per-submodel defaults < user properties.
ov::AnyMap with_flux2_defaults(const std::shared_ptr<ov::Model>& model, const ov::AnyMap& properties) {
    const auto role = detect_submodel(model);
    if (role == Flux2Submodel::UNKNOWN) {
        LOG_WARN("Flux2CompiledModel could not identify the Flux.2-klein submodel; "
                 "applying only base defaults.");
    } else {
        LOG_INFO("Flux2CompiledModel: applying '" << to_cstr(role) << "' submodel config.");
    }

    ov::AnyMap config = {
        {std::string(::intel_npu::NPUW_DEVICES::key()), "NPU"},
        {std::string(::intel_npu::COMPILER_DYNAMIC_QUANTIZATION::key()), "YES"},
        {std::string(::intel_npu::NPUW_FOLD::key()), "YES"},
        {std::string(::intel_npu::NPUW_DQ::key()), "NO"},
        {std::string(::intel_npu::NPUW_DQ_FULL::key()), "NO"},
        {std::string(::intel_npu::NPUW_UNFOLD_IREQS::key()), "NO"},
    };
    merge_config_with(config, submodel_config(role));
    // User-provided properties take precedence over the defaults above.
    merge_config_with(config, properties);
    return config;
}

}  // namespace

ov::npuw::Flux2CompiledModel::PreparedState ov::npuw::Flux2CompiledModel::prepare(
    const std::shared_ptr<ov::Model>& model,
    const ov::AnyMap& properties) {
    auto prepared_properties = with_flux2_defaults(model, properties);
    model->set_friendly_name(model->get_friendly_name() + "_flux2");

    if (cfg_get<::intel_npu::NPUW_PLAN>(prepared_properties).empty()) {
        // Fold shape-compute chains into Constants before partitioning, but only on the
        // online path (no NPUW_PLAN). When a plan is loaded, node identities must be left
        // untouched so they still match the plan XML. Strict no-op unless the model has a
        // VariadicSplit with a non-constant split_lengths (see the function's definition).
        ov::npuw::patterns::util::foldShapeComputeChainsForConstAttrs(model);
    }

    return {model, std::move(prepared_properties)};
}

std::shared_ptr<ov::npuw::ICompiledModel> ov::npuw::Flux2CompiledModel::make_compiled_model(
    const std::shared_ptr<ov::Model>& model,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    return std::make_shared<ov::npuw::CompiledModel>(model, plugin, properties);
}

ov::npuw::Flux2CompiledModel::Flux2CompiledModel(const std::shared_ptr<ov::Model>& model,
                                                 const std::shared_ptr<const ov::IPlugin>& plugin,
                                                 const ov::AnyMap& properties,
                                                 CompiledModelFactory factory)
    : Flux2CompiledModel(prepare(model, properties), plugin, std::move(factory)) {}

ov::npuw::Flux2CompiledModel::Flux2CompiledModel(PreparedState prepared,
                                                 const std::shared_ptr<const ov::IPlugin>& plugin,
                                                 CompiledModelFactory factory)
    : ov::npuw::ICompiledModel(prepared.model, plugin),
      m_compiled_model(factory(prepared.model, plugin, prepared.properties)) {
    OPENVINO_ASSERT(m_compiled_model != nullptr, "Flux2CompiledModel requires a valid inner compiled model");
}

void ov::npuw::Flux2CompiledModel::export_model(std::ostream& stream) const {
    using namespace ov::npuw::s11n;
    write_header(stream, NPUW_FLUX2_COMPILED_MODEL_INDICATOR);
    m_compiled_model->export_model(stream);
}

std::shared_ptr<ov::npuw::ICompiledModel> ov::npuw::Flux2CompiledModel::import_model(
    std::istream& stream,
    const std::shared_ptr<const ov::IPlugin>& plugin,
    const ov::AnyMap& properties) {
    LOG_INFO("Deserializing Flux2CompiledModel...");
    LOG_BLOCK();

    using namespace ov::npuw::s11n;

    read_and_check_header(stream, NPUW_FLUX2_COMPILED_MODEL_INDICATOR, "Flux2CompiledModel");

    // The rest of the stream is the inner CompiledModel ORC blob.
    // After import it is fully self-contained; no outer Flux2 wrapper is needed
    // because the partitioning is already baked in and port mappings are consistent.
    return ov::npuw::CompiledModel::import_model(stream, plugin, properties);
}

std::shared_ptr<const ov::Model> ov::npuw::Flux2CompiledModel::get_runtime_model() const {
    return m_compiled_model->get_runtime_model();
}

void ov::npuw::Flux2CompiledModel::set_property(const ov::AnyMap& properties) {
    m_compiled_model->set_property(properties);
}

ov::Any ov::npuw::Flux2CompiledModel::get_property(const std::string& name) const {
    return m_compiled_model->get_property(name);
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::Flux2CompiledModel::create_sync_infer_request() const {
    auto self = std::static_pointer_cast<const Flux2CompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::Flux2InferRequest>(std::move(self));
}

ov::npuw::Flux2InferRequest::Flux2InferRequest(std::shared_ptr<const Flux2CompiledModel> compiled_model)
    : ov::ISyncInferRequest(compiled_model),
      m_compiled_model(std::move(compiled_model)) {}

void ov::npuw::Flux2InferRequest::ensure_inner_request_locked() const {
    if (m_inner_request == nullptr) {
        m_inner_request = m_compiled_model->m_compiled_model->create_infer_request();
        OPENVINO_ASSERT(m_inner_request != nullptr, "Flux2 infer request requires a valid inner request");
    }
}

const ov::Output<const ov::Node>& ov::npuw::Flux2InferRequest::map_port_locked(
    const ov::Output<const ov::Node>& port) const {
    ensure_inner_request_locked();

    const auto& outer_inputs = m_compiled_model->inputs();
    const auto& inner_inputs = m_inner_request->get_compiled_model()->inputs();
    for (size_t i = 0; i < outer_inputs.size(); ++i) {
        if (outer_inputs[i] == port) {
            OPENVINO_ASSERT(i < inner_inputs.size(), "Input port index is out of range in inner infer request");
            return inner_inputs[i];
        }
    }

    const auto& outer_outputs = m_compiled_model->outputs();
    const auto& inner_outputs = m_inner_request->get_compiled_model()->outputs();
    for (size_t i = 0; i < outer_outputs.size(); ++i) {
        if (outer_outputs[i] == port) {
            OPENVINO_ASSERT(i < inner_outputs.size(), "Output port index is out of range in inner infer request");
            return inner_outputs[i];
        }
    }

    OPENVINO_THROW("Unknown Flux2 infer request port: ", port.get_any_name());
}

void ov::npuw::Flux2InferRequest::infer() {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    m_inner_request->infer();
}

ov::SoPtr<ov::ITensor> ov::npuw::Flux2InferRequest::get_tensor(const ov::Output<const ov::Node>& port) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    return m_inner_request->get_tensor(map_port_locked(port));
}

void ov::npuw::Flux2InferRequest::set_tensor(const ov::Output<const ov::Node>& port,
                                             const ov::SoPtr<ov::ITensor>& tensor) {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    m_inner_request->set_tensor(map_port_locked(port), tensor);
}

void ov::npuw::Flux2InferRequest::check_tensors() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    // Trigger lazy inner request initialization; the JustInferRequest constructor
    // allocates all sub-tensors during construction, so nothing more is needed here.
    ensure_inner_request_locked();
}

std::vector<ov::SoPtr<ov::IVariableState>> ov::npuw::Flux2InferRequest::query_state() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    return m_inner_request->query_state();
}

std::vector<ov::ProfilingInfo> ov::npuw::Flux2InferRequest::get_profiling_info() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    ensure_inner_request_locked();
    return m_inner_request->get_profiling_info();
}
