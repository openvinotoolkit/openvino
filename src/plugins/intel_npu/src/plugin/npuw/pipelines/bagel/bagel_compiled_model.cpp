// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bagel_compiled_model.hpp"

#include <algorithm>
#include <sstream>

#include "bagel_infer_request.hpp"
#include "intel_npu/config/npuw.hpp"
#include "npuw/llm_compiled_model_utils.hpp"
#include "npuw/logging.hpp"
#include "openvino/runtime/properties.hpp"

namespace {

bool is_bagel_key(const std::string& key) {
    return key.rfind("NPUW_BAGEL", 0) == 0;
}

std::map<std::string, std::string> to_string_map(const ov::AnyMap& params) {
    std::map<std::string, std::string> result;
    for (const auto& [key, value] : params) {
        if (value.is<bool>()) {
            result.emplace(key, value.as<bool>() ? "YES" : "NO");
        } else {
            result.emplace(key, value.as<std::string>());
        }
    }
    return result;
}

std::vector<std::size_t> parse_buckets(const std::string& csv, const char* option) {
    std::vector<std::size_t> buckets;
    std::stringstream ss(csv);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            buckets.push_back(std::stoull(item));
        }
    }
    OPENVINO_ASSERT(!buckets.empty(), option, " must list at least one size, got '", csv, "'");
    std::sort(buckets.begin(), buckets.end());
    return buckets;
}

std::size_t pick_bucket(const std::vector<std::size_t>& buckets, std::size_t n, const char* what) {
    const auto it = std::lower_bound(buckets.begin(), buckets.end(), n);
    OPENVINO_ASSERT(it != buckets.end(), "NPUW_BAGEL: ", what, " ", n, " exceeds the largest bucket ", buckets.back());
    return *it;
}

}  // namespace

ov::npuw::BagelCompiledModel::BagelCompiledModel(const std::shared_ptr<ov::Model>& model,
                                                 const std::shared_ptr<const ov::IPlugin>& plugin,
                                                 const ov::AnyMap& properties)
    : ov::npuw::ICompiledModel(model, plugin),
      m_model(model),
      m_core(plugin->get_core()),
      m_mode(util::has_input(model, "und_index") && util::has_input(model, "gen_index") ? Mode::LANGUAGE_MODEL
                                                                                        : Mode::PER_SHAPE),
      m_options_desc(std::make_shared<::intel_npu::OptionsDesc>()),
      m_cfg(m_options_desc) {
    ::intel_npu::registerNPUWBagelOptions(*m_options_desc);

    ov::AnyMap bagel_props;
    for (const auto& [key, value] : properties) {
        (is_bagel_key(key) ? bagel_props : m_inner_props).emplace(key, value);
    }
    m_cfg.parseEnvVars();
    m_cfg.update(to_string_map(bagel_props));
    m_seq_buckets = parse_buckets(m_cfg.get<::intel_npu::NPUW_BAGEL_SEQ_BUCKETS>(), "NPUW_BAGEL_SEQ_BUCKETS");
    m_kv_buckets = parse_buckets(m_cfg.get<::intel_npu::NPUW_BAGEL_KV_BUCKETS>(), "NPUW_BAGEL_KV_BUCKETS");

    LOG_INFO("BagelCompiledModel: " << model->get_friendly_name() << " in "
                                    << (m_mode == Mode::LANGUAGE_MODEL ? "language-model" : "per-shape")
                                    << " mode; variants are compiled on first use.");
}

std::size_t ov::npuw::BagelCompiledModel::seq_bucket(std::size_t n) const {
    return pick_bucket(m_seq_buckets, n, "sequence length");
}

std::size_t ov::npuw::BagelCompiledModel::kv_bucket(std::size_t n) const {
    return pick_bucket(m_kv_buckets, n, "KV length");
}

ov::SoPtr<ov::ICompiledModel> ov::npuw::BagelCompiledModel::compile_variant(
    const std::map<std::size_t, ov::PartialShape>& shapes,
    const std::string& tag) const {
    auto model = m_model->clone();
    model->reshape(shapes);
    LOG_INFO("BagelCompiledModel: compiling " << m_model->get_friendly_name() << " variant " << tag);
    return ov::SoPtr<ov::ICompiledModel>(ov::npuw::ICompiledModel::create(model, get_plugin(), m_inner_props), nullptr);
}

ov::SoPtr<ov::ICompiledModel> ov::npuw::BagelCompiledModel::lm_variant(const LMKey& key) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (auto it = m_lm_variants.find(key); it != m_lm_variants.end()) {
        return it->second;
    }
    const auto S = static_cast<int64_t>(key.seq);
    const auto G = static_cast<int64_t>(key.gen);
    const auto P = static_cast<int64_t>(key.kv);

    std::map<std::size_t, ov::PartialShape> shapes;
    const auto& inputs = m_model->inputs();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto& in = inputs[i];
        const auto& name = in.get_any_name();
        auto shape = in.get_partial_shape();
        if (name == "inputs_embeds") {
            shape[0] = S;
        } else if (name == "und_index") {
            shape[0] = S - G;
        } else if (name == "gen_index" || name == "x_t" || name == "vae_position_ids") {
            shape[0] = G;
        } else if (name == "position_ids") {
            shape[0] = S;
        } else if (name == "attention_mask") {
            shape[2] = S;
            shape[3] = P + S;
        } else if (name.rfind("past_key_values.", 0) == 0) {
            shape[2] = P;
        }
        OPENVINO_ASSERT(shape.is_static(), "NPUW_BAGEL: no static shape rule for input ", name, " ", shape);
        shapes.emplace(i, shape);
    }

    std::stringstream tag;
    tag << "S=" << S << " n_gen=" << G << " P=" << P;
    auto compiled = compile_variant(shapes, tag.str());
    m_lm_variants.emplace(key, compiled);
    return compiled;
}

ov::SoPtr<ov::ICompiledModel> ov::npuw::BagelCompiledModel::shape_variant(const ShapeKey& key) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (auto it = m_shape_variants.find(key); it != m_shape_variants.end()) {
        return it->second;
    }
    const auto& inputs = m_model->inputs();
    OPENVINO_ASSERT(key.size() == inputs.size(), "NPUW_BAGEL: expected ", inputs.size(), " input shapes");

    std::map<std::size_t, ov::PartialShape> shapes;
    std::stringstream tag;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        shapes.emplace(i, ov::PartialShape(key[i]));
        tag << (i ? " " : "") << key[i];
    }
    auto compiled = compile_variant(shapes, tag.str());
    m_shape_variants.emplace(key, compiled);
    return compiled;
}

std::shared_ptr<ov::ISyncInferRequest> ov::npuw::BagelCompiledModel::create_sync_infer_request() const {
    auto self = std::static_pointer_cast<const BagelCompiledModel>(shared_from_this());
    return std::make_shared<ov::npuw::BagelInferRequest>(std::move(self));
}

void ov::npuw::BagelCompiledModel::export_model(std::ostream&) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ov::Model> ov::npuw::BagelCompiledModel::get_runtime_model() const {
    return m_model;
}

void ov::npuw::BagelCompiledModel::set_property(const ov::AnyMap& properties) {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto& [key, value] : properties) {
        m_inner_props[key] = value;
    }
    for (auto& item : m_lm_variants) {
        item.second->set_property(properties);
    }
    for (auto& item : m_shape_variants) {
        item.second->set_property(properties);
    }
}

ov::Any ov::npuw::BagelCompiledModel::get_property(const std::string& name) const {
    if (name == ov::model_name.name()) {
        return m_model->get_friendly_name();
    }
    if (name == ov::supported_properties.name()) {
        return std::vector<ov::PropertyName>{ov::model_name, ov::supported_properties};
    }
    std::lock_guard<std::mutex> lock(m_mutex);
    if (auto it = m_inner_props.find(name); it != m_inner_props.end()) {
        return it->second;
    }
    OPENVINO_THROW("NPUW_BAGEL: unsupported property ", name);
}
