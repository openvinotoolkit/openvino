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
#include "npuw/util.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_update.hpp"
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

// index_copy is exported as ScatterElementsUpdate(data, Broadcast(index[K], axes_mapping = {0}), updates, axis = 0):
// a copy of whole rows, which ScatterUpdate does row-wise instead of element-wise. Returns the rewritten count
std::size_t convert_index_copy_to_scatter_update(const std::shared_ptr<ov::Model>& model) {
    std::size_t count = 0;
    for (const auto& node : model->get_ordered_ops()) {
        const auto seu = ov::as_type_ptr<ov::op::v12::ScatterElementsUpdate>(node);
        if (!seu || seu->get_reduction() != ov::op::v12::ScatterElementsUpdate::Reduction::NONE) {
            continue;
        }
        const auto axis = ov::as_type_ptr<ov::op::v0::Constant>(seu->get_input_node_shared_ptr(3));
        const auto bcast = ov::as_type_ptr<ov::op::v1::Broadcast>(seu->get_input_node_shared_ptr(1));
        if (!axis || axis->cast_vector<int64_t>() != std::vector<int64_t>{0} || !bcast ||
            bcast->get_broadcast_spec().m_type != ov::op::AutoBroadcastType::NONE) {
            continue;
        }
        const auto axes = ov::as_type_ptr<ov::op::v0::Constant>(bcast->get_input_node_shared_ptr(2));
        const auto index = bcast->input_value(0);
        const auto& data_shape = seu->get_input_partial_shape(0);
        if (!axes || axes->cast_vector<int64_t>() != std::vector<int64_t>{0} || index.get_partial_shape().rank() != 1 ||
            data_shape.rank().is_dynamic() || data_shape.size() < 2) {
            continue;
        }

        // Rows are flattened to 2D so every rank maps to the same [N, M] row scatter
        std::vector<int64_t> row_shape{0};
        std::vector<int64_t> full_shape{0};
        int64_t row = 1;
        for (std::size_t d = 1; d < data_shape.size(); ++d) {
            if (data_shape[d].is_dynamic()) {
                row = -1;
                break;
            }
            row *= data_shape[d].get_length();
            full_shape.push_back(data_shape[d].get_length());
        }
        if (row < 0) {
            continue;
        }
        row_shape.push_back(row);
        const auto flat = [&](const ov::Output<ov::Node>& value, const std::vector<int64_t>& shape) {
            const auto target = ov::op::v0::Constant::create(ov::element::i64, {shape.size()}, shape);
            return std::make_shared<ov::op::v1::Reshape>(value, target, true)->output(0);
        };
        const auto zero = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
        const auto scatter = std::make_shared<ov::op::v3::ScatterUpdate>(flat(seu->input_value(0), row_shape),
                                                                         index,
                                                                         flat(seu->input_value(2), row_shape),
                                                                         zero);
        const auto result = flat(scatter, full_shape).get_node_shared_ptr();
        result->set_friendly_name(seu->get_friendly_name());
        ov::copy_runtime_info(seu, {scatter, result});
        ov::replace_node(seu, result);
        ++count;
    }
    return count;
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
    // Variants share one weights bank so the language-model weights are kept once; caller properties win
    const ov::AnyMap defaults = m_mode == Mode::LANGUAGE_MODEL
                                    ? ov::AnyMap{{"NPUW_DEVICES", "NPU"},
                                                 {"NPUW_FOLD", "YES"},
                                                 {"NPUW_FUNCALL_FOR_ALL", "YES"},
                                                 {"NPUW_FUNCALL_ASYNC", "YES"},
                                                 {"NPUW_DCOFF_TYPE", "f16"},
                                                 {"NPUW_DCOFF_SCALE", "YES"},
                                                 {"NPUW_WEIGHTS_BANK", ov::npuw::util::generate_random_string()}}
                                    : ov::AnyMap{{"NPUW_ONLINE_PIPELINE", "NONE"}};
    m_inner_props.insert(defaults.begin(), defaults.end());
    m_cfg.parseEnvVars();
    m_cfg.update(to_string_map(bagel_props));
    m_seq_buckets = parse_buckets(m_cfg.get<::intel_npu::NPUW_BAGEL_SEQ_BUCKETS>(), "NPUW_BAGEL_SEQ_BUCKETS");
    m_kv_buckets = parse_buckets(m_cfg.get<::intel_npu::NPUW_BAGEL_KV_BUCKETS>(), "NPUW_BAGEL_KV_BUCKETS");
    if (m_mode == Mode::LANGUAGE_MODEL) {
        m_model = m_model->clone();
        LOG_INFO("BagelCompiledModel: converted " << convert_index_copy_to_scatter_update(m_model)
                                                  << " index_copy ScatterElementsUpdate ops to ScatterUpdate");
    }

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
    const std::string& tag,
    bool drop_logits) const {
    auto model = m_model->clone();
    model->reshape(shapes);
    if (drop_logits) {
        const auto results = model->get_results();
        const auto logits = std::find_if(results.begin(), results.end(), [](const auto& result) {
            return result->input_value(0).get_names().count("logits") > 0;
        });
        if (logits != results.end()) {
            model->remove_result(*logits);
        }
    }
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
    // Only decode-sized calls read logits; for longer sequences the LM head output costs gigabytes
    auto compiled = compile_variant(shapes, tag.str(), key.seq > m_seq_buckets.front());
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
    auto compiled = compile_variant(shapes, tag.str(), false);
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
