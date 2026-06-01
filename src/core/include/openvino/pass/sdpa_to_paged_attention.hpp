// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {
namespace paged_attention {

struct Options {
    bool use_per_layer_block_indices_inputs;
    bool use_score_outputs;
    bool allow_score_aggregation;
    bool allow_cache_rotation;
    bool allow_xattention;
    bool allow_adaptive_rkv;
    bool allow_qq_bias;
};

template <typename NodeT, typename VectorT>
class NamedNodeStore {
public:
    NamedNodeStore() = default;

    explicit NamedNodeStore(VectorT current_nodes) : m_current_nodes(std::move(current_nodes)) {}

    std::shared_ptr<NodeT> get(const std::string& name) const {
        return find(name);
    }

    std::shared_ptr<NodeT> operator[](const std::string& name) const {
        auto node = find(name);
        OPENVINO_ASSERT(node, "Missing model node: ", name);
        return node;
    }

    bool remove(const std::string& name) {
        auto node = find(name);
        if (!node) {
            return false;
        }
        m_nodes.erase(std::remove(m_nodes.begin(), m_nodes.end(), node), m_nodes.end());
        return true;
    }

    VectorT& items() {
        return m_nodes;
    }

    const VectorT& items() const {
        return m_nodes;
    }

    std::shared_ptr<NodeT> find(const std::string& name) const {
        auto find_in = [&](const VectorT& nodes) -> std::shared_ptr<NodeT> {
            auto it = std::find_if(nodes.begin(), nodes.end(), [&](const std::shared_ptr<NodeT>& node) {
                return node && node->get_friendly_name() == name;
            });
            return it == nodes.end() ? nullptr : *it;
        };

        if (auto current = find_in(m_current_nodes)) {
            return current;
        }

        return find_in(m_nodes);
    }

protected:
    VectorT m_nodes;
    VectorT m_current_nodes;
};

struct PaParams : public NamedNodeStore<ov::op::v0::Parameter, ov::ParameterVector> {
    PaParams() = default;

    explicit PaParams(ov::ParameterVector current_params)
        : NamedNodeStore<ov::op::v0::Parameter, ov::ParameterVector>(std::move(current_params)) {}

    std::shared_ptr<ov::op::v0::Parameter> add(const std::string& name,
                                               const ov::element::Type& element_type,
                                               const ov::PartialShape& shape) {
        auto existing = find(name);
        if (existing) {
            OPENVINO_ASSERT(existing->get_element_type() == element_type,
                            "Existing parameter element type mismatch for '",
                            name,
                            "'.");
            OPENVINO_ASSERT(existing->get_partial_shape() == shape,
                            "Existing parameter shape mismatch for '",
                            name,
                            "'.");
            return existing;
        }
        auto param = std::make_shared<ov::op::v0::Parameter>(element_type, shape);
        param->set_friendly_name(name);
        OPENVINO_ASSERT(param->get_output_size() == 1);
        param->get_output_tensor(0).set_names({name});
        m_nodes.push_back(param);
        return param;
    }
};

struct PaResults : public NamedNodeStore<ov::op::v0::Result, ov::ResultVector> {
    PaResults() = default;

    explicit PaResults(ov::ResultVector current_results)
        : NamedNodeStore<ov::op::v0::Result, ov::ResultVector>(std::move(current_results)) {}

    std::shared_ptr<ov::op::v0::Result> add(const std::string& name, const ov::Output<ov::Node>& output) {
        auto existing = find(name);
        if (existing) {
            OPENVINO_ASSERT(existing->get_output_size() == 1,
                            "Result '",
                            name,
                            "' is expected to have a single output.");
            const auto& names = existing->get_output_tensor(0).get_names();
            OPENVINO_ASSERT(names.count(name) != 0, "Result '", name, "' does not contain the expected tensor name.");
            return existing;
        }
        auto result = std::make_shared<ov::op::v0::Result>(output);
        result->set_friendly_name(name);
        OPENVINO_ASSERT(result->get_output_size() == 1);
        result->get_output_tensor(0).set_names({name});
        m_nodes.push_back(result);
        return result;
    }
};
}  // namespace paged_attention
/**
 * @brief The transformation replaces KV-cache processing part in LLMs by PagedAttention operation.
 * NOTE:
 * The transformation may throw an exception when some configuration of the model failed:
 * i.e. the SDPA node is absent in the model. This means the graph cannot be processed for the PA scenario,
 * so the GenAI pipeline (the only pipeline the transformation is used in so far) will fallback to the SDPA
 * implementaion and run inference using it.
 * \ingroup ov_pass_cpp_api
 */
class OPENVINO_API SDPAToPagedAttention : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SDPAToPagedAttention");

    explicit SDPAToPagedAttention(bool use_per_layer_block_indices_inputs = false,
                                  bool use_score_outputs = false,
                                  bool allow_score_aggregation = false,
                                  bool allow_cache_rotation = false,
                                  bool allow_xattention = false,
                                  bool allow_adaptive_rkv = false,
                                  bool allow_qq_bias = false);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    paged_attention::PaParams m_params;
    paged_attention::PaResults m_results;
    paged_attention::Options m_options;
};
}  // namespace pass
}  // namespace ov
