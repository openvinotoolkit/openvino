// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/op/parameter.hpp"
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

inline std::shared_ptr<ov::op::v0::Parameter> get_or_add_named_parameter(
    std::map<std::string, std::shared_ptr<ov::op::v0::Parameter>>& created_params,
    const std::string& name,
    const ov::element::Type& element_type,
    const ov::PartialShape& shape) {
    auto it = created_params.find(name);
    if (it != created_params.end()) {
        const auto& existing = it->second;
        OPENVINO_ASSERT(existing->get_element_type() == element_type,
                        "Existing parameter element type mismatch for '",
                        name,
                        "'.");
        OPENVINO_ASSERT(existing->get_partial_shape() == shape, "Existing parameter shape mismatch for '", name, "'.");
        return existing;
    }
    auto param = std::make_shared<ov::op::v0::Parameter>(element_type, shape);
    param->set_friendly_name(name);
    OPENVINO_ASSERT(param->get_output_size() == 1);
    param->get_output_tensor(0).set_names({name});
    created_params.emplace(name, param);
    return param;
}

inline std::shared_ptr<ov::op::v0::Parameter> get_or_add_named_parameter(
    std::map<std::string, std::shared_ptr<ov::op::v0::Parameter>>& created_params,
    const std::string& name) {
    const auto it = created_params.find(name);
    OPENVINO_ASSERT(it != created_params.end(), "Missing model parameter: ", name);
    return it->second;
}
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
    paged_attention::Options m_options;
};
}  // namespace pass
}  // namespace ov
