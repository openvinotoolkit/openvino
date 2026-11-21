// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "select_last_chunk_logits.hpp"

#include <utility>

#include "../llm_compiled_model_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/parameter.hpp"

namespace {

std::shared_ptr<ov::op::v0::Result> find_result(const std::shared_ptr<ov::Model>& model,
                                                const std::string& output_name) {
    for (const auto& result : model->get_results()) {
        if (result->output(0).get_names().count(output_name) > 0) {
            return result;
        }
    }
    return nullptr;
}

}  // namespace

ov::npuw::SelectLastLogits::SelectLastLogits(uint32_t batch_dim,
                                             std::size_t max_generation_token_len,
                                             std::string output_name)
    : m_batch_dim(batch_dim),
      m_max_generation_token_len(max_generation_token_len),
      m_output_name(std::move(output_name)) {}

bool ov::npuw::SelectLastLogits::run_on_model(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(m_batch_dim <= 1u, "Unexpected batch dimension: ", m_batch_dim);
    OPENVINO_ASSERT(m_max_generation_token_len > 0u, "Maximum generation token length must be positive.");

    const auto result = find_result(model, m_output_name);
    OPENVINO_ASSERT(result, "Missing output for logits selection: ", m_output_name);

    const auto& output_shape = result->input_value(0).get_partial_shape();
    OPENVINO_ASSERT(output_shape.rank().is_static() && output_shape.rank().get_length() == 3,
                    "Selected output must be rank-3.");

    const uint32_t sequence_dim = 1u - m_batch_dim;
    OPENVINO_ASSERT(output_shape[sequence_dim].is_static() &&
                        output_shape[sequence_dim].get_length() >= static_cast<int64_t>(m_max_generation_token_len),
                    "Selected output sequence length is smaller than maximum generation token length.");

    const auto indices =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::Shape{m_max_generation_token_len});
    indices->set_friendly_name(ov::npuw::util::kLogitsSelectionIndicesParamName);
    indices->output(0).set_names({ov::npuw::util::kLogitsSelectionIndicesParamName});
    const auto gather_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {sequence_dim});
    const auto selected = std::make_shared<ov::op::v8::Gather>(result->input_value(0), indices, gather_axis);

    result->input(0).replace_source_output(selected);
    model->add_parameters({indices});
    result->validate_and_infer_types();
    model->validate_nodes_and_infer_types();
    return true;
}
