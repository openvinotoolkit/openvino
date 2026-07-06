// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "model_builder_types.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/result.hpp"

namespace ov {
namespace test {
namespace npuw {

/// rt_info key GenAI stamps on the Eagle3 "last_hidden_state" Results it adds.
/// NPUW's LM-head cut skips Results carrying it (see llm_compiled_model.cpp).
inline constexpr const char* kManuallyAddedOutput = "manually_added_output";

/// Result carrying the "manually_added_output" rt_info marker, mirroring
/// GenAI's transform_hidden_state (eagle3_model_transforms.cpp).
std::shared_ptr<ov::op::v0::Result> make_manually_added_result(const ov::Output<ov::Node>& value,
                                                               const std::string& name);

/// Eagle3 target-side hidden state capture. Concatenates the captured layer
/// outputs on the last axis ("eagle3_hidden_states_concat") and, when fc_weight
/// is truthy, projects the concat back to hidden_size ("eagle3_hidden_state_fc")
/// the way GenAI's move_fc_from_draft_to_main rehomes the draft's fc into the
/// target model for the NPU pipeline. An empty fc_weight returns the raw concat
/// (continuous-batching pipeline form).
ov::Output<ov::Node> make_eagle3_hidden_capture(const ov::OutputVector& captured_layers,
                                                size_t hidden_size,
                                                ov::element::Type precision,
                                                const WeightFn& fc_weight);

/// Draft-to-target vocab mapping table: i64 Constant [draft_vocab_size] holding
/// per-token id offsets (target_id = draft_id + d2t[draft_id]), as in the raw
/// Eagle3 draft export's "d2t" output. GenAI extracts and removes it before the
/// model reaches NPUW; emit it only to model the raw export form.
ov::Output<ov::Node> make_eagle3_d2t(size_t draft_vocab_size, size_t target_vocab_size);

}  // namespace npuw
}  // namespace test
}  // namespace ov
