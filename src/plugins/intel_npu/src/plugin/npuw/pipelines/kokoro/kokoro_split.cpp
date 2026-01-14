// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kokoro_split.hpp"

#include <cstddef>
#include <memory>

#include "npuw/logging.hpp"
#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/util/node_util.hpp"

namespace ov::npuw {

//  Main logic for kokoro model is splitting it into two parts,
//  replacing repeat_interleave with custom processing which would be handled on host side during
//  infer request execution. Part 1 will be named "model_a", part 2 - "model_b".
KokoroSplitResult KokoroSplit::split_model(const std::shared_ptr<ov::Model>& model, const KokoroConfig& config) {
    KokoroSplitResult result;

    result.model_a = create_model_a(model, config);
    result.model_b = create_model_b(model, config);

    return result;
}

// pred_dur - our prediction of durations, which will be used to generate alignment matrix
// (we would multiply en and asr blocks with it) on host side and also the data we are going chunking by.
std::shared_ptr<ov::Node> ov::npuw::KokoroSplit::find_pred_dur_node(const std::shared_ptr<ov::Model>& model) {
    // TODO Look for pred_dur node name or Sequence Max -> Convert (?) -> Squeeze -> Result
    for (const auto& op : model->get_results()) {
        const auto& name = op->get_name();
        if (name == "pred_dur") {
            return op;
        }
        for (const auto& output_name : op->output(0).get_names()) {
            if (output_name == "pred_dur") {
                return op;
            }
        }
    }

    return nullptr;
}

// en_matmul - left hand side of matmul for encoder block (bert encoder output, contain acoustic features, "how to say")
std::shared_ptr<ov::Node> ov::npuw::KokoroSplit::find_en_matmul_node(const std::shared_ptr<ov::Model>& model) {
    // FIXME hardcoded names from kokoro model
    for (const auto& op : model->get_ops()) {
        const auto& name = op->get_friendly_name();
        if (name == "aten::matmul/MatMul") {
            return op;
        }
    }

    return nullptr;
}

// asr_matmul - left hand side of matmul for asr block (kokoro decoder input, contain text features, asr - automatic
// speech recognition, "what to say")
std::shared_ptr<ov::Node> ov::npuw::KokoroSplit::find_asr_matmul_node(const std::shared_ptr<ov::Model>& model) {
    // FIXME hardcoded names from kokoro model
    for (const auto& op : model->get_ops()) {
        const auto& name = op->get_friendly_name();
        if (name == "aten::matmul/MatMul_1") {
            return op;
        }
    }

    return nullptr;
}

std::shared_ptr<ov::Model> KokoroSplit::create_model_a(const std::shared_ptr<ov::Model>& model,
                                                       const KokoroConfig& config) {
    LOG_DEBUG("Generating Kokoro Model A");
    OPENVINO_ASSERT(model);
    auto model_a_source = model->clone();

    // For model A we need to find three outputs
    std::shared_ptr<ov::Node> pred_dur_node = find_pred_dur_node(model_a_source);
    std::shared_ptr<ov::Node> en_matmul_node = find_en_matmul_node(model_a_source);
    std::shared_ptr<ov::Node> asr_matmul_node = find_asr_matmul_node(model_a_source);

    OPENVINO_ASSERT(pred_dur_node, "pred_dur node not found");
    OPENVINO_ASSERT(en_matmul_node, "en_matmul node not found");
    OPENVINO_ASSERT(asr_matmul_node, "asr_matmul node not found");

    ov::Output<ov::Node> pred_dur_out;
    if (auto res = ov::as_type_ptr<ov::op::v0::Result>(pred_dur_node)) {
        pred_dur_out = res->input_value(0);
    } else {
        pred_dur_out = pred_dur_node->output(0);
    }

    auto en_lhs = en_matmul_node->input_value(0);
    auto asr_lhs = asr_matmul_node->input_value(0);

    ov::OutputVector outputs = {pred_dur_out, en_lhs, asr_lhs};
    auto model_a = std::make_shared<ov::Model>(outputs, model_a_source->get_parameters(), "KokoroModelA");

    model_a->validate_nodes_and_infer_types();
    model_a->output(0).get_tensor().set_names({"pred_dur"});
    model_a->output(1).get_tensor().set_names({"en_left"});
    model_a->output(2).get_tensor().set_names({"asr_left"});

    return model_a;
}

std::shared_ptr<ov::Model> ov::npuw::KokoroSplit::create_model_b(const std::shared_ptr<ov::Model>& model,
                                                                 const KokoroConfig& config) {
    LOG_DEBUG("Generating Kokoro Model B");
    OPENVINO_ASSERT(model);
    OPENVINO_ASSERT(config.block_size > 0, "Kokoro: block_size must be > 0");
    auto model_b_source = model->clone();

    std::shared_ptr<ov::Node> en_matmul_node = find_en_matmul_node(model_b_source);
    std::shared_ptr<ov::Node> asr_matmul_node = find_asr_matmul_node(model_b_source);

    OPENVINO_ASSERT(en_matmul_node, "en_matmul node not found in model B source");
    OPENVINO_ASSERT(asr_matmul_node, "asr_matmul node not found in model B source");

    // Get channels from LHS input of MatMul
    // Shape is [..., channels, time] or similar.
    auto get_channels = [](const std::shared_ptr<ov::Node>& node) -> size_t {
        auto lhs = node->input_value(0);
        auto shape = lhs.get_partial_shape();
        OPENVINO_ASSERT(shape.rank().is_static() && shape.rank().get_length() >= 2, "MatMul LHS rank must be >= 2");
        auto chan_dim = shape[shape.rank().get_length() - 2];
        OPENVINO_ASSERT(chan_dim.is_static(), "MatMul LHS channel dimension must be static");
        return chan_dim.get_length();
    };

    size_t en_channels = get_channels(en_matmul_node);
    size_t asr_channels = get_channels(asr_matmul_node);

    auto en_dtype = en_matmul_node->output(0).get_element_type();
    auto asr_dtype = asr_matmul_node->output(0).get_element_type();

    auto en_param = std::make_shared<ov::op::v0::Parameter>(
        en_dtype,
        ov::PartialShape{1,
                         static_cast<ov::Dimension::value_type>(en_channels),
                         static_cast<ov::Dimension::value_type>(config.block_size)});
    en_param->set_friendly_name("en_block");
    en_param->output(0).get_tensor().set_names({"en_block"});

    auto asr_param = std::make_shared<ov::op::v0::Parameter>(
        asr_dtype,
        ov::PartialShape{1,
                         static_cast<ov::Dimension::value_type>(asr_channels),
                         static_cast<ov::Dimension::value_type>(config.block_size)});
    asr_param->set_friendly_name("asr_block");
    asr_param->output(0).get_tensor().set_names({"asr_block"});

    // Replace MatMul outputs with parameters
    en_matmul_node->output(0).replace(en_param->output(0));
    asr_matmul_node->output(0).replace(asr_param->output(0));

    // Collect parameters
    ov::ParameterVector new_inputs;
    for (const auto& param : model_b_source->get_parameters()) {
        if (!param->output(0).get_target_inputs().empty()) {
            new_inputs.push_back(param);
        }
    }
    new_inputs.push_back(en_param);
    new_inputs.push_back(asr_param);

    auto model_b = std::make_shared<ov::Model>(model_b_source->get_results(), new_inputs, "KokoroModelB");
    // Validate nodes to propagate shapes from new parameters
    model_b->validate_nodes_and_infer_types();
    return model_b;
}

}  // namespace ov::npuw
