// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "optimize_lincache_layout.hpp"

#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../logging.hpp"
#include "../util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"

namespace {

constexpr const char* layout_optimized = "npuw_lincache_layout_optimized";

// Returns the cache_params.{past|present}.conv.N name carried by the tensor, if any.
std::optional<std::string> conv_cache_name(const ov::descriptor::Tensor& tensor, const std::string& past_or_present) {
    for (const auto& name : tensor.get_names()) {
        if (name.find(".conv.") != std::string::npos && ov::npuw::util::matchLinCacheString(name, past_or_present)) {
            return name;
        }
    }
    return std::nullopt;
}

// cache_params.past.conv.12 -> "12"
std::string layer_index(const std::string& name) {
    return name.substr(name.rfind('.') + 1);
}

bool is_rank3(const ov::PartialShape& shape) {
    return shape.rank().is_static() && shape.rank().get_length() == 3;
}

std::shared_ptr<ov::op::v0::Constant> swap_last_two_axes() {
    return ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 2, 1});
}

}  // namespace

bool ov::npuw::util::OptimizeLinCacheLayout::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (model->has_rt_info(layout_optimized) && model->get_rt_info<bool>(layout_optimized)) {
        return false;
    }

    std::map<std::string, std::shared_ptr<ov::op::v0::Parameter>> past;
    std::map<std::string, std::shared_ptr<ov::op::v0::Result>> present;

    for (const auto& param : model->get_parameters()) {
        if (const auto name = conv_cache_name(param->get_output_tensor(0), "past")) {
            past.emplace(layer_index(*name), param);
        }
    }
    for (const auto& result : model->get_results()) {
        if (const auto name = conv_cache_name(result->get_input_tensor(0), "present")) {
            present.emplace(layer_index(*name), result);
        }
    }
    if (past.empty()) {
        return false;
    }

    // The pipeline copies present -> past byte-wise, so every past state needs a present
    // counterpart and both must be the expected [batch, channels, kernel] rank-3 tensors.
    // Apply the transform to all pairs or to none to keep the copy consistent.
    for (const auto& [idx, param] : past) {
        const auto it = present.find(idx);
        if (it == present.end()) {
            LOG_DEBUG("Conv state " << idx << " has no present counterpart, lincache layout left unchanged");
            return false;
        }
        if (!is_rank3(param->get_partial_shape()) || !is_rank3(it->second->get_input_partial_shape(0))) {
            LOG_DEBUG("Conv state " << idx << " is not rank-3, lincache layout left unchanged");
            return false;
        }
    }

    // A pass-through Result can share its tensor names with a Parameter. Snapshot all
    // I/O names before reconnecting any consumers, and restore them after validation.
    std::vector<std::pair<ov::Output<ov::Node>, std::unordered_set<std::string>>> io_names;
    for (const auto& input : model->inputs()) {
        io_names.emplace_back(input, input.get_names());
    }
    for (const auto& output : model->outputs()) {
        io_names.emplace_back(output, output.get_names());
    }

    for (auto& [idx, param] : past) {
        // Parameter: [batch, channels, kernel] -> [batch, kernel, channels], then restore the
        // original layout for the consumers with a Transpose.
        auto shape = param->get_partial_shape();
        std::swap(shape[1], shape[2]);
        const auto consumers = param->output(0).get_target_inputs();
        param->set_partial_shape(shape);
        param->validate_and_infer_types();

        auto to_kernel_innermost = std::make_shared<ov::op::v1::Transpose>(param, swap_last_two_axes());
        to_kernel_innermost->set_friendly_name(param->get_friendly_name() + "/lincache_layout");
        for (auto consumer : consumers) {
            consumer.replace_source_output(to_kernel_innermost);
        }

        // Result: transpose the produced state into the new layout.
        auto result = present.at(idx);
        auto source = result->input_value(0);

        auto to_channels_innermost = std::make_shared<ov::op::v1::Transpose>(source, swap_last_two_axes());
        to_channels_innermost->set_friendly_name(result->get_friendly_name() + "/lincache_layout");
        source.set_names({});
        result->input(0).replace_source_output(to_channels_innermost);
        result->validate_and_infer_types();

        LOG_DEBUG("Conv state " << idx << " re-laid out to " << shape);
    }

    model->validate_nodes_and_infer_types();
    for (auto& [port, names] : io_names) {
        port.set_names(names);
    }
    model->set_rt_info(true, layout_optimized);
    return true;
}
