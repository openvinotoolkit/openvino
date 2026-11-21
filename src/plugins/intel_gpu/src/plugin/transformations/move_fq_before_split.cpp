// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "move_fq_before_split.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

namespace {

using FakeQuantizes = std::vector<std::shared_ptr<ov::op::v0::FakeQuantize>>;

std::shared_ptr<ov::op::v0::FakeQuantize> get_single_fq_consumer(const ov::Output<ov::Node>& output) {
    const auto consumers = output.get_target_inputs();
    if (consumers.size() != 1 || consumers.begin()->get_index() != 0) {
        return nullptr;
    }
    return ov::as_type_ptr<ov::op::v0::FakeQuantize>(consumers.begin()->get_node()->shared_from_this());
}

bool has_per_tensor_constant_ranges(const std::shared_ptr<ov::op::v0::FakeQuantize>& fq) {
    for (size_t i = 1; i < fq->get_input_size(); ++i) {
        const auto range = ov::as_type_ptr<ov::op::v0::Constant>(fq->get_input_node_shared_ptr(i));
        if (!range || ov::shape_size(range->get_shape()) != 1) {
            return false;
        }
    }
    return true;
}

bool feeds_result(const std::shared_ptr<ov::op::v0::FakeQuantize>& fq) {
    const auto consumers = fq->output(0).get_target_inputs();
    return std::any_of(consumers.begin(), consumers.end(), [](const ov::Input<ov::Node>& consumer) {
        return ov::is_type<ov::op::v0::Result>(consumer.get_node());
    });
}

// Empty unless every split output feeds an identical per-tensor FakeQuantize that can move to the split input.
FakeQuantizes get_movable_fake_quantizes(const std::shared_ptr<ov::Node>& split) {
    FakeQuantizes fqs;
    for (const auto& output : split->outputs()) {
        auto fq = get_single_fq_consumer(output);
        if (!fq || feeds_result(fq)) {
            return {};
        }
        fqs.push_back(fq);
    }

    const auto& first = fqs.front();
    if (!has_per_tensor_constant_ranges(first)) {
        return {};
    }
    for (size_t i = 1; i < fqs.size(); ++i) {
        if (!ov::op::util::have_same_fake_quantize_params(first, fqs[i])) {
            return {};
        }
    }
    return fqs;
}

}  // namespace

bool MoveFakeQuantizeBeforeSplit::run_on_model(const std::shared_ptr<ov::Model>& model) {
    bool rewritten = false;
    for (const auto& split : model->get_ordered_ops()) {
        if (!ov::is_type_any_of<ov::op::v1::Split, ov::op::v1::VariadicSplit>(split) || transformation_callback(split)) {
            continue;
        }

        const auto fqs = get_movable_fake_quantizes(split);
        if (fqs.empty()) {
            continue;
        }

        const auto& first = fqs.front();
        ov::OutputVector fq_inputs{split->input_value(0)};
        for (size_t i = 1; i < first->get_input_size(); ++i) {
            fq_inputs.push_back(first->input_value(i));
        }
        const auto new_fq = first->clone_with_new_inputs(fq_inputs);
        new_fq->set_friendly_name(split->get_friendly_name() + "/fq_input_0");
        ov::copy_runtime_info(ov::NodeVector(fqs.begin(), fqs.end()), new_fq);
        split->input(0).replace_source_output(new_fq->output(0));

        for (size_t i = 0; i < fqs.size(); ++i) {
            fqs[i]->output(0).replace(split->output(i));
        }
        rewritten = true;
    }
    return rewritten;
}

}  // namespace ov::intel_gpu
