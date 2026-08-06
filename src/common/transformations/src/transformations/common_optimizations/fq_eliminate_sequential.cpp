// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fq_eliminate_sequential.hpp"

#include <memory>
#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/util/squeeze_base.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace op_util = ov::op::util;

namespace ov::pass {

namespace {

bool is_value_preserving(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type_any_of<v1::Reshape, v1::Transpose, op_util::SqueezeBase, v0::Unsqueeze>(node);
}

}  // namespace

bool FakeQuantizeEliminateSequential::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(FakeQuantizeEliminateSequential);

    bool eliminated = false;
    for (const auto& op : model->get_ordered_ops()) {
        auto fq1 = ov::as_type_ptr<v0::FakeQuantize>(op);
        if (!fq1) {
            continue;
        }

        // Walk forward from FQ1 following its consumers and collect the redundant FakeQuantizes.
        // Traversal continues only through FQ1, value-preserving ops, and redundant FakeQuantizes, so
        // branches into several consumers are handled as well. visit_path_forward queries a node's
        // consumers right after the visitor runs, so the redundant FakeQuantizes are only collected
        // here and detached afterwards to avoid dereferencing a freed node.
        std::vector<std::shared_ptr<v0::FakeQuantize>> redundant_fqs;
        std::unordered_set<ov::Node*> visited;
        auto skip_node = [&](ov::Node* node) {
            auto shared_node = node->shared_from_this();
            if (shared_node == fq1 || is_value_preserving(shared_node)) {
                return false;
            }
            return !op_util::have_same_fake_quantize_params(fq1, ov::as_type_ptr<v0::FakeQuantize>(shared_node));
        };
        auto collect_redundant_fq = [&](ov::Node* node) {
            if (auto fq2 = ov::as_type_ptr<v0::FakeQuantize>(node->shared_from_this()); fq2 && fq2 != fq1) {
                redundant_fqs.push_back(fq2);
            }
        };
        op_util::visit_path_forward(fq1.get(), visited, collect_redundant_fq, skip_node);

        for (const auto& fq2 : redundant_fqs) {
            eliminated = replace_output_update_name(fq2->output(0), fq2->input_value(0)) || eliminated;
        }
    }

    return eliminated;
}

}  // namespace ov::pass
