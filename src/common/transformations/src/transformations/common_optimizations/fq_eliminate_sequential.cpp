// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fq_eliminate_sequential.hpp"

#include <algorithm>
#include <cstring>
#include <memory>
#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
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

constexpr size_t input_low_idx = 1;
constexpr size_t input_high_idx = 2;
constexpr size_t output_low_idx = 3;
constexpr size_t output_high_idx = 4;
constexpr size_t max_checked_values = size_t{1} << 20;

bool is_value_preserving(const std::shared_ptr<ov::Node>& node) {
    return ov::is_type_any_of<v1::Reshape, v1::Transpose, op_util::SqueezeBase, v0::Unsqueeze>(node);
}

bool have_same_range(const std::shared_ptr<v0::FakeQuantize>& fq, size_t lhs, size_t rhs) {
    return fq->get_input_partial_shape(lhs) == fq->get_input_partial_shape(rhs) &&
           ov::compare_constants(fq->get_input_node_shared_ptr(lhs), fq->get_input_node_shared_ptr(rhs));
}

bool has_same_input_and_output_range(const std::shared_ptr<v0::FakeQuantize>& fq) {
    return have_same_range(fq, input_low_idx, output_low_idx) && have_same_range(fq, input_high_idx, output_high_idx);
}

std::shared_ptr<v0::Constant> get_range(const std::shared_ptr<v0::FakeQuantize>& fq, size_t index) {
    return ov::as_type_ptr<v0::Constant>(fq->get_input_node_shared_ptr(index));
}

std::shared_ptr<v0::Constant> to_f32(const std::shared_ptr<v0::Constant>& constant) {
    return std::make_shared<v0::Constant>(ov::element::f32, constant->get_shape(), constant->cast_vector<float>());
}

bool is_idempotent(const std::shared_ptr<v0::FakeQuantize>& fq) {
    const auto& type = fq->get_output_element_type(0);
    if (!type.is_real() || !fq->has_evaluate() || fq->get_auto_broadcast().m_type != ov::op::AutoBroadcastType::NUMPY) {
        return false;
    }
    const size_t levels = fq->get_levels();
    if (levels < 2) {
        return false;
    }

    ov::TensorVector ranges;
    ov::PartialShape range_shape = ov::PartialShape{};
    for (size_t index = input_low_idx; index <= output_high_idx; ++index) {
        const auto range = get_range(fq, index);
        if (!range || !ov::PartialShape::broadcast_merge_into(range_shape,
                                                              range->get_shape(),
                                                              ov::op::AutoBroadcastType::NUMPY)) {
            return false;
        }
        ranges.push_back(range->get_tensor_view());
    }
    const auto range_size = ov::shape_size(range_shape.to_shape());
    const size_t rows = levels + 2;
    if (range_size == 0 || rows > max_checked_values / range_size) {
        return false;
    }

    auto values_shape = range_shape.to_shape();
    values_shape.insert(values_shape.begin(), rows);
    std::vector<float> grid(rows * range_size);
    for (size_t row = 0; row < rows; ++row) {
        float value = static_cast<float>(row);
        if (row == 0) {
            value = 0.25f;
        } else if (row == levels) {
            value = -1.0f;
        }
        std::fill_n(grid.begin() + row * range_size, range_size, value);
    }
    auto grid_input = v0::Constant::create(ov::element::f32, values_shape, grid);
    auto grid_low = v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto grid_high = v0::Constant::create(ov::element::f32, ov::Shape{}, {static_cast<float>(levels - 1)});
    auto output_low = to_f32(get_range(fq, output_low_idx));
    auto output_high = to_f32(get_range(fq, output_high_idx));
    auto grid_fq = std::make_shared<v0::FakeQuantize>(grid_input, grid_low, grid_high, output_low, output_high, levels);
    ov::TensorVector grid_outputs{ov::Tensor(ov::element::f32, values_shape)};
    if (!grid_fq->evaluate(grid_outputs,
                           {grid_input->get_tensor_view(),
                            grid_low->get_tensor_view(),
                            grid_high->get_tensor_view(),
                            output_low->get_tensor_view(),
                            output_high->get_tensor_view()})) {
        return false;
    }

    ov::TensorVector fq_outputs{ov::Tensor(type, values_shape)};
    if (type == ov::element::f32) {
        fq_outputs = grid_outputs;
    } else if (!v0::Convert().evaluate(fq_outputs, grid_outputs)) {
        return false;
    }

    ov::TensorVector inputs{fq_outputs[0]};
    inputs.insert(inputs.end(), ranges.begin(), ranges.end());
    ov::TensorVector repeated{ov::Tensor(type, values_shape)};
    if (!fq->evaluate(repeated, inputs)) {
        return false;
    }
    return std::memcmp(repeated[0].data(), fq_outputs[0].data(), fq_outputs[0].get_byte_size()) == 0;
}

bool is_per_tensor(const std::shared_ptr<v0::FakeQuantize>& fq) {
    for (size_t index = input_low_idx; index < fq->get_input_size(); ++index) {
        const auto& shape = fq->get_input_partial_shape(index);
        if (shape.is_dynamic() || ov::shape_size(shape.to_shape()) != 1) {
            return false;
        }
    }
    return true;
}

}  // namespace

bool FakeQuantizeEliminateSequential::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(FakeQuantizeEliminateSequential);

    bool eliminated = false;
    for (const auto& op : model->get_ordered_ops()) {
        auto fq1 = ov::as_type_ptr<v0::FakeQuantize>(op);
        if (!fq1 || !has_same_input_and_output_range(fq1)) {
            continue;
        }
        const bool per_tensor = is_per_tensor(fq1);

        // Walk forward from FQ1 following its consumers and collect the redundant FakeQuantizes.
        // Traversal continues only through FQ1, value-preserving ops, and redundant FakeQuantizes, so
        // branches into several consumers are handled as well. visit_path_forward queries a node's
        // consumers right after the visitor runs, so the redundant FakeQuantizes are only collected
        // here and detached afterwards to avoid dereferencing a freed node.
        std::vector<std::shared_ptr<v0::FakeQuantize>> redundant_fqs;
        std::unordered_set<ov::Node*> visited;
        auto skip_node = [&](ov::Node* node) {
            auto shared_node = node->shared_from_this();
            if (shared_node == fq1) {
                return false;
            }
            if (is_value_preserving(shared_node)) {
                return !per_tensor;
            }
            return !op_util::have_same_fake_quantize_params(fq1, ov::as_type_ptr<v0::FakeQuantize>(shared_node));
        };
        auto collect_redundant_fq = [&](ov::Node* node) {
            if (auto fq2 = ov::as_type_ptr<v0::FakeQuantize>(node->shared_from_this()); fq2 && fq2 != fq1) {
                redundant_fqs.push_back(fq2);
            }
        };
        op_util::visit_path_forward(fq1.get(), visited, collect_redundant_fq, skip_node);
        if (redundant_fqs.empty() || !is_idempotent(fq1)) {
            continue;
        }

        for (const auto& fq2 : redundant_fqs) {
            eliminated = replace_output_update_name(fq2->output(0), fq2->input_value(0)) || eliminated;
        }
    }

    return eliminated;
}

}  // namespace ov::pass
