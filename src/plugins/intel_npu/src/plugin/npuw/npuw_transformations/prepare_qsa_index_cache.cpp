// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prepare_qsa_index_cache.hpp"

#include <numeric>
#include <string>
#include <vector>

#include "../util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/ops.hpp"
#include "openvino/op/qsa_prepare.hpp"

namespace {

using namespace ov;
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;
namespace v8 = ov::op::v8;
namespace v12 = ov::op::v12;

bool constant_is(const Output<Node>& output, int64_t expected) {
    const auto value = ov::util::get_constant_from_source(output);
    if (!value || ov::shape_size(value->get_shape()) != 1) {
        return false;
    }
    return value->cast_vector<int64_t>()[0] == expected;
}

bool slice_axis_one(const std::shared_ptr<v8::Slice>& slice) {
    return slice && slice->get_input_size() == 5 && constant_is(slice->input_value(1), 0) &&
           constant_is(slice->input_value(3), 1) && constant_is(slice->input_value(4), 1);
}

struct Plan {
    std::shared_ptr<v0::Parameter> parameter;
    std::shared_ptr<v1::Select> buffer_select;
    std::shared_ptr<v1::Less> previous_block_mask;
    std::shared_ptr<v0::Unsqueeze> range_row;
    std::shared_ptr<v0::Unsqueeze> condition;
    std::shared_ptr<v8::Slice> output_slice;
    size_t capacity = 0;
};

std::shared_ptr<v0::Result> find_present_result(const std::shared_ptr<ov::Model>& model, const std::string& name) {
    auto present_name = name;
    present_name.replace(present_name.find(".past."), std::string(".past.").size(), ".present.");
    std::shared_ptr<v0::Result> result;
    for (const auto& candidate : model->get_results()) {
        if (candidate->output(0).get_names().count(present_name)) {
            OPENVINO_ASSERT(!result, "Duplicate QSA state output: ", present_name);
            result = candidate;
        }
    }
    OPENVINO_ASSERT(result, "Missing QSA state output: ", present_name);
    return result;
}

Plan find_plan(const std::shared_ptr<ov::Model>& model,
               const std::shared_ptr<v0::Parameter>& parameter,
               const std::string& name,
               size_t context_size) {
    const auto block_size = ov::npuw::util::qsaIndexBlockSize(model, name);
    const auto capacity = context_size / block_size;
    OPENVINO_ASSERT(capacity > 0, "QSA context size must contain at least one complete block: ", name);
    const auto shape = parameter->get_partial_shape();
    OPENVINO_ASSERT(shape.rank().is_static() && shape.size() == 3 && shape[2].is_static() && shape[2].get_length() > 0,
                    "Invalid QSA block-index cache shape for ",
                    name,
                    ": ",
                    shape);
    const auto result = find_present_result(model, name);
    const auto present_name = result->output(0).get_any_name();

    auto output_slice = ov::as_type_ptr<v8::Slice>(result->get_input_node_shared_ptr(0));
    OPENVINO_ASSERT(slice_axis_one(output_slice), "Unsupported QSA state output slice: ", present_name);
    auto scatter = ov::as_type_ptr<v12::ScatterElementsUpdate>(output_slice->get_input_node_shared_ptr(0));
    OPENVINO_ASSERT(scatter && scatter->get_reduction() == v12::ScatterElementsUpdate::Reduction::NONE &&
                        scatter->get_use_init_val() && constant_is(scatter->input_value(3), 1),
                    "Unsupported QSA block-index update: ",
                    present_name);
    auto select = ov::as_type_ptr<v1::Select>(scatter->get_input_node_shared_ptr(0));
    OPENVINO_ASSERT(
        select && select->output(0).get_target_inputs().size() == 1 && constant_is(select->input_value(2), 0),
        "Unsupported QSA block-index buffer: ",
        present_name);
    auto input_slice = ov::as_type_ptr<v8::Slice>(select->get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(slice_axis_one(input_slice) && input_slice->output(0).get_target_inputs().size() == 1,
                    "Unsupported QSA block-index input slice: ",
                    present_name);
    auto concat = ov::as_type_ptr<v0::Concat>(input_slice->get_input_node_shared_ptr(0));
    OPENVINO_ASSERT(concat && concat->get_axis() == 1 && concat->get_input_size() == 2 &&
                        concat->input_value(0) == parameter->output(0),
                    "QSA block-index input is not the expected state buffer: ",
                    name);

    auto condition = ov::as_type_ptr<v0::Unsqueeze>(select->get_input_node_shared_ptr(0));
    auto less = condition ? ov::as_type_ptr<v1::Less>(condition->get_input_node_shared_ptr(0)) : nullptr;
    auto row = less ? ov::as_type_ptr<v0::Unsqueeze>(less->get_input_node_shared_ptr(0)) : nullptr;
    auto range = row ? ov::as_type_ptr<v4::Range>(row->get_input_node_shared_ptr(0)) : nullptr;
    auto end = range ? ov::as_type_ptr<v1::Add>(range->get_input_node_shared_ptr(1)) : nullptr;
    auto slice_end = ov::as_type_ptr<v0::Unsqueeze>(output_slice->get_input_node_shared_ptr(2));
    auto input_end = ov::as_type_ptr<v0::Unsqueeze>(input_slice->get_input_node_shared_ptr(2));
    OPENVINO_ASSERT(condition && constant_is(condition->input_value(1), -1) && row &&
                        constant_is(row->input_value(1), 0) && range && constant_is(range->input_value(0), 0) &&
                        constant_is(range->input_value(2), 1) && end && constant_is(end->input_value(1), 1) &&
                        slice_end && input_end && input_end->input_value(0) == end->output(0) &&
                        slice_end->input_value(0) == end->input_value(0),
                    "Unsupported QSA block-count mask: ",
                    present_name);

    auto index_broadcast = ov::as_type_ptr<v3::Broadcast>(scatter->get_input_node_shared_ptr(1));
    auto index_expand =
        index_broadcast ? ov::as_type_ptr<v0::Unsqueeze>(index_broadcast->get_input_node_shared_ptr(0)) : nullptr;
    auto index_select =
        index_expand ? ov::as_type_ptr<v1::Select>(index_expand->get_input_node_shared_ptr(0)) : nullptr;
    OPENVINO_ASSERT(index_select && index_select->input_value(2) == end->input_value(0),
                    "Unsupported QSA sentinel index: ",
                    present_name);
    return {parameter, select, less, row, condition, output_slice, capacity};
}

}  // namespace

bool ov::npuw::PrepareQSAIndexCache::run_on_model(const std::shared_ptr<ov::Model>& model) {
    std::vector<Plan> plans;
    size_t fused_count = 0;
    for (const auto& parameter : model->get_parameters()) {
        for (const auto& name : parameter->output(0).get_names()) {
            if (ov::npuw::util::isQSABlockIndexCache(name)) {
                auto source = find_present_result(model, name)->input_value(0);
                if (const auto convert = ov::as_type_ptr<v0::Convert>(source.get_node_shared_ptr())) {
                    source = convert->input_value(0);
                }
                if (const auto prepare = ov::as_type_ptr<ov::op::v17::QSAPrepare>(source.get_node_shared_ptr())) {
                    const auto block_size = ov::npuw::util::qsaIndexBlockSize(model, name);
                    OPENVINO_ASSERT(source.get_index() == 0 && prepare->input_value(1) == parameter->output(0) &&
                                        prepare->get_key_block_size() == static_cast<int64_t>(block_size) &&
                                        m_context_size / block_size > 0,
                                    "Unsupported fused QSA block-index state: ",
                                    name);
                    ++fused_count;
                    break;
                }
                plans.push_back(find_plan(model, parameter, name, m_context_size));
                break;
            }
        }
    }
    for (const auto& plan : plans) {
        const auto shape = plan.parameter->get_partial_shape();
        const auto head_dim = static_cast<size_t>(shape[2].get_length());
        std::vector<int64_t> positions(plan.capacity + 1);
        std::iota(positions.begin(), positions.end(), 0);
        auto range =
            v0::Constant::create(plan.range_row->get_input_element_type(0), Shape{positions.size()}, positions);
        auto row = plan.range_row->clone_with_new_inputs({range, plan.range_row->input_value(1)});
        auto less = plan.previous_block_mask->clone_with_new_inputs({row, plan.previous_block_mask->input_value(1)});
        auto condition = plan.condition->clone_with_new_inputs({less, plan.condition->input_value(1)});
        auto zero = v0::Constant::create(plan.parameter->get_element_type(), Shape{1, 1, head_dim}, {0});
        auto buffer = std::make_shared<v0::Concat>(OutputVector{plan.parameter, zero}, 1);
        auto end = v0::Constant::create(plan.output_slice->get_input_element_type(2), Shape{1}, {plan.capacity});
        ov::copy_runtime_info({plan.buffer_select, plan.output_slice},
                              {range, row, less, condition, zero, buffer, end});
        plan.buffer_select->input(0).replace_source_output(condition);
        plan.buffer_select->input(1).replace_source_output(buffer);
        plan.output_slice->input(2).replace_source_output(end);
    }
    return !plans.empty() || fused_count != 0;
}
