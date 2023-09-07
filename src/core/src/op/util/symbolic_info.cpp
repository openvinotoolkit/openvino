// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/symbolic_info.hpp"

#include <openvino/core/model.hpp>

#include "openvino/core/dimension_tracker.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"

namespace {
std::shared_ptr<ov::TableOfEquivalence> get_table(const ov::RTMap& rt_info) {
    const auto& type = ov::SymbolicInfo::get_type_info_static();
    if (!rt_info.count(type))
        return nullptr;
    return rt_info.at(type).as<ov::SymbolicInfo>().get_table();
}
}  // namespace

void ov::set_up_symbolic_info(const std::shared_ptr<ov::Model>& model,
                              const std::shared_ptr<ov::TableOfEquivalence>& table) {
    auto& rt_info = model->get_rt_info();
    rt_info[ov::SymbolicInfo::get_type_info_static()] = ov::SymbolicInfo(true, table);
}

void ov::set_up_symbolic_info(const ov::Output<ov::Node>& output,
                              const std::shared_ptr<ov::TableOfEquivalence>& table) {
    auto& rt_info = output.get_tensor().get_rt_info();
    rt_info[ov::SymbolicInfo::get_type_info_static()] = ov::SymbolicInfo(true, table);
}

bool ov::skip_invalidation(const ov::descriptor::Tensor& tensor) {
    const auto& rt_info = tensor.get_rt_info();
    const auto& type = ov::SymbolicInfo::get_type_info_static();
    return rt_info.count(type) && rt_info.at(type).as<ov::SymbolicInfo>().get_skip_invalidation();
}

std::shared_ptr<ov::TableOfEquivalence> ov::table_of_equivalence(const ov::descriptor::Tensor& tensor) {
    const auto& rt_info = tensor.get_rt_info();
    return get_table(rt_info);
}

std::shared_ptr<ov::TableOfEquivalence> ov::table_of_equivalence(const std::shared_ptr<ov::Model>& model) {
    const auto& rt_info = model->get_rt_info();
    return get_table(rt_info);
}

void ov::populate_tensor_with_missing_labels(ov::descriptor::Tensor& tensor) {
    if (auto table = ov::table_of_equivalence(tensor)) {
        auto label_values = tensor.get_value_label();
        if (label_values.empty()) {
            const auto& pshape = tensor.get_partial_shape();
            if (pshape.is_dynamic())
                return;
            label_values.resize(ov::shape_size(pshape.to_shape()), ov::no_label);
        }
        for (auto& label : label_values)
            if (label == ov::no_label)
                label = table->get_next_label();
        tensor.set_value_label(label_values);
    }
}

void ov::remove_symbolic_info(const std::shared_ptr<ov::Model>& model, bool outermost_model) {
    const auto& type = ov::SymbolicInfo::get_type_info_static();
    auto& model_rt_info = model->get_rt_info();
    if (model_rt_info.count(type))
        model_rt_info.erase(type);
    for (const auto& op : model->get_ops()) {
        if (auto multi_subgraph_op = std::dynamic_pointer_cast<op::util::MultiSubGraphOp>(op))
            for (const auto& sub_graph : multi_subgraph_op->get_functions())
                if (sub_graph)
                    remove_symbolic_info(sub_graph, false);
        for (auto& output : op->outputs()) {
            auto& rt_info = output.get_tensor().get_rt_info();
            if (rt_info.count(type))
                rt_info.erase(type);
        }
    }
    if (outermost_model)
        model->validate_nodes_and_infer_types();
}
