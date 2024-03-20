// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/symbolic_info.hpp"

#include <openvino/core/model.hpp>

#include "openvino/op/util/multi_subgraph_base.hpp"

void ov::set_up_symbolic_info(const ov::Output<ov::Node>& output) {
    auto& rt_info = output.get_tensor().get_rt_info();
    rt_info[ov::SymbolicInfo::get_type_info_static()] = ov::SymbolicInfo(true);
}

bool ov::skip_invalidation(const ov::descriptor::Tensor& tensor) {
    const auto& rt_info = tensor.get_rt_info();
    const auto& type = ov::SymbolicInfo::get_type_info_static();
    return rt_info.count(type) && rt_info.at(type).as<ov::SymbolicInfo>().get_skip_invalidation();
}

void ov::populate_tensor_with_missing_symbols(ov::descriptor::Tensor& tensor) {
    if (!tensor.get_rt_info().count(ov::SymbolicInfo::get_type_info_static()))
        return;
    auto symbol_values = tensor.get_value_symbol();
    if (symbol_values.empty()) {
        const auto& pshape = tensor.get_partial_shape();
        if (pshape.is_dynamic())
            return;
        symbol_values.resize(ov::shape_size(pshape.to_shape()), nullptr);
    }
    for (auto& symbol : symbol_values)
        if (symbol == nullptr)
            symbol = std::make_shared<Symbol>();
    tensor.set_value_symbol(symbol_values);
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
