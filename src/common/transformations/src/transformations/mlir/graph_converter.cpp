// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #include "mlir/IR/BuiltinAttributes.h"
// #include "mlir/IR/BuiltinTypes.h"

#include "common/conversion_context.hpp"
#include "graph_converter.hpp"


namespace ov {
namespace mlir {

using namespace ::mlir;


std::string GraphConverter::rt_info_convertor () {
    return "__mlir_convertor";
}


GraphConverter::GraphConverter(mlir::MLIRContext* context, mlir::OpBuilder* block_builder)
    : _ctx(context, block_builder, [this](NodePtr node) { return getInputs(node); }, [this](const Dimension& dim) { return get_dimension_value(dim); }) {}

SmallVector<mlir::Value> GraphConverter::getInputs(NodePtr node) {
    SmallVector<mlir::Value> out;
    out.reserve(node->get_input_size());
    for (const auto& input : node->inputs()) {
        out.push_back(nodeOutputMap.at(input.get_source_output()));
    }
    return out;
}

void GraphConverter::addOutputs(NodePtr node, mlir::Operation* op) {
    const auto results = op->getOpResults();

    OPENVINO_ASSERT(
        results.size() == node->get_output_size(),
        "Mismatch between original Node '{0}' number of outputs '{1}' and created number of outputs '{2}'",
        node->get_friendly_name(),
        node->get_output_size(),
        results.size());

    for (const auto& res : results) {
        nodeOutputMap.emplace(node->output(res.getResultNumber()), res);
    }
}

void GraphConverter::convert(NodePtr node) {
    auto convertor = node->get_rt_info()[rt_info_convertor()].as<Convertor>();
    auto mlirOp = convertor(_ctx, node);
    addOutputs(node, mlirOp);
}

void GraphConverter::set_convertor(NodePtr node, const Convertor& convertor) {
    Convertor local_copy = convertor;
    auto as_any = ov::Any(local_copy);
    node->get_rt_info()[rt_info_convertor()] = as_any;
}

Value GraphConverter::get_dimension_value(const Dimension& d) {
    auto symbol = d.get_symbol();
    assert(symbol);
    symbol = ov::symbol::ancestor_of(symbol);
    // Suppose all dimensions are known and the map is populated
    // FIXME: Add dimensions on demand to avoid unnecessary operations in the produced MLIR
    assert(dimension_map.count(symbol));
    return dimension_map.at(symbol);
}


const std::string& subgraph_mark() {
    static const std::string mark = "__subgraph_mlir_mark";
    return mark;
}

void set_subgraph_mark(NodePtr node) {
    node->get_rt_info()[subgraph_mark()];
}

bool get_subgraph_mark(NodePtr node) {
    return node->get_rt_info().count(subgraph_mark());
}


MarkPattern::MarkPattern(NodePtr pattern, GraphConverter::Convertor convertor) {
    auto callback = [convertor](ov::pass::pattern::Matcher& m) {
        // TODO: support multi-node patterns marking
        auto node = m.get_match_root();
        set_subgraph_mark(node);
        GraphConverter::set_convertor(node, convertor);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, "MarkPattern");
    register_matcher(m, callback);
}


} // namespace mlir
} // namespace ov