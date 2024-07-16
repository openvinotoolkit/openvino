// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #include "mlir/IR/BuiltinAttributes.h"
// #include "mlir/IR/BuiltinTypes.h"

#include "conversion_context.hpp"


namespace ov {
namespace mlir {

using namespace ::mlir;


std::string ConversionContext::rt_info_convertor () {
    return "__mlir_convertor";
}


ConversionContext::ConversionContext(mlir::MLIRContext* context, mlir::OpBuilder* block_builder)
    : context(context),
        block_builder(block_builder) {}

SmallVector<mlir::Value> ConversionContext::getInputs(NodePtr node) {
    SmallVector<mlir::Value> out;
    out.reserve(node->get_input_size());
    for (const auto& input : node->inputs()) {
        out.push_back(nodeOutputMap.at(input.get_source_output()));
    }
    return out;
}

void ConversionContext::addOutputs(NodePtr node, mlir::Operation* op) {
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

void ConversionContext::convert(NodePtr node) {
    auto convertor = node->get_rt_info()[rt_info_convertor()].as<Convertor>();
    convertor(*this, node);
}

void ConversionContext::set_convertor(NodePtr node, const Convertor& convertor) {
    Convertor local_copy = convertor;
    auto as_any = ov::Any(local_copy);
    node->get_rt_info()[rt_info_convertor()] = as_any;
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


MarkPattern::MarkPattern(NodePtr pattern, ConversionContext::Convertor convertor) {
    auto callback = [convertor](ov::pass::pattern::Matcher& m) {
        // TODO: support multi-node patterns marking
        auto node = m.get_match_root();
        set_subgraph_mark(node);
        ConversionContext::set_convertor(node, convertor);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, "MarkPattern");
    register_matcher(m, callback);
}


} // namespace mlir
} // namespace ov