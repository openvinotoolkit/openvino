// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #include "mlir/IR/BuiltinAttributes.h"
// #include "mlir/IR/BuiltinTypes.h"

#include "graph_converter.hpp"

#include <utility>

#include "common/conversion_context.hpp"
#include "intel_gpu/primitives/mlir_primitive.hpp"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "openvino/util/common_util.hpp"

namespace ov::intel_gpu::mlir {

using namespace ::mlir;

namespace {

// A malformed request is a converter bug, so fail right away instead of silently ignoring the hint.
void validate_requested_order(const ov::Input<ov::Node>& input, ArrayRef<int64_t> order) {
    // Verify the requested order is valid for this input (static rank, permutation of the rank)
    const auto rank = input.get_partial_shape().rank();
    OPENVINO_ASSERT(rank.is_static() && order.size() == static_cast<size_t>(rank.get_length()) && mlir::isPermutationVector(order),
                    "[GPU] MLIR: invalid physical order requested for input ",
                    input.get_index(),
                    " of ",
                    input.get_node()->get_friendly_name());
    // The order must have a cldnn format: rethrow with the context of the request.
    const std::vector<size_t> dim_order(order.begin(), order.end());
    try {
        cldnn::mlir_primitive::format_from_dim_order(dim_order);
    } catch (const ov::Exception& e) {
        OPENVINO_THROW("[GPU] MLIR: no cldnn format for the physical order ",
                       ov::util::vector_to_string(dim_order),
                       " requested for input ",
                       input.get_index(),
                       " of ",
                       input.get_node()->get_friendly_name(),
                       ":\n",
                       e.what());
    }
}

}  // namespace

std::string GraphConverter::rt_info_convertor() {
    return "__mlir_convertor";
}

GraphConverter::GraphConverter(mlir::MLIRContext* context, mlir::OpBuilder* block_builder)
    : _ctx(
          context,
          block_builder,
          [this](const NodePtr& node) {
              return getInputs(node);
          },
          [this](const Dimension& dim) {
              return get_dimension_value(dim);
          },
          [this](const ov::Input<ov::Node>& input, DimOrder order) {
              request_input_order(input, std::move(order));
          }) {}

void GraphConverter::request_input_order(const ov::Input<ov::Node>& input, DimOrder order) {
    validate_requested_order(input, order);

    const auto source = input.get_source_output();
    const auto [it, inserted] = requested_orders.emplace(source, order);
    OPENVINO_ASSERT(inserted || it->second == order, "[GPU] MLIR: conflicting physical orders are requested for ", source);
}

ArrayRef<int64_t> GraphConverter::get_requested_order(const ov::Output<ov::Node>& source) const {
    const auto it = requested_orders.find(source);
    return it == requested_orders.end() ? ArrayRef<int64_t>{} : ArrayRef<int64_t>(it->second);
}

SmallVector<mlir::Value> GraphConverter::getInputs(const NodePtr& node) {
    SmallVector<mlir::Value> out;
    out.reserve(node->get_input_size());
    for (const auto& input : node->inputs()) {
        out.push_back(nodeOutputMap.at(input.get_source_output()));
    }
    return out;
}

void GraphConverter::addOutputs(const NodePtr& node, mlir::Operation* op) {
    const auto results = op->getOpResults();

    OPENVINO_ASSERT(results.size() == node->get_output_size(),
                    "Mismatch between original Node '{0}' number of outputs '{1}' and created number of outputs '{2}'",
                    node->get_friendly_name(),
                    node->get_output_size(),
                    results.size());

    for (const auto& res : results) {
        nodeOutputMap.emplace(node->output(res.getResultNumber()), res);
    }
}

void GraphConverter::convert(const NodePtr& node) {
    const auto& rt_info = node->get_rt_info();
    auto it = rt_info.find(rt_info_convertor());
    OPENVINO_ASSERT(it != rt_info.end(), "No MLIR converter registered for node ", node->get_type_name());
    auto convertor = it->second.as<Convertor>();
    auto* mlirOp = convertor(_ctx, node);
    addOutputs(node, mlirOp);
}

void GraphConverter::set_convertor(const NodePtr& node, const Convertor& convertor) {
    Convertor local_copy = convertor;
    auto as_any = ov::Any(local_copy);
    node->get_rt_info()[rt_info_convertor()] = as_any;
}

Value GraphConverter::get_dimension_value(const Dimension& d) {
    auto symbol = d.get_symbol();
    OPENVINO_ASSERT(symbol, "Dimension ", d, " has no symbol and cannot be converted to an MLIR value");
    symbol = ov::symbol::ancestor_of(symbol);
    // Suppose all dimensions are known and the map is populated
    // FIXME: Add dimensions on demand to avoid unnecessary operations in the produced MLIR
    const auto it = dimension_map.find(symbol);
    OPENVINO_ASSERT(it != dimension_map.end(), "Symbol of dimension ", d, " is not present in the dimension map");
    return it->second;
}

const std::string& subgraph_mark() {
    static const std::string mark = "__subgraph_mlir_mark";
    return mark;
}

void set_subgraph_mark(const NodePtr& node, const std::string& name) {
    node->get_rt_info()[subgraph_mark()] = name;
}

bool has_subgraph_mark(const NodePtr& node) {
    return node->get_rt_info().count(subgraph_mark()) != 0U;
}

std::string get_subgraph_mark(const NodePtr& node) {
    const auto& rti = node->get_rt_info();
    auto it = rti.find(subgraph_mark());
    return it == rti.end() ? "" : it->second.as<std::string>();
}

MarkPattern::MarkPattern(const NodePtr& pattern, const GraphConverter::Convertor& convertor) {
    auto callback = [convertor](ov::pass::pattern::Matcher& m) {
        auto node = m.get_match_root();
        set_subgraph_mark(node);
        GraphConverter::set_convertor(node, convertor);
        return true;
    };
    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, "MarkPattern");
    register_matcher(m, callback);
}

MarkPattern::MarkPattern(const NodePtr& pattern, const MarkPattern::Callback& callback) {
    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, "MarkPattern");
    register_matcher(m, callback);
}

}  // namespace ov::intel_gpu::mlir