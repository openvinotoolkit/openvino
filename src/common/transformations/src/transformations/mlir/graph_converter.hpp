// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

#include "mlir/IR/Value.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"

#include "common/typedefs.hpp"
#include "common/convert_common.hpp"
#include "common/conversion_context.hpp"

namespace ov {
namespace mlir {

using ::mlir::Value;
using ::mlir::MLIRContext;
using ::mlir::OpBuilder;
using ::mlir::Operation;
using ::mlir::SmallVector;
using ::mlir::ValueRange;

class GraphConverter {
    static std::string rt_info_convertor ();
    ConversionContext _ctx;

public:
    using Convertor = std::function<Operation*(ConversionContext&, NodePtr)>;
    using NodeOutputMap = std::map<ov::Output<ov::Node>, mlir::Value>;

    static const std::map<ov::DiscreteTypeInfo, Convertor> convertors;
    NodeOutputMap nodeOutputMap;
    std::map<SymbolPtr, Value> dimension_map;

    GraphConverter(mlir::MLIRContext* context, mlir::OpBuilder* block_builder);

    SmallVector<mlir::Value> getInputs(NodePtr node);
    void addOutputs(NodePtr node, mlir::Operation* op);

    static void set_convertor(NodePtr node, const Convertor& convertor);

    void convert(NodePtr node);

    Value get_dimension_value(const Dimension& d);
};


const std::string& subgraph_mark();

void set_subgraph_mark(NodePtr node);

bool get_subgraph_mark(NodePtr node);

class MarkPattern : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("MarkPattern", "0");
    MarkPattern(NodePtr pattern, GraphConverter::Convertor convertor);
};

} // namespace mlir
} // namespace ov