// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/JitRunner.h"
#include "mlir/ExecutionEngine/OptUtils.h"

#include "openvino/op/op.hpp"

#include "convert_common.hpp"


namespace ov {
namespace mlir {

using ::mlir::OwningOpRef;
using ::mlir::ModuleOp;
using ::mlir::ExecutionEngine;
using ::mlir::ModuleOp;

enum MlirMode {
    MLIR_MODE_TPP,
    MLIR_MODE_GC,
    MLIR_MODE_DEFAULT,
};


class MLIREvaluate {
    OwningOpRef<ModuleOp> module;  // FIXME: needs to be kept?
    std::unique_ptr<ExecutionEngine> engine;

public:

    MLIREvaluate(OwningOpRef<ModuleOp> _module, MlirMode mode);
    bool invoke_packed(std::vector<void*>& args);
};


// Maps [output index][dimension index] -> [input index][dimension index] to infer shapes for entire subgraph
using DimensionsMap = std::vector<std::vector<std::tuple<size_t, size_t>>>;


class OPENVINO_API MLIROp : public ov::op::Op {
    std::shared_ptr<MLIREvaluate> engine;
    OVOutputTypes output_types;
    DimensionsMap dimensions_map;

public:

    OPENVINO_OP("MLIROp");

    MLIROp(const ov::OutputVector& args, std::shared_ptr<MLIREvaluate> engine, const OVOutputTypes& output_types, const DimensionsMap& dimensions_map);
    void validate_and_infer_types() override;
    NodePtr clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

} // namespace mlir
} // namespace ov