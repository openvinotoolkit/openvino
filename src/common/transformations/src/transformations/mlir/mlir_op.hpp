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

class MLIREvaluate {
    OwningOpRef<ModuleOp> module;  // FIXME: needs to be kept?
    std::unique_ptr<ExecutionEngine> engine;

public:

    MLIREvaluate(OwningOpRef<ModuleOp> _module);
    bool invoke_packed(std::vector<void*>& args);
};


class OPENVINO_API MLIROp : public ov::op::Op {
    std::shared_ptr<MLIREvaluate> engine;
    OVOutputTypes output_types;

public:

    OPENVINO_OP("MLIROp");

    MLIROp(const ov::OutputVector& args, std::shared_ptr<MLIREvaluate> engine, const OVOutputTypes& output_types);
    void validate_and_infer_types() override;
    NodePtr clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

} // namespace mlir
} // namespace ov