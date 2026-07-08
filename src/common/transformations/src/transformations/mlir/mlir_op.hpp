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
#include "openvino/core/shape.hpp"

#include "common/convert_common.hpp"

#include "gc/Utils/Error.h"
#include "gc/ExecutionEngine/GPURuntime/GpuOclRuntime.h"

namespace ov {
namespace mlir {

using ::mlir::OwningOpRef;
using ::mlir::ModuleOp;
using ::mlir::ExecutionEngine;

class MLIROp;

class MLIREvaluateBase {
public:
    virtual bool requires_packed_args() const = 0;
    // ::invoke() doesn't require any args preprocessing so we can pass tensors as is
    virtual bool invoke(const ov::TensorVector& inputs, ov::TensorVector& outputs, const ov::EvaluationContext& evaluationContext) = 0;
    virtual bool invoke_packed(std::vector<void*>& args, const ov::EvaluationContext& evaluationContext) = 0;
    virtual ~MLIREvaluateBase() = default;
};

class MLIREvaluateGcGPU : public MLIREvaluateBase {
    std::shared_ptr<const gc::gpu::OclModule> module;

public:
    MLIREvaluateGcGPU(OwningOpRef<ModuleOp> _module, std::shared_ptr<ov::EvaluationContext> loweringContext);

    bool requires_packed_args() const override { return !module->isStatic; }
    bool invoke(const ov::TensorVector& inputs, ov::TensorVector& outputs, const ov::EvaluationContext& evaluationContext) override;
    bool invoke_packed(std::vector<void*>& args, const ov::EvaluationContext& evaluationContext) override;

private:
    gc::gpu::OclContext build_ocl_context(const ov::EvaluationContext& evaluationContext);
    static void maybe_set_result_event(const ov::EvaluationContext& evaluationContext, gc::gpu::OclContext& ctx);
};

// Maps [output index][dimension index] -> [input index][dimension index] to infer shapes for entire subgraph
using DimensionsMap = std::vector<std::vector<std::tuple<size_t, size_t>>>;


class OPENVINO_API MLIROp : public ov::op::Op {
    std::shared_ptr<MLIREvaluateBase> engine;
    OVOutputTypes output_types;
    DimensionsMap dimensions_map;

public:

    OPENVINO_OP("MLIROp");

    MLIROp(const ov::OutputVector& args, std::shared_ptr<MLIREvaluateBase> engine,
                                        const OVOutputTypes& output_types, const DimensionsMap& dimensions_map);
    void validate_and_infer_types() override;
    NodePtr clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool evaluate(ov::TensorVector& output_values, const ov::TensorVector& input_values,
                  const ov::EvaluationContext& evaluationContext) const override;
    bool has_evaluate() const override;
    std::vector<ov::PartialShape> shape_infer(const std::vector<ov::PartialShape>& input_shapes) const;
};

} // namespace mlir
} // namespace ov