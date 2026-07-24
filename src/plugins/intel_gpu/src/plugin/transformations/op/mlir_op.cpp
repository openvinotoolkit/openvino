// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef GRAPH_COMPILER

#include "intel_gpu/op/mlir_op.hpp"

#include <cassert>
#include <memory>
#include <vector>

#include "openvino/core/shape.hpp"
#include "../mlir/interface/mlir_evaluate_base.hpp"
#include "../mlir/interface/properties.hpp"

namespace ov::intel_gpu::op {

namespace {

// Descriptor packed into the 5-slot layout expected by
// MLIREvaluateGcGPU::invoke_packed:
//   [aligned, rank, shape*, strides*, is_usm]
// TODO: u4/i4 types are not supported
struct MemRefDescriptor {
    MemRefDescriptor() = default;

    MemRefDescriptor(ov::Tensor tensor, const ov::PartialShape& module_input_shape)
        : allocated(tensor.data()),
          aligned(tensor.data()),
          offset(0) {
        if (module_input_shape.rank() == shape_size(tensor.get_shape())) {
            shape.assign(tensor.get_shape().begin(), tensor.get_shape().end());
        } else {
            auto it = tensor.get_shape().begin();
            std::advance(it, module_input_shape.rank().get_length());
            shape.assign(tensor.get_shape().begin(), it);

            if (std::any_of(it, tensor.get_shape().end(), [](size_t dim) { return dim != 1; })) {
                OPENVINO_THROW("Mismatch in shape sizes");
            }
        }

        strides.resize(shape.size());
        const auto& byte_strides = tensor.get_strides();
        auto element_size = tensor.get_element_type().size();
        for (size_t i = 0; i < strides.size(); ++i) {
            assert(byte_strides[i] % element_size == 0);
            // TODO: handle case when stride is not aligned (restrict at OV API level)
            strides[i] = byte_strides[i] / element_size;
        }
    }

    explicit MemRefDescriptor(ov::Tensor tensor) : MemRefDescriptor(tensor, tensor.get_shape()) {}

    void* allocated;
    void* aligned;
    int64_t offset;
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;

    void append_to_packed_args(std::vector<void*>& args, bool is_usm) {
        args.push_back(aligned);
        args.push_back(reinterpret_cast<void*>(shape.size()));
        args.push_back(shape.data());
        args.push_back(strides.data());
        args.push_back(reinterpret_cast<void*>(static_cast<uintptr_t>(is_usm)));
    }
};

}  // namespace

MLIROp::MLIROp(const ov::OutputVector& args,
               std::shared_ptr<mlir::MLIREvaluateBase> engine,
               const OVOutputTypes& output_types,
               const DimensionsMap& dimensions_map)
    : Op(args),
      engine(std::move(engine)),
      output_types(output_types),
      dimensions_map(dimensions_map) {
    constructor_validate_and_infer_types();
}

std::vector<ov::PartialShape> MLIROp::shape_infer(const std::vector<ov::PartialShape>& input_shapes) const {
    OPENVINO_ASSERT(dimensions_map.size() == output_types.size(),
                    "MLIROp::shape_infer: dimensions_map size (", dimensions_map.size(),
                    ") does not match output_types size (", output_types.size(), ")");

    std::vector<ov::PartialShape> output_shapes;
    output_shapes.reserve(output_types.size());
    for (size_t i = 0; i < output_types.size(); ++i) {
        ov::PartialShape resolved = std::get<1>(output_types[i]);
        OPENVINO_ASSERT(dimensions_map[i].size() == resolved.size(),
                        "MLIROp::shape_infer: dimensions_map[", i, "] size (", dimensions_map[i].size(),
                        ") does not match output ", i, " rank (", resolved.size(), ")");

        for (size_t j = 0; j < resolved.size(); ++j) {
            if (!resolved[j].is_dynamic()) {
                continue;
            }
            size_t input_index, dim_index;
            std::tie(input_index, dim_index) = dimensions_map[i][j];
            OPENVINO_ASSERT(input_index < input_shapes.size(),
                            "MLIROp::shape_infer: dimensions_map[", i, "][", j, "] refers to input ",
                            input_index, " but only ", input_shapes.size(), " input shapes provided");
            OPENVINO_ASSERT(dim_index < input_shapes[input_index].size(),
                            "MLIROp::shape_infer: dimensions_map[", i, "][", j, "] refers to dim ",
                            dim_index, " of input ", input_index, " (rank ",
                            input_shapes[input_index].size(), ")");
            resolved[j] = input_shapes[input_index][dim_index];
        }
        output_shapes.push_back(resolved);
    }
    return output_shapes;
}

void MLIROp::validate_and_infer_types() {
    std::vector<ov::PartialShape> input_shapes;
    input_shapes.reserve(get_input_size());
    for (size_t i = 0; i < get_input_size(); ++i) {
        input_shapes.push_back(get_input_partial_shape(i));
    }
    auto output_shapes = shape_infer(input_shapes);

    set_output_size(output_types.size());
    for (size_t i = 0; i < output_types.size(); ++i) {
        set_output_type(i, std::get<0>(output_types[i]), output_shapes[i]);
    }
}

std::shared_ptr<ov::Node> MLIROp::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<MLIROp>(new_args, engine, output_types, dimensions_map);
}

bool MLIROp::evaluate(ov::TensorVector& outputs,
                     const ov::TensorVector& inputs,
                     const ov::EvaluationContext& evaluationContext) const {
    if (!engine->requires_packed_args()) {
        return engine->invoke(inputs, outputs, evaluationContext);
    }

    std::vector<MemRefDescriptor> memref_args;
    memref_args.reserve(inputs.size() + outputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& initial_shape = get_input_partial_shape(i);
        memref_args.emplace_back(inputs[i], initial_shape);
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
        // TODO: Optimize by adding all dimensions to dimensions_map, not only dynamic
        ov::Shape target;
        ov::PartialShape expected = get_output_partial_shape(i);
        for (size_t j = 0; j < expected.size(); ++j) {
            auto dim = expected[j];
            if (dim.is_dynamic()) {
                size_t input_index, dim_index;
                std::tie(input_index, dim_index) = dimensions_map[i][j];
                target.push_back(inputs[input_index].get_shape()[dim_index]);
            } else {
                target.push_back(dim.get_length());
            }
        }
        outputs[i].set_shape(target);
        memref_args.emplace_back(outputs[i]);
    }

    std::vector<bool> is_usm;
    auto it = evaluationContext.find(ov::internal::mlir_meta::is_kernel_arg_usm.name());
    if (it != evaluationContext.end()) {
        is_usm = it->second.as<std::vector<bool>>();
    } else {
        is_usm.assign(memref_args.size(), false);
    }
    OPENVINO_ASSERT(is_usm.size() == memref_args.size(),
                    "[GPU] MLIROp::evaluate: is_usm and memref count mismatch");

    std::vector<void*> args;
    for (size_t k = 0; k < memref_args.size(); ++k) {
        memref_args[k].append_to_packed_args(args, is_usm[k]);
    }

    return engine->invoke_packed(args, evaluationContext);
}

bool MLIROp::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    return evaluate(outputs, inputs, ov::EvaluationContext());
}

bool MLIROp::has_evaluate() const {
    return true;
}

}  // namespace ov::intel_gpu::op

#endif  // GRAPH_COMPILER
