// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef ENABLE_MLIR_FOR_GPU

#    include "intel_gpu/op/mlir_op.hpp"

#    include <memory>
#    include <utility>
#    include <vector>

#    include "openvino/core/shape.hpp"

namespace ov::intel_gpu::op {

MLIROp::MLIROp(const ov::OutputVector& args,
               std::shared_ptr<mlir::MLIRGpuProgram> program,
               OVOutputTypes output_types,
               DimensionsMap dimensions_map)
    : Op(args),
      program(std::move(program)),
      output_types(std::move(output_types)),
      dimensions_map(std::move(dimensions_map)) {
    constructor_validate_and_infer_types();
}

std::vector<ov::PartialShape> MLIROp::shape_infer(const std::vector<ov::PartialShape>& input_shapes) const {
    OPENVINO_ASSERT(dimensions_map.size() == output_types.size(),
                    "MLIROp::shape_infer: dimensions_map size (",
                    dimensions_map.size(),
                    ") does not match output_types size (",
                    output_types.size(),
                    ")");

    std::vector<ov::PartialShape> output_shapes;
    output_shapes.reserve(output_types.size());
    for (size_t i = 0; i < output_types.size(); ++i) {
        ov::PartialShape resolved = std::get<1>(output_types[i]);
        OPENVINO_ASSERT(dimensions_map[i].size() == resolved.size(),
                        "MLIROp::shape_infer: dimensions_map[",
                        i,
                        "] size (",
                        dimensions_map[i].size(),
                        ") does not match output ",
                        i,
                        " rank (",
                        resolved.size(),
                        ")");

        for (size_t j = 0; j < resolved.size(); ++j) {
            if (!resolved[j].is_dynamic()) {
                continue;
            }
            size_t input_index = 0, dim_index = 0;
            std::tie(input_index, dim_index) = dimensions_map[i][j];
            OPENVINO_ASSERT(input_index < input_shapes.size(),
                            "MLIROp::shape_infer: dimensions_map[",
                            i,
                            "][",
                            j,
                            "] refers to input ",
                            input_index,
                            " but only ",
                            input_shapes.size(),
                            " input shapes provided");
            OPENVINO_ASSERT(dim_index < input_shapes[input_index].size(),
                            "MLIROp::shape_infer: dimensions_map[",
                            i,
                            "][",
                            j,
                            "] refers to dim ",
                            dim_index,
                            " of input ",
                            input_index,
                            " (rank ",
                            input_shapes[input_index].size(),
                            ")");
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
    return std::make_shared<MLIROp>(new_args, program, output_types, dimensions_map);
}

}  // namespace ov::intel_gpu::op

#endif  // ENABLE_MLIR_FOR_GPU
