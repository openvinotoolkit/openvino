// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/op/gated_residual.hpp"
#include "utils.hpp"

namespace ov::op::internal {
namespace gated_residual_shape_inference {

template <class TDim>
TDim make_dim(size_t value) {
    return TDim(static_cast<typename TDim::value_type>(value));
}

/// Rank is dynamic, or the last dimension is compatible with `expected`.
template <class TShape>
bool last_dim_matches(const TShape& shape, size_t expected) {
    using TDim = typename TShape::value_type;
    return shape.rank().is_dynamic() ||
           (shape.size() > 0 && shape[shape.size() - 1].compatible(make_dim<TDim>(expected)));
}

/// Rank is dynamic, or the shape is [rows, cols] with compatible dimensions.
template <class TShape>
bool matrix_matches(const TShape& shape, size_t rows, size_t cols) {
    using TDim = typename TShape::value_type;
    return shape.rank().is_dynamic() || (shape.size() == 2 && shape[0].compatible(make_dim<TDim>(rows)) &&
                                         shape[1].compatible(make_dim<TDim>(cols)));
}

/// A static shape of one element (any rank).
template <class TShape>
bool has_single_element(const TShape& shape) {
    if (shape.rank().is_dynamic()) {
        return false;
    }
    size_t count = 1;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i].is_dynamic()) {
            return false;
        }
        count *= static_cast<size_t>(shape[i].get_length());
    }
    return count == 1;
}

}  // namespace gated_residual_shape_inference

/// Output shapes of GatedResidual and the validation of the input shapes. The element types are checked by the op.
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const GatedResidual* op, const std::vector<T>& input_shapes) {
    using namespace gated_residual_shape_inference;
    using TDim = typename T::value_type;
    using Mode = GatedResidual::Mode;

    const auto& config = op->get_config();
    // The op has checked that C * D does not overflow.
    const size_t cd = config.hc_count * config.hidden_size;

    const auto& residual = input_shapes[0];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           residual.rank().is_dynamic() || residual.size() >= 2,
                           "GatedResidual requires the residual input rank to be at least 2.");
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           last_dim_matches(residual, cd),
                           "GatedResidual residual last dimension must be C*D = ",
                           cd,
                           ", got ",
                           residual);

    if (config.mode == Mode::WRITE) {
        NODE_VALIDATION_CHECK(op, input_shapes.size() == 3, "GatedResidual (WRITE) expects 3 inputs.");
        const auto& y = input_shapes[1];
        const auto& gate = input_shapes[2];
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               last_dim_matches(y, config.hidden_size),
                               "GatedResidual (WRITE) sub-layer output last dimension must be D = ",
                               config.hidden_size);
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               last_dim_matches(gate, config.hc_count),
                               "GatedResidual (WRITE) write gate last dimension must be C = ",
                               config.hc_count);
        // Y [.., D] and the gate [.., C] carry one row per residual row: same rank and compatible leading dimensions.
        auto rank = residual.rank();
        NODE_SHAPE_INFER_CHECK(
            op,
            input_shapes,
            ov::Rank::merge(rank, rank, y.rank()) && ov::Rank::merge(rank, rank, gate.rank()),
            "GatedResidual (WRITE) residual, sub-layer output and write gate must have the same rank, got ",
            residual,
            ", ",
            y,
            " and ",
            gate);
        // R' has the residual layout; its width is known even when the residual's is not.
        TRShape output;
        if (residual.rank().is_static()) {
            output = residual;
        } else {
            output = PartialShape::dynamic(rank);
        }
        if (rank.is_static()) {
            const auto last = static_cast<size_t>(rank.get_length()) - 1;
            // A dynamic-rank input has no dimensions to merge.
            for (const auto* other : {&y, &gate}) {
                if (other->rank().is_dynamic()) {
                    continue;
                }
                for (size_t i = 0; i < last; ++i) {
                    NODE_SHAPE_INFER_CHECK(op,
                                           input_shapes,
                                           TDim::merge(output[i], output[i], (*other)[i]),
                                           "GatedResidual (WRITE) residual, sub-layer output and write gate must have "
                                           "matching leading dimensions, got ",
                                           residual,
                                           ", ",
                                           y,
                                           " and ",
                                           gate);
                }
            }
            output[last] = make_dim<TDim>(cd);
        }
        return {std::move(output)};
    }

    // READ / FINAL_MIX
    const auto layout = op->get_read_layout();
    NODE_VALIDATION_CHECK(op,
                          layout.has_value() && layout->input_count == input_shapes.size(),
                          "GatedResidual (READ/FINAL_MIX) got an unexpected number of inputs: ",
                          input_shapes.size());
    const auto& gain = input_shapes[layout->norm_weight];
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           gain.rank().is_dynamic() || (gain.size() == 1 && gain[0].compatible(make_dim<TDim>(cd))),
                           "GatedResidual norm weight must have shape [C*D] = [",
                           cd,
                           "], got ",
                           gain);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           matrix_matches(input_shapes[layout->w_down], config.lowrank, cd),
                           "GatedResidual W_down must have shape [L, C*D] = [",
                           config.lowrank,
                           ", ",
                           cd,
                           "], got ",
                           input_shapes[layout->w_down]);
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           matrix_matches(input_shapes[layout->w_up], cd, config.lowrank),
                           "GatedResidual W_up must have shape [C*D, L] = [",
                           cd,
                           ", ",
                           config.lowrank,
                           "], got ",
                           input_shapes[layout->w_up]);

    if (layout->compressed) {
        // scale / zp are [N, G]; the group size along K is K / G (not an attribute). A single-element one (per-tensor)
        // is also accepted, as by FullyConnectedCompressed: the backend broadcasts it.
        const auto check_group = [&](size_t scale_idx, size_t zp_idx, size_t n, size_t k, const char* name) {
            const auto& scale = input_shapes[scale_idx];
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   scale.rank().is_dynamic() || has_single_element(scale) ||
                                       (scale.size() == 2 && scale[0].compatible(make_dim<TDim>(n))),
                                   "GatedResidual ",
                                   name,
                                   " scale must have shape [",
                                   n,
                                   ", G] or one element, got ",
                                   scale);
            if (layout->has_zp) {
                const auto& zp = input_shapes[zp_idx];
                NODE_SHAPE_INFER_CHECK(
                    op,
                    input_shapes,
                    zp.rank().is_dynamic() || scale.rank().is_dynamic() || has_single_element(zp) || zp == scale,
                    "GatedResidual ",
                    name,
                    " zero point must match the scale shape or be one element, got ",
                    zp);
            }
            if (scale.rank().is_static() && !has_single_element(scale) && scale[1].is_static()) {
                const auto g = static_cast<size_t>(scale[1].get_length());
                NODE_SHAPE_INFER_CHECK(op,
                                       input_shapes,
                                       g != 0 && k % g == 0,
                                       "GatedResidual ",
                                       name,
                                       ": K = ",
                                       k,
                                       " is not divisible by the scale group count G = ",
                                       g);
            }
        };
        check_group(layout->scale_down, layout->zp_down, config.lowrank, cd, "W_down");
        check_group(layout->scale_up, layout->zp_up, cd, config.lowrank, "W_up");
        if (config.produce_write_gate) {
            check_group(layout->scale_inject, layout->zp_inject, config.hc_count, cd, "W_inject");
        }
    }
    if (config.produce_write_gate) {
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               matrix_matches(input_shapes[layout->w_inject], config.hc_count, cd),
                               "GatedResidual W_inject must have shape [C, C*D] = [",
                               config.hc_count,
                               ", ",
                               cd,
                               "], got ",
                               input_shapes[layout->w_inject]);
    }

    std::vector<TRShape> output_shapes(config.produce_write_gate ? 2 : 1, residual);
    if (residual.rank().is_static()) {
        output_shapes[0][residual.size() - 1] = make_dim<TDim>(config.hidden_size);
        if (config.produce_write_gate) {
            output_shapes[1][residual.size() - 1] = make_dim<TDim>(config.hc_count);
        }
    }
    return output_shapes;
}

}  // namespace ov::op::internal
