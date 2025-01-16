// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "snippets/shape_inference/shape_infer_instances.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils/utils.hpp"
#include "openvino/op/select.hpp"
namespace ov {
namespace snippets {
using Result = IShapeInferSnippets::Result;
/*
 * Merge SRC to DST with broadcasting rules defined by the Autobroadcast specifier
 */
bool broadcast_merge_into(VectorDims& dst, const VectorDims& src, const ov::op::AutoBroadcastSpec& autob) {
    // Ranks are both static.
    const auto dst_rank = static_cast<int64_t>(dst.size());
    const auto src_rank = static_cast<int64_t>(src.size());
    switch (autob.m_type) {
        case ov::op::AutoBroadcastType::NONE:
            return true;
        case ov::op::AutoBroadcastType::NUMPY: {
            const auto new_rank = std::max(dst_rank, src_rank);
            VectorDims dims(new_rank);
            bool success = true;
            for (int64_t i = 0; i < new_rank; i++) {
                auto dsti = i < (new_rank - dst_rank) ? 1 : dst[i - (new_rank - dst_rank)];
                auto srci = i < (new_rank - src_rank) ? 1 : src[i - (new_rank - src_rank)];
                success &= utils::broadcast_merge_dim(dims[i], dsti, srci);
            }
            dst = std::move(dims);
            return success;
        }
        case ov::op::AutoBroadcastType::PDPD: {
            int64_t axis = autob.m_axis;
            if (src_rank > dst_rank || axis < -1)
                return false;

            axis = (axis == -1) ? (dst_rank - src_rank) : axis;
            if (src_rank + axis > dst_rank)
                return false;

            bool success = true;
            for (int64_t i = 0; i < src_rank; ++i) {
                if (!utils::is_dynamic_value(dst[axis + i]) && !utils::is_dynamic_value(src[i])) {
                    if (src[i] > dst[axis + i])
                        return false;
                }
                success &= utils::broadcast_merge_dim(dst[axis + i], dst[axis + i], src[i]);
            }
            return success;
        }
        default:
            OPENVINO_THROW("Unsupported auto broadcast type: ", autob.m_type);
    }
    return false;
}
/*
 * Merge SRC to DST, no broadcasting is allowed
 */
bool merge_into(VectorDims& dst, const VectorDims& src) {
    auto merge_dim = [](size_t& dst, const size_t& d1, const size_t& d2) {
        if (d1 == d2 || utils::is_dynamic_value(d1)) {
            dst = d2;
        } else if (utils::is_dynamic_value(d2)) {
            dst = d1;
        } else {
            return false;
        }
        return true;
    };
    if (dst.size() != src.size())
        return false;

    bool success = true;
    for (size_t i = 0; i < dst.size(); i++)
        success &= merge_dim(dst[i], dst[i], src[i]);
    return success;
}

Result NumpyBroadcastShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
        OPENVINO_ASSERT(!input_shapes.empty(), "No input shapes were provided for NumpyBroadcastShapeInfer");
        auto output_shape = input_shapes[0].get();
        for (size_t i = 1; i < input_shapes.size(); i++) {
            OPENVINO_ASSERT(broadcast_merge_into(output_shape, input_shapes[i], ov::op::AutoBroadcastType::NUMPY),
                            "Failed to broadcast-merge input shapes in NumpyBroadcastShapeInfer");
        }
        return {{std::move(output_shape)}, ShapeInferStatus::success};
}

template<class BroadcastOP>
BroadcastShapeInfer<BroadcastOP>::BroadcastShapeInfer(const std::shared_ptr<Node>& n) {
        static_assert(std::is_base_of<snippets::op::BroadcastMove, BroadcastOP>() ||
                      std::is_base_of<snippets::op::BroadcastLoad, BroadcastOP>(),
                      "This ShapeInfer class could be used only for BroadcastMove and BroadcastLoad operations.");
        broadcast_op = as_type_ptr<BroadcastOP>(n);
        OPENVINO_ASSERT(broadcast_op, "Invalid node passed to BroadcastShapeInfer.",
                        "Expected ", typeid(BroadcastOP).name(), "got ", n->get_type_name());
}

template<class BroadcastOP>
Result BroadcastShapeInfer<BroadcastOP>::infer(const std::vector<VectorDimsRef>& input_shapes) {
    auto out_shape = input_shapes[0].get();
    const auto& bcasted_dim = broadcast_op->get_bcast_dimension();
    OPENVINO_ASSERT(bcasted_dim.is_static());
    out_shape.back() = bcasted_dim.get_length();
    return {{out_shape}, ShapeInferStatus::success};
}

//// Note: we need to manually create template instances here, so they can be reused in Broadcast* headers.
template class BroadcastShapeInfer<op::BroadcastMove>;
template class BroadcastShapeInfer<op::BroadcastLoad>;

SelectShapeInfer::SelectShapeInfer(const std::shared_ptr<Node>& n) {
    const auto& select = as_type_ptr<ov::op::v1::Select>(n);
    OPENVINO_ASSERT(select, "Invalid node passed to SelectShapeInfer.");
    m_broadcast_spec = select->get_auto_broadcast();
}

Result SelectShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 3, "Invalid number of shapes passed SelectShapeInfer");
    VectorDims result_shape;
    if (m_broadcast_spec == ov::op::AutoBroadcastType::PDPD) {
        result_shape = input_shapes[1];  // 'then' tensor
        // in PDPD type, Broadcast-merging 'else' into 'then' one way not each other.
        OPENVINO_ASSERT(broadcast_merge_into(result_shape, input_shapes[2], m_broadcast_spec),
                        "'Else' tensor shape is not broadcastable.");
        OPENVINO_ASSERT(broadcast_merge_into(result_shape, input_shapes[0], m_broadcast_spec),
                        "'Cond' tensor shape is not broadcastable.");
    } else {
        result_shape = input_shapes[2];
        for (int input_port = 1; input_port >= 0; input_port--) {
            if (m_broadcast_spec.m_type == ov::op::AutoBroadcastType::NONE) {
                OPENVINO_ASSERT(merge_into(result_shape, input_shapes[input_port]),
                                "Argument shapes are inconsistent.");
            } else if (m_broadcast_spec.m_type == ov::op::AutoBroadcastType::NUMPY) {
                OPENVINO_ASSERT(broadcast_merge_into(result_shape, input_shapes[input_port], m_broadcast_spec),
                                "Argument shapes are inconsistent.");
            } else {
                OPENVINO_THROW("Unsupported auto broadcast specification");
            }
        }
    }
    return {{result_shape}, ShapeInferStatus::success};
}

Result HorizonOpShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes in HorizonShapeInfer");
    auto output_shapes = input_shapes[0].get();
    if (!output_shapes.empty())
        output_shapes.back() = 1;
    return {{output_shapes}, ShapeInferStatus::success};
}

BrgemmShapeInfer::BrgemmShapeInfer(const std::shared_ptr<Node>& n) {
    for (const auto& in : n->inputs()) {
        const auto& port = lowered::PortDescriptorUtils::get_port_descriptor_ptr(in);
        m_io_layouts.push_back(port->get_layout());
    }
    const auto& port = lowered::PortDescriptorUtils::get_port_descriptor_ptr(n->output(0));
    m_io_layouts.push_back(port->get_layout());
}

Result BrgemmShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 2 || input_shapes.size() == 3, "BRGEMM expects 2 or 3 input shapes for shape inference");

    // Todo: Ideally we should use the layout stored in PortDescriptors. Can we do it?
    const auto& arg0_shape = ov::snippets::utils::get_planar_vdims(input_shapes[0].get(), m_io_layouts[0]);
    const auto& arg1_shape = ov::snippets::utils::get_planar_vdims(input_shapes[1].get(), m_io_layouts[1]);

    size_t arg0_rank = arg0_shape.size(), arg1_rank = arg1_shape.size();

    // temporary shapes to calculate output shape
    VectorDims arg0_shape_tmp(arg0_shape), arg1_shape_tmp(arg1_shape);

    // one-dimensional tensors unsqueezing is applied to each input independently.
    if (arg0_rank == 1) {
        // If the first input is 1D tensor, it is unsqueezed to 2D tensor (row vector)
        // by adding axes with size 1 at ROW_INDEX_DIM, to the left of the shape.
        // For example {S} will be reshaped to {1, S}.
        arg0_shape_tmp.insert(arg0_shape_tmp.begin(), 1);
        arg0_rank = arg0_shape_tmp.size();
    }
    if (arg1_rank == 1) {
        // If the second input is 1D tensor, it is unsqueezed to 2D tensor (column vector)
        // by adding axes with size 1 at COL_INDEX_DIM, to the right of the shape.
        // For example {S} will be reshaped to {S, 1}.
        arg1_shape_tmp.insert(arg1_shape_tmp.end(), 1);
        arg1_rank = arg1_shape_tmp.size();
    }

    // add 1 to begin to align shape ranks if needed
    if (arg0_rank < arg1_rank)
        arg0_shape_tmp.insert(arg0_shape_tmp.begin(), arg1_rank - arg0_rank, 1);
    else if (arg0_rank > arg1_rank)
        arg1_shape_tmp.insert(arg1_shape_tmp.begin(), arg0_rank - arg1_rank, 1);

    size_t max_rank = arg0_shape_tmp.size();
    VectorDims output_shape(max_rank);
    for (size_t i = 0; i < max_rank - 2; ++i) {
        if (!utils::broadcast_merge_dim(output_shape[i], arg0_shape_tmp[i], arg1_shape_tmp[i]))
            OPENVINO_THROW("Incompatible MatMul batch dimension. Can't merge dim ", arg0_shape_tmp[i],
                           " with dim ", arg1_shape_tmp[i], " at index=", i);
    }
    output_shape[output_shape.size() - 2] = arg0_shape_tmp[arg0_shape_tmp.size() - 2];  // M
    output_shape[output_shape.size() - 1] = arg1_shape_tmp[arg1_shape_tmp.size() - 1];  // N

    // removing the temporary axes from originally 1D tensors.
    if (arg0_shape.size() == 1) {
        output_shape.erase(output_shape.begin() + output_shape.size() - 2);
    }
    if (arg1_shape.size() == 1) {
        output_shape.erase(output_shape.begin() + output_shape.size() - 1);
    }
    output_shape = ov::snippets::utils::get_planar_vdims(output_shape, m_io_layouts.back());
    return {{output_shape}, snippets::ShapeInferStatus::success};
}

ReduceShapeInfer::ReduceShapeInfer(const std::shared_ptr<Node>& n) {
    const auto& reduce = as_type_ptr<ov::snippets::op::ReduceBase>(n);
    OPENVINO_ASSERT(reduce, "Invalid node passed to ReduceShapeInfer.");
    m_axis = reduce->get_axis();
}

Result ReduceShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 1, "Invalid number of shapes passed ReduceShapeInfer");
    VectorDims result_shape = input_shapes[0].get();
    result_shape[m_axis] = 1;
    return {{result_shape}, ShapeInferStatus::success};
}

ReshapeShapeInfer::ReshapeShapeInfer(const std::shared_ptr<Node>& n) {
    const auto& reshape = as_type_ptr<ov::snippets::op::Reshape>(n);
    OPENVINO_ASSERT(reshape, "Invalid node passed to ReshapeShapeInfer.");
    const auto& partial_shape = reshape->get_target_shape();
    OPENVINO_ASSERT(partial_shape.is_static(), "target_shape of reshape op should be static in ReshapeShapeInfer");
    target_shape = partial_shape.get_shape();
    target_shape_volume = utils::get_shape_size(target_shape);
}

Result ReshapeShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 1, "Invalid number of shapes is passed in ReshapeShapeInfer");
    const auto input_shape_volume = utils::get_shape_size(input_shapes[0].get());
    OPENVINO_ASSERT(input_shape_volume == target_shape_volume, "Tensor volume should be the same after reshape in ReshapeShapeInfer");

    return {{target_shape}, ShapeInferStatus::success};
}

} // namespace snippets
} // namespace ov
