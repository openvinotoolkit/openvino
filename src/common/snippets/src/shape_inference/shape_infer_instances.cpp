// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "snippets/shape_inference/shape_infer_instances.hpp"
#include "snippets/snippets_isa.hpp"
#include "openvino/op/select.hpp"
namespace ov {
namespace snippets {
using Result = IShapeInferSnippets::Result;
/*
 * Merge SRC to DST with broadcasting rules defined by the Autobroadcast specifier
 */
bool broadcast_merge_into(VectorDims& dst, const VectorDims& src, const ov::op::AutoBroadcastSpec& autob) {
    auto broadcast_merge_dim = [](size_t& dst, const size_t& d1, const size_t& d2) {
        if (d1 == d2 || d1 == 1 || d1 == IShapeInferSnippets::DYNAMIC_DIMENSION) {
            dst = d2;
        } else if (d2 == 1 || d2 == IShapeInferSnippets::DYNAMIC_DIMENSION) {
            dst = d1;
        } else {
           return false;
        }
        return true;
    };
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
                success &= broadcast_merge_dim(dims[i], dsti, srci);
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
                if (dst[axis + i] != IShapeInferSnippets::DYNAMIC_DIMENSION &&
                    src[i] != IShapeInferSnippets::DYNAMIC_DIMENSION) {
                    if (src[i] > dst[axis + i])
                        return false;
                }
                success &= broadcast_merge_dim(dst[axis + i], dst[axis + i], src[i]);
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
        if (d1 == d2 || d1 == IShapeInferSnippets::DYNAMIC_DIMENSION) {
            dst = d2;
        } else if (d2 == IShapeInferSnippets::DYNAMIC_DIMENSION) {
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
        const auto& broadcast = as_type_ptr<BroadcastOP>(n);
        OPENVINO_ASSERT(broadcast, "Invalid node passed to BroadcastShapeInfer.",
                        "Expected ", typeid(BroadcastOP).name(), "got ", n->get_type_name());
        const auto last_dim = *broadcast->get_output_shape().rbegin();
        m_broadcasted_dim = last_dim.is_dynamic() ? IShapeInferSnippets::DYNAMIC_DIMENSION : last_dim.get_length();
}
template<class BroadcastOP>
Result BroadcastShapeInfer<BroadcastOP>::infer(const std::vector<VectorDimsRef>& input_shapes) {
    auto out_shape = input_shapes[0].get();
    out_shape.back() = m_broadcasted_dim;
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

} // namespace snippets
} // namespace ov
