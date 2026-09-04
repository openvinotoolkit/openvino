// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "snippets/shape_inference/shape_infer_instances.hpp"

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "select_shape_inference.hpp"
#include "snippets/lowered/port_descriptor.hpp"
#include "snippets/op/broadcastload.hpp"
#include "snippets/op/broadcastmove.hpp"
#include "snippets/op/reduce.hpp"
#include "snippets/shape_inference/shape_inference.hpp"
#include "snippets/shape_types.hpp"
#include "snippets/utils/utils.hpp"

namespace ov::snippets {
using Result = IShapeInferSnippets::Result;

namespace {
class VectorDimsAdapter {
public:
    using ShapeContainer = VectorDimsAdapter;

    VectorDimsAdapter() = default;
    explicit VectorDimsAdapter(VectorDims dims) : m_dims(std::move(dims)) {}

    static bool merge_into(VectorDimsAdapter& dst, const VectorDimsAdapter& src) {
        auto dst_shape = utils::vdims_to_pshape(dst.m_dims);
        const auto success = PartialShape::merge_into(dst_shape, utils::vdims_to_pshape(src.m_dims));
        dst.m_dims = to_vector_dims(dst_shape);
        return success;
    }

    static bool broadcast_merge_into(VectorDimsAdapter& dst,
                                     const VectorDimsAdapter& src,
                                     const ov::op::AutoBroadcastSpec& autob) {
        auto dst_shape = utils::vdims_to_pshape(dst.m_dims);
        const auto success = PartialShape::broadcast_merge_into(dst_shape, utils::vdims_to_pshape(src.m_dims), autob);
        dst.m_dims = to_vector_dims(dst_shape);
        return success;
    }

    VectorDims&& take_dims() {
        return std::move(m_dims);
    }

private:
    static VectorDims to_vector_dims(const PartialShape& shape) {
        VectorDims dims;
        dims.reserve(shape.size());
        std::transform(shape.begin(), shape.end(), std::back_inserter(dims), utils::dimension_to_size_t);
        return dims;
    }

    VectorDims m_dims;
};
}  // namespace

/*
 * Merge SRC to DST with broadcasting rules defined by the Autobroadcast specifier
 */
bool broadcast_merge_into(VectorDims& dst, const VectorDims& src, const ov::op::AutoBroadcastSpec& autob) {
    VectorDimsAdapter dst_adapter(dst);
    const auto success = VectorDimsAdapter::broadcast_merge_into(dst_adapter, VectorDimsAdapter(src), autob);
    dst = dst_adapter.take_dims();
    return success;
}
/*
 * Merge SRC to DST, no broadcasting is allowed
 */
bool merge_into(VectorDims& dst, const VectorDims& src) {
    VectorDimsAdapter dst_adapter(dst);
    const auto success = VectorDimsAdapter::merge_into(dst_adapter, VectorDimsAdapter(src));
    dst = dst_adapter.take_dims();
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

template <class BroadcastOP>
BroadcastShapeInfer<BroadcastOP>::BroadcastShapeInfer(const std::shared_ptr<Node>& n)
    : broadcast_op(as_type_ptr<BroadcastOP>(n)) {
    static_assert(std::is_base_of<snippets::op::BroadcastMove, BroadcastOP>() ||
                      std::is_base_of<snippets::op::BroadcastLoad, BroadcastOP>(),
                  "This ShapeInfer class could be used only for BroadcastMove and BroadcastLoad operations.");

    OPENVINO_ASSERT(broadcast_op,
                    "Invalid node passed to BroadcastShapeInfer.",
                    "Expected ",
                    typeid(BroadcastOP).name(),
                    "got ",
                    n->get_type_name());
}

template <class BroadcastOP>
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
    ov::op::v1::Select select;
    select.set_auto_broadcast(m_broadcast_spec);

    std::vector<VectorDimsAdapter> adapted_input_shapes;
    adapted_input_shapes.reserve(input_shapes.size());
    for (const auto& input_shape : input_shapes) {
        adapted_input_shapes.emplace_back(input_shape.get());
    }

    auto output_shapes = ov::op::v1::shape_infer(&select, adapted_input_shapes);
    return {{{output_shapes[0].take_dims()}}, ShapeInferStatus::success};
}

Result HorizonOpShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() == 1, "Got invalid number of input shapes in HorizonShapeInfer");
    auto output_shapes = input_shapes[0].get();
    if (!output_shapes.empty()) {
        output_shapes.back() = 1;
    }
    return {{output_shapes}, ShapeInferStatus::success};
}

BrgemmShapeInfer::BrgemmShapeInfer(const std::shared_ptr<Node>& n) {
    // Only first 2 inputs are used for shape inference
    for (size_t i = 0; i < 2; ++i) {
        const auto& in = n->input(i);
        const auto& port = lowered::PortDescriptorUtils::get_port_descriptor_ptr(in);
        m_io_layouts.push_back(port->get_layout());
    }
    const auto& port = lowered::PortDescriptorUtils::get_port_descriptor_ptr(n->output(0));
    m_io_layouts.push_back(port->get_layout());
}

Result BrgemmShapeInfer::infer(const std::vector<VectorDimsRef>& input_shapes) {
    OPENVINO_ASSERT(input_shapes.size() >= 2, "Unexpected input_shapes count");

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
    if (arg0_rank < arg1_rank) {
        arg0_shape_tmp.insert(arg0_shape_tmp.begin(), arg1_rank - arg0_rank, 1);
    } else if (arg0_rank > arg1_rank) {
        arg1_shape_tmp.insert(arg1_shape_tmp.begin(), arg0_rank - arg1_rank, 1);
    }

    size_t max_rank = arg0_shape_tmp.size();
    VectorDims output_shape(max_rank);
    for (size_t i = 0; i < max_rank - 2; ++i) {
        if (!utils::broadcast_merge_dim(output_shape[i], arg0_shape_tmp[i], arg1_shape_tmp[i])) {
            OPENVINO_THROW("Incompatible MatMul batch dimension. Can't merge dim ",
                           arg0_shape_tmp[i],
                           " with dim ",
                           arg1_shape_tmp[i],
                           " at index=",
                           i);
        }
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

}  // namespace ov::snippets
