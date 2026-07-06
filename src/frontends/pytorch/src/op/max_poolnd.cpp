// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"
// Included after utils.hpp: this header opens ns ov::frontend::pytorch::op, and utils.hpp uses
// unqualified op:: expecting ov::op, so it must be parsed before that namespace becomes visible.
#include "max_poolnd.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {

// Assert at runtime that spatial axis `spatial_idx` (0-based within the trailing `dims` axes) has
// extent exactly `k`: a Reshape whose target copies every axis from the runtime shape but pins this
// one to `k`. When extent == k it is an identity; any other extent (a strided pool with k < extent)
// makes the element counts disagree and the Reshape fails loudly instead of collapsing the axis to 1.
Output<Node> guard_full_extent_axis(ov::pass::NodeRegistry& rg,
                                    const Output<Node>& tensor,
                                    int dims,
                                    int spatial_idx,
                                    const Output<Node>& k) {
    auto zero = rg.make<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto one = rg.make<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto neg_dims = rg.make<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-dims});
    auto shape = rg.make<v3::ShapeOf>(tensor, element::i64);
    // Leading (batch / channel / earlier-spatial) axes, copied verbatim; empty when there is no
    // batch axis (rank == dims), which Concat handles.
    auto batch = rg.make<v8::Slice>(shape, zero, neg_dims, one, zero);
    Output<Node> k_i64 = k;
    if (k_i64.get_element_type() != element::i64) {
        k_i64 = rg.make<v0::Convert>(k_i64, element::i64);
    }
    // Normalize the scalar kernel element to a 1-D [1] so it slots into the shape.
    auto k_1d = rg.make<v1::Reshape>(k_i64, one, false);
    OutputVector parts{batch};
    for (int t = 0; t < dims; ++t) {
        if (t == spatial_idx) {
            parts.push_back(k_1d);
        } else {
            // Copy this spatial axis's current extent (negative index works for dynamic rank).
            auto idx = rg.make<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{t - dims});
            parts.push_back(rg.make<v8::Gather>(shape, idx, zero));
        }
    }
    auto target = rg.make<v0::Concat>(parts, 0);
    auto guarded = rg.make<v1::Reshape>(tensor, target, false);
    // Op-labeled name: the CPU plugin embeds the node name in the runtime error.
    guarded->set_friendly_name("aten::max_pool" + std::to_string(dims) + "d/require_full_extent");
    return guarded;
}

}  // namespace

// Decompose a max_pool whose kernel_size is only known at runtime. OpenVINO's MaxPool takes the
// window as constructor attributes, so a runtime-computed kernel (e.g. F.max_pool2d(x, [1,
// x.size(3)])) is only handled as a "global" pool over the full extent of a spatial axis -- exactly
// a ReduceMax (keep_dims=True to match MaxPool). Any other configuration is rejected.
// Per spatial axis, elem_is_const/elem_const_val classify the kernel element: constant 1 = window
// of 1 (identity), non-constant = full-extent pool (reduce, via negative axes for any rank/batch).
OutputVector build_dynamic_kernel_max_pool(ov::pass::NodeRegistry& rg,
                                           int dims,
                                           bool return_indices,
                                           const Output<Node>& input,
                                           const std::vector<bool>& elem_is_const,
                                           const std::vector<int64_t>& elem_const_val,
                                           const std::vector<Output<Node>>& elem_runtime_val,
                                           bool stride_is_default,
                                           const std::vector<int64_t>& pads,
                                           const std::vector<int64_t>& dilations,
                                           bool ceil_mode) {
    PYTORCH_OP_CONVERSION_CHECK(static_cast<int>(elem_is_const.size()) == dims,
                                "aten::max_pool",
                                dims,
                                "d: could not interpret the non-constant kernel_size (expected ",
                                dims,
                                " spatial entries).");
    PYTORCH_OP_CONVERSION_CHECK(!return_indices,
                                "aten::max_pool",
                                dims,
                                "d with a non-constant kernel_size and return_indices=True is not supported.");
    // The decomposition is exact only for a full-extent pool with the default window placement:
    // default stride (=kernel), zero padding, dilation 1, ceil_mode=False.
    PYTORCH_OP_CONVERSION_CHECK(stride_is_default,
                                "aten::max_pool",
                                dims,
                                "d with a non-constant kernel_size is only supported with the default stride "
                                "(stride=kernel_size).");
    for (auto p : pads) {
        PYTORCH_OP_CONVERSION_CHECK(p == 0,
                                    "aten::max_pool",
                                    dims,
                                    "d with a non-constant kernel_size is only supported with zero padding.");
    }
    for (auto d : dilations) {
        PYTORCH_OP_CONVERSION_CHECK(d == 1,
                                    "aten::max_pool",
                                    dims,
                                    "d with a non-constant kernel_size is only supported with dilation 1.");
    }
    PYTORCH_OP_CONVERSION_CHECK(!ceil_mode,
                                "aten::max_pool",
                                dims,
                                "d with a non-constant kernel_size is only supported with ceil_mode=False.");

    PYTORCH_OP_CONVERSION_CHECK(static_cast<int>(elem_runtime_val.size()) == dims,
                                "aten::max_pool",
                                dims,
                                "d: could not recover the runtime kernel_size elements.");

    // Per spatial axis: reduce it (full-extent) or leave it (window 1). Each runtime kernel element
    // is guarded to equal the axis extent first, so a strided pool (kernel < extent) fails loudly.
    Output<Node> guarded = input;
    std::vector<int64_t> reduce_axes;
    for (int i = 0; i < dims; ++i) {
        const int64_t axis = static_cast<int64_t>(i) - dims;  // negative index of this spatial axis
        if (!elem_is_const[i]) {
            // Runtime kernel element: must span the whole axis (global pool).
            guarded = guard_full_extent_axis(rg, guarded, dims, i, elem_runtime_val[i]);
            reduce_axes.push_back(axis);
        } else if (elem_const_val[i] == 1) {
            // Window of 1 with default stride is an identity along this axis.
            continue;
        } else {
            // A static window > 1 next to a dynamic axis is a sliding-window pool a ReduceMax cannot represent.
            PYTORCH_OP_CONVERSION_CHECK(false,
                                        "aten::max_pool",
                                        dims,
                                        "d with a non-constant kernel_size is only supported when every pooled axis "
                                        "spans its full extent (the kernel along statically-sized axes must be 1).");
        }
    }
    PYTORCH_OP_CONVERSION_CHECK(!reduce_axes.empty(),
                                "aten::max_pool",
                                dims,
                                "d: a non-constant kernel_size that pools no axis is unexpected.");

    auto axes = rg.make<v0::Constant>(element::i64, Shape{reduce_axes.size()}, reduce_axes);
    auto res = rg.make<v1::ReduceMax>(guarded, axes, /*keep_dims=*/true);
    return {res};
}

OutputVector build_static_max_pool(ov::pass::NodeRegistry& rg,
                                   Output<Node> input,
                                   int dims,
                                   bool return_indices,
                                   const Shape& kernel,
                                   const Strides& strides,
                                   const Shape& pads,
                                   const Strides& dilations,
                                   RoundingType rounding_type) {
    auto input_shape = rg.make<v3::ShapeOf>(input);

    auto const_0 = v0::Constant::create(element::i64, Shape{1}, {0});
    auto const_1 = v0::Constant::create(element::i64, Shape{1}, {1});
    bool is_static = input.get_partial_shape().rank().is_static();
    bool no_batch_dim = is_static && input.get_partial_shape().rank().get_length() == dims + 1;

    if (is_static) {
        if (no_batch_dim) {
            input = rg.make<v0::Unsqueeze>(input, const_0);
        }
    } else {
        input = rg.make<v0::Unsqueeze>(input, const_0);
        auto unsqueeze_shape = rg.make<v3::ShapeOf>(input);
        auto rank = rg.make<v0::ShapeOf>(unsqueeze_shape);
        auto end_index = rg.make<v1::Add>(rank, const_1);
        auto start_index = v0::Constant::create(element::i64, Shape{1}, {-dims - 2});
        auto reshape_pattern = rg.make<v8::Slice>(unsqueeze_shape, start_index, end_index, const_1, const_0);
        input = rg.make<v1::Reshape>(input, reshape_pattern, true);
    }

    auto res = rg.make<v14::MaxPool>(input,
                                     strides,
                                     dilations,
                                     pads,
                                     pads,
                                     kernel,
                                     rounding_type,
                                     PadType::EXPLICIT,
                                     element::i64,
                                     2);
    if (is_static) {
        if (no_batch_dim) {
            if (return_indices) {
                auto out1 = res->output(0);
                auto out2 = res->output(1);
                out1 = rg.make<v0::Squeeze>(out1, const_0);
                out2 = rg.make<v0::Squeeze>(out2, const_0);
                return {out1, out2};
            } else {
                auto squeezed = rg.make<v0::Squeeze>(res, const_0);
                return {squeezed};
            }
        } else {
            if (return_indices) {
                return {res->output(0), res->output(1)};
            } else {
                return {res};
            }
        }

    } else {
        auto pooled_output_shape = rg.make<v3::ShapeOf>(res);

        auto start_index_input = v0::Constant::create(element::i64, Shape{1}, {-dims});
        auto slice_input_shape = rg.make<v8::Slice>(input_shape, const_0, start_index_input, const_1, const_0);

        auto start_index_pooled = v0::Constant::create(element::i64, Shape{1}, {-dims});
        auto end_index_pooled = v0::Constant::create(element::i64, Shape{1}, {2 + dims});
        auto slice_pooled_output_shape =
            rg.make<v8::Slice>(pooled_output_shape, start_index_pooled, end_index_pooled, const_1, const_0);

        auto concat_shape = rg.make<v0::Concat>(OutputVector{slice_input_shape, slice_pooled_output_shape}, 0);
        if (return_indices) {
            auto out1 = res->output(0);
            auto out2 = res->output(1);
            out1 = rg.make<v1::Reshape>(out1, concat_shape, true);
            out2 = rg.make<v1::Reshape>(out2, concat_shape, true);
            return {out1, out2};
        } else {
            auto reshaped = rg.make<v1::Reshape>(res, concat_shape, true);
            return {reshaped};
        }
    }
}

OutputVector translate_max_pool_base(const NodeContext& context, int dims, bool return_indices) {
    num_inputs_check(context, 2, 6);
    auto input = context.get_input(0);

    // The MaxPool kernel is a constructor attribute (must be known at conversion time), but PyTorch
    // can build kernel_size from runtime values (e.g. [1, x.size(3)]). Probe the kernel elements; if
    // any is non-constant at translation time, defer the decision to MaxPoolDynamicKernelResolver: a
    // dynamic size can become static once shapes propagate (e.g. convert_model(input=...)), in which
    // case it lowers to the ordinary static MaxPool; otherwise it falls back to a ReduceMax.
    bool kernel_is_static = true;
    for (const auto& e : get_list_as_outputs(context.get_input(1))) {
        if (!ov::util::get_constant_from_source(e)) {
            kernel_is_static = false;
            break;
        }
    }
    if (!kernel_is_static) {
        // The base always yields return_indices ? 2 : 1 outputs; keep that arity on the placeholder
        // so downstream consumers (and the FX list-construct wrapper) see the expected ports.
        const size_t num_outputs = return_indices ? 2 : 1;
        auto fw_node = std::make_shared<PtFrameworkNode>(context.get_decoder(), context.inputs(), num_outputs);
        context.mark_node(fw_node);
        return fw_node->outputs();
    }

    // All kernel elements are constant -> build the static MaxPool directly.
    auto kernel = context.const_input<Shape>(1);
    Strides strides;
    if (!context.input_is_none(2)) {
        strides = context.const_input<Strides>(2);
    }
    if (context.input_is_none(2) || strides.size() == 0) {
        // In case strides are not provided default is kernel
        strides = kernel;
    }
    Shape pads;
    if (context.input_is_none(3)) {
        pads = Shape(kernel.size(), 0);
    } else {
        pads = context.const_input<Shape>(3);  // pytorch supports only symmetric paddings
    }
    auto dilations = Strides(dims, 1);
    if (!context.input_is_none(4)) {
        dilations = context.const_input<Strides>(4);
    }
    RoundingType rounding_type;
    if (context.input_is_none(5)) {
        rounding_type = RoundingType::FLOOR;
    } else {
        rounding_type = context.const_input<bool>(5) ? RoundingType::CEIL_TORCH : RoundingType::FLOOR;
    }

    ov::pass::NodeRegistry rg;
    auto res = build_static_max_pool(rg, input, dims, return_indices, kernel, strides, pads, dilations, rounding_type);
    context.mark_nodes(rg.get());
    return res;
};

OutputVector translate_max_pool1d(const NodeContext& context) {
    return translate_max_pool_base(context, 1, context.get_output_size() == 2);
};

OutputVector translate_max_pool2d(const NodeContext& context) {
    return translate_max_pool_base(context, 2, context.get_output_size() == 2);
};

OutputVector translate_max_pool3d(const NodeContext& context) {
    return translate_max_pool_base(context, 3, context.get_output_size() == 2);
};

OutputVector translate_max_pool2d_fx(const NodeContext& context) {
    auto output = translate_max_pool_base(context, 2, true);
    return {context.mark_node(make_list_construct(output))};
};

OutputVector translate_max_pool3d_fx(const NodeContext& context) {
    auto output = translate_max_pool_base(context, 3, true);
    return {context.mark_node(make_list_construct(output))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
