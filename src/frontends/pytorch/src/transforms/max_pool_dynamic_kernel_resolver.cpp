// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "max_pool_dynamic_kernel_resolver.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"
// After utils.hpp: op/max_poolnd.hpp opens ns ...::pytorch::op, but utils.hpp uses unqualified
// op:: (== ov::op).
#include "op/max_poolnd.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;

namespace {

// Deferred max_pool op-types emitted as PtFrameworkNode by translate_max_pool_base (mirrors the
// TorchScript and FX op_table entries).
const std::initializer_list<std::string> MAX_POOL_TYPES = {"aten::max_pool1d",
                                                           "aten::max_pool1d_with_indices",
                                                           "aten::max_pool2d",
                                                           "aten::max_pool2d_with_indices",
                                                           "aten::max_pool3d",
                                                           "aten::max_pool3d_with_indices",
                                                           "aten.max_pool2d_with_indices.default",
                                                           "aten.max_pool3d_with_indices.default"};

// Recover the spatial rank (1/2/3) from the op-type string.
int dims_from_op_type(const std::string& op_type) {
    if (op_type.find("max_pool1d") != std::string::npos)
        return 1;
    if (op_type.find("max_pool2d") != std::string::npos)
        return 2;
    return 3;
}

// True when input `index` is absent or an explicit `none` node (an omitted optional argument).
bool input_is_none_or_missing(const std::shared_ptr<ov::op::util::FrameworkNode>& fw, size_t index) {
    if (index >= fw->get_input_size())
        return true;
    return is_none_node(fw->input_value(index));
}

// Fold an optional list attribute (stride/padding/dilation) to i64. Empty vector when the input is
// absent/None/empty ("use the default"). Raises OpConversionFailure when present but not
// constant-foldable: a runtime stride/padding/dilation fits neither lowering, and treating it as
// the default would miscompute. `attr_name` names the attribute in the message.
std::vector<int64_t> fold_optional_list(const std::shared_ptr<ov::op::util::FrameworkNode>& fw,
                                        size_t index,
                                        int dims,
                                        const char* attr_name) {
    if (input_is_none_or_missing(fw, index))
        return {};
    auto src = fw->input_value(index);
    if (is_empty_list(src))
        return {};
    auto folded = ov::util::get_constant_from_source(concat_list_construct(src));
    PYTORCH_OP_CONVERSION_CHECK(folded,
                                "aten::max_pool",
                                dims,
                                "d with a non-constant ",
                                attr_name,
                                " is not supported.");
    return folded->cast_vector<int64_t>();
}

// Fold the optional ceil_mode flag. False (the default) when absent/None; raises OpConversionFailure
// when present but not constant-foldable (a runtime ceil_mode fits neither lowering).
bool fold_ceil_mode(const std::shared_ptr<ov::op::util::FrameworkNode>& fw, size_t index, int dims) {
    if (input_is_none_or_missing(fw, index))
        return false;
    auto ceil_c = ov::util::get_constant_from_source(fw->input_value(index));
    PYTORCH_OP_CONVERSION_CHECK(ceil_c, "aten::max_pool", dims, "d with a non-constant ceil_mode is not supported.");
    return ceil_c->cast_vector<bool>()[0];
}

// Runtime-assert that spatial axis `spatial_idx` (within the trailing `dims` axes) has extent `k`:
// a Reshape copying every axis but pinning this one to `k`. Identity when extent == k; a strided
// pool (k < extent) makes the element counts disagree, so the Reshape fails loudly.
Output<Node> guard_full_extent_axis(ov::pass::NodeRegistry& rg,
                                    const Output<Node>& tensor,
                                    int dims,
                                    int spatial_idx,
                                    const Output<Node>& k) {
    auto zero = rg.make<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto one = rg.make<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto neg_dims = rg.make<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-dims});
    auto shape = rg.make<v3::ShapeOf>(tensor, element::i64);
    // Leading (non-pooled) axes, copied verbatim; empty when rank == dims (Concat handles it).
    auto batch = rg.make<v8::Slice>(shape, zero, neg_dims, one, zero);
    Output<Node> k_i64 = k;
    if (k_i64.get_element_type() != element::i64) {
        k_i64 = rg.make<v0::Convert>(k_i64, element::i64);
    }
    // Reshape the scalar kernel element to 1-D [1] so it slots into the shape.
    auto k_1d = rg.make<v1::Reshape>(k_i64, one, false);
    OutputVector parts{batch};
    for (int t = 0; t < dims; ++t) {
        if (t == spatial_idx) {
            parts.push_back(k_1d);
        } else {
            // Copy this axis's current extent (negative index works for dynamic rank).
            auto idx = rg.make<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{t - dims});
            parts.push_back(rg.make<v8::Gather>(shape, idx, zero));
        }
    }
    auto target = rg.make<v0::Concat>(parts, 0);
    auto guarded = rg.make<v1::Reshape>(tensor, target, false);
    // The CPU plugin embeds this name in the runtime error.
    guarded->set_friendly_name("aten::max_pool" + std::to_string(dims) + "d/require_full_extent");
    return guarded;
}

// Decompose a max_pool whose kernel_size is only known at runtime. MaxPool takes the window as
// constructor attributes, so a runtime kernel (e.g. F.max_pool2d(x, [1, x.size(3)])) is handled
// only as a global pool over a full spatial axis -- a ReduceMax (keep_dims=True to match MaxPool).
// Per axis: elem_const_val 1 = window of 1 (identity), non-constant = full-extent pool (reduce).
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
    // Exact only for a full-extent pool with default placement: stride=kernel, no pad, dilation 1,
    // ceil_mode=False.
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

    // Per axis: reduce it (full-extent) or leave it (window 1). Each runtime element is guarded to
    // equal the axis extent first, so a strided pool (kernel < extent) fails loudly.
    Output<Node> guarded = input;
    std::vector<int64_t> reduce_axes;
    for (int i = 0; i < dims; ++i) {
        const int64_t axis = static_cast<int64_t>(i) - dims;  // negative index of this spatial axis
        if (!elem_is_const[i]) {
            // Runtime element: must span the whole axis (global pool).
            guarded = guard_full_extent_axis(rg, guarded, dims, i, elem_runtime_val[i]);
            reduce_axes.push_back(axis);
        } else if (elem_const_val[i] == 1) {
            continue;  // window of 1 (default stride) is an identity
        } else {
            // A static window > 1 is a sliding-window pool a ReduceMax cannot represent.
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

}  // namespace

MaxPoolDynamicKernelResolver::MaxPoolDynamicKernelResolver() {
    auto fw_pattern = ov::pass::pattern::wrap_type<ov::op::util::FrameworkNode>(fw_node_predicate(MAX_POOL_TYPES));

    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& m) {
        auto fw_node = ov::as_type_ptr<ov::op::util::FrameworkNode>(m.get_match_root());
        if (!fw_node) {
            return false;
        }
        const auto& attrs = fw_node->get_attrs();
        const auto op_type_it = attrs.find(PtFrameworkNode::op_type_key);
        if (op_type_it == attrs.end()) {
            return false;
        }
        const std::string op_type = op_type_it->second;
        const int dims = dims_from_op_type(op_type);
        const bool return_indices = fw_node->get_output_size() == 2;

        // Re-probe the kernel_size now that shapes have propagated. get_list_as_outputs handles both
        // the SequenceMark form and a list already lowered to elementwise outputs.
        std::vector<bool> elem_is_const;
        std::vector<int64_t> elem_const_val;
        std::vector<Output<Node>> elem_runtime_val;
        bool kernel_is_static = true;
        for (const auto& e : get_list_as_outputs(fw_node->input_value(1))) {
            if (const auto c = ov::util::get_constant_from_source(e)) {
                for (auto val : c->cast_vector<int64_t>()) {
                    elem_is_const.push_back(true);
                    elem_const_val.push_back(val);
                    elem_runtime_val.emplace_back();
                }
            } else {
                elem_is_const.push_back(false);
                elem_const_val.push_back(-1);
                elem_runtime_val.push_back(e);
                kernel_is_static = false;
            }
        }
        // A scalar kernel_size applies to every axis. Copy element 0 into locals first:
        // assign(dims, elem_runtime_val[0]) would alias into the vector as it reallocates (UB).
        if (elem_is_const.size() == 1 && dims > 1) {
            const bool c0 = elem_is_const[0];
            const int64_t v0 = elem_const_val[0];
            const Output<Node> r0 = elem_runtime_val[0];
            elem_is_const.assign(dims, c0);
            elem_const_val.assign(dims, v0);
            elem_runtime_val.assign(dims, r0);
        }

        ov::pass::NodeRegistry rg;
        OutputVector new_outputs;

        // Folding the optional attributes and the ReduceMax guards raise OpConversionFailure for a
        // config neither lowering can honor. Route it to the standard unconverted-ops reporter
        // instead of letting it escape run_passes.
        try {
            if (kernel_is_static) {
                // Kernel now fully constant -> build the static MaxPool.
                Shape kernel;
                for (auto v : elem_const_val)
                    kernel.push_back(static_cast<size_t>(v));

                auto stride_vals = fold_optional_list(fw_node, 2, dims, "stride");
                Strides strides;
                if (stride_vals.empty()) {
                    strides = kernel;  // default stride is kernel
                } else {
                    for (auto v : stride_vals)
                        strides.push_back(static_cast<size_t>(v));
                }
                auto pad_vals = fold_optional_list(fw_node, 3, dims, "padding");
                Shape pads;
                if (pad_vals.empty()) {
                    pads = Shape(kernel.size(), 0);
                } else {
                    for (auto v : pad_vals)
                        pads.push_back(static_cast<size_t>(v));
                }
                auto dil_vals = fold_optional_list(fw_node, 4, dims, "dilation");
                Strides dilations(dims, 1);
                if (!dil_vals.empty()) {
                    dilations.clear();
                    for (auto v : dil_vals)
                        dilations.push_back(static_cast<size_t>(v));
                }
                RoundingType rounding_type =
                    fold_ceil_mode(fw_node, 5, dims) ? RoundingType::CEIL_TORCH : RoundingType::FLOOR;
                new_outputs = op::build_static_max_pool(rg,
                                                        fw_node->input_value(0),
                                                        dims,
                                                        return_indices,
                                                        kernel,
                                                        strides,
                                                        pads,
                                                        dilations,
                                                        rounding_type);
            } else {
                // Kernel still dynamic -> the ReduceMax full-extent decomposition (with guards).
                bool stride_is_default = input_is_none_or_missing(fw_node, 2) || is_empty_list(fw_node->input_value(2));
                auto pads = fold_optional_list(fw_node, 3, dims, "padding");
                auto dilations = fold_optional_list(fw_node, 4, dims, "dilation");
                bool ceil_mode = fold_ceil_mode(fw_node, 5, dims);
                new_outputs = build_dynamic_kernel_max_pool(rg,
                                                            dims,
                                                            return_indices,
                                                            fw_node->input_value(0),
                                                            elem_is_const,
                                                            elem_const_val,
                                                            elem_runtime_val,
                                                            stride_is_default,
                                                            pads,
                                                            dilations,
                                                            ceil_mode);
            }
        } catch (const std::exception& e) {
            // Unsupported config: annotate the placeholder so the unconverted-ops reporter surfaces it.
            add_exception_to_fw_node(fw_node, e.what());
            return false;
        }

        // copy_runtime_info (not _and_name) so the guard nodes keep their op-labeled friendly names
        // (e.g. aten::max_poolNd/require_full_extent, asserted by the runtime-guard test).
        ov::copy_runtime_info(fw_node, rg.get());
        replace_node(fw_node, new_outputs);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fw_pattern,
                                                          "ov::frontend::pytorch::pass::MaxPoolDynamicKernelResolver");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
