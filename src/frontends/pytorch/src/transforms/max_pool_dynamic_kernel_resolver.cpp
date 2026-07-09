// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "max_pool_dynamic_kernel_resolver.hpp"

#include <optional>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

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

// Fold an optional list attribute (stride/padding/dilation) to i64. Returns an (possibly empty)
// vector when foldable -- empty means the input is absent/None/empty ("use the default"). Returns
// nullopt when the input is present but not constant-foldable: a runtime stride/padding/dilation
// fits neither lowering, so the caller rejects the op (leaving the framework node) instead.
std::optional<std::vector<int64_t>> fold_optional_list(const std::shared_ptr<ov::op::util::FrameworkNode>& fw,
                                                       size_t index) {
    if (input_is_none_or_missing(fw, index))
        return std::vector<int64_t>{};
    auto src = fw->input_value(index);
    if (is_empty_list(src))
        return std::vector<int64_t>{};
    auto folded = ov::util::get_constant_from_source(concat_list_construct(src));
    if (!folded)
        return std::nullopt;
    return folded->cast_vector<int64_t>();
}

// Fold the optional ceil_mode flag. False (the default) when absent/None; nullopt when present but
// not constant-foldable (a runtime ceil_mode fits neither lowering).
std::optional<bool> fold_ceil_mode(const std::shared_ptr<ov::op::util::FrameworkNode>& fw, size_t index) {
    if (input_is_none_or_missing(fw, index))
        return false;
    auto ceil_c = ov::util::get_constant_from_source(fw->input_value(index));
    if (!ceil_c)
        return std::nullopt;
    return ceil_c->cast_vector<bool>()[0];
}

// Windowed max over one spatial axis with a runtime window `k` and stride `s` (no padding, dilation 1,
// floor rounding). `axis` is the negative index of the spatial axis. Builds a (out, k) gather-index
// matrix idx[j, w] = j*s + w with out = (L - k)/s + 1, gathers the windows along `axis` (inserting a
// new window axis right after it), then ReduceMax over that window axis -- exact, no Loop. When
// k == extent and s == k this reduces to a global pool over the axis (out = 1).
Output<Node> windowed_max_axis(ov::pass::NodeRegistry& rg,
                               const Output<Node>& tensor,
                               int dims,
                               int spatial_idx,
                               const Output<Node>& k,
                               const Output<Node>& s) {
    const int64_t axis = static_cast<int64_t>(spatial_idx) - dims;  // negative index of this axis
    auto axis_c = rg.make<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{axis});
    auto axis_s = rg.make<v0::Constant>(element::i64, Shape{}, std::vector<int64_t>{axis});
    auto zero = rg.make<v0::Constant>(element::i64, Shape{}, std::vector<int64_t>{0});
    auto one = rg.make<v0::Constant>(element::i64, Shape{}, std::vector<int64_t>{1});

    auto to_i64_scalar = [&](const Output<Node>& v) -> Output<Node> {
        Output<Node> x = v;
        if (x.get_element_type() != element::i64) {
            x = rg.make<v0::Convert>(x, element::i64);
        }
        // Reshape to a scalar so Range/arithmetic below are unambiguous.
        auto empty = rg.make<v0::Constant>(element::i64, Shape{0}, std::vector<int64_t>{});
        return rg.make<v1::Reshape>(x, empty, false);
    };
    auto k_s = to_i64_scalar(k);
    auto s_s = to_i64_scalar(s);

    auto shape = rg.make<v3::ShapeOf>(tensor, element::i64);
    auto L = rg.make<v1::Reshape>(rg.make<v8::Gather>(shape, axis_c, zero),
                                  rg.make<v0::Constant>(element::i64, Shape{0}, std::vector<int64_t>{}),
                                  false);  // scalar extent L
    // out = (L - k)/s + 1
    auto out = rg.make<v1::Add>(rg.make<v1::Divide>(rg.make<v1::Subtract>(L, k_s), s_s), one);
    auto j = rg.make<v4::Range>(zero, out, one, element::i64);  // (out,)
    auto w = rg.make<v4::Range>(zero, k_s, one, element::i64);  // (k,)
    auto one_1d = rg.make<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{1});
    auto zero_1d = rg.make<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{0});
    auto j_col = rg.make<v0::Unsqueeze>(j, one_1d);                         // (out, 1)
    auto w_row = rg.make<v0::Unsqueeze>(w, zero_1d);                        // (1, k)
    auto idx = rg.make<v1::Add>(rg.make<v1::Multiply>(j_col, s_s), w_row);  // (out, k)
    // Gather windows along `axis`: inserts the k axis right after `axis`. `axis` is negative, so the
    // window axis sits at negative index `axis` in the result; reduce it away.
    auto gathered = rg.make<v8::Gather>(tensor, idx, axis_s);
    return rg.make<v1::ReduceMax>(gathered, axis_c, /*keep_dims=*/false);
}

// Build the windowed-gather + ReduceMax decomposition for a runtime kernel. Per spatial axis: max
// over sliding windows of size k with stride s (default stride = kernel). A constant window of 1
// with default stride is an identity and is skipped. The config is validated by the caller before
// this pure builder runs, so it never rejects.
OutputVector build_dynamic_kernel_max_pool(ov::pass::NodeRegistry& rg,
                                           int dims,
                                           const Output<Node>& input,
                                           const std::vector<bool>& elem_is_const,
                                           const std::vector<int64_t>& elem_const_val,
                                           const std::vector<Output<Node>>& elem_runtime_val,
                                           const std::vector<bool>& stride_is_const,
                                           const std::vector<int64_t>& stride_const_val,
                                           const std::vector<Output<Node>>& stride_runtime_val,
                                           bool stride_is_default) {
    Output<Node> res = input;
    for (int i = 0; i < dims; ++i) {
        // Kernel element k for this axis (runtime Output or a constant).
        Output<Node> k = elem_is_const[i]
                             ? Output<Node>(rg.make<v0::Constant>(element::i64, Shape{}, elem_const_val[i]))
                             : elem_runtime_val[i];
        // Stride element s: default (stride == kernel) when stride is absent/None/empty; else the
        // provided (constant or runtime) stride element.
        Output<Node> s;
        if (stride_is_default) {
            s = k;
        } else {
            s = stride_is_const[i] ? Output<Node>(rg.make<v0::Constant>(element::i64, Shape{}, stride_const_val[i]))
                                   : stride_runtime_val[i];
        }
        // Skip an identity axis: constant kernel 1 with default stride leaves the axis unchanged.
        if (elem_is_const[i] && elem_const_val[i] == 1 && stride_is_default) {
            continue;
        }
        res = windowed_max_axis(rg, res, dims, i, k, s);
    }
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
        // Op label used in the annotations below; the unconverted-ops reporter surfaces them.
        const std::string op_label = "aten::max_pool" + std::to_string(dims) + "d";

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

        // We don't throw from transformations: on any config neither lowering can honor, annotate the
        // framework node with the reason and return false, leaving it for the unconverted-ops reporter.
        ov::pass::NodeRegistry rg;
        OutputVector new_outputs;

        if (kernel_is_static) {
            // Kernel now fully constant -> build the static MaxPool.
            Shape kernel;
            for (auto v : elem_const_val)
                kernel.push_back(static_cast<size_t>(v));

            auto stride_vals = fold_optional_list(fw_node, 2);
            if (!stride_vals) {
                add_exception_to_fw_node(fw_node, op_label + " with a non-constant stride is not supported.");
                return false;
            }
            Strides strides;
            if (stride_vals->empty()) {
                strides = kernel;  // default stride is kernel
            } else {
                for (auto v : *stride_vals)
                    strides.push_back(static_cast<size_t>(v));
            }
            auto pad_vals = fold_optional_list(fw_node, 3);
            if (!pad_vals) {
                add_exception_to_fw_node(fw_node, op_label + " with a non-constant padding is not supported.");
                return false;
            }
            Shape pads;
            if (pad_vals->empty()) {
                pads = Shape(kernel.size(), 0);
            } else {
                for (auto v : *pad_vals)
                    pads.push_back(static_cast<size_t>(v));
            }
            auto dil_vals = fold_optional_list(fw_node, 4);
            if (!dil_vals) {
                add_exception_to_fw_node(fw_node, op_label + " with a non-constant dilation is not supported.");
                return false;
            }
            Strides dilations(dims, 1);
            if (!dil_vals->empty()) {
                dilations.clear();
                for (auto v : *dil_vals)
                    dilations.push_back(static_cast<size_t>(v));
            }
            auto ceil_mode = fold_ceil_mode(fw_node, 5);
            if (!ceil_mode) {
                add_exception_to_fw_node(fw_node, op_label + " with a non-constant ceil_mode is not supported.");
                return false;
            }
            RoundingType rounding_type = *ceil_mode ? RoundingType::CEIL_TORCH : RoundingType::FLOOR;
            new_outputs = build_static_max_pool(rg,
                                                fw_node->input_value(0),
                                                dims,
                                                return_indices,
                                                kernel,
                                                strides,
                                                pads,
                                                dilations,
                                                rounding_type);
        } else {
            // Kernel still dynamic -> the windowed-gather + ReduceMax decomposition. It is exact for a
            // sliding window with any kernel/stride, but not for return_indices, padding, dilation, or
            // ceil_mode: reject those, leaving the framework node for the reporter.
            if (return_indices) {
                add_exception_to_fw_node(
                    fw_node,
                    op_label + " with a non-constant kernel_size and return_indices=True is not supported.");
                return false;
            }
            auto pad_vals = fold_optional_list(fw_node, 3);
            if (!pad_vals) {
                add_exception_to_fw_node(fw_node, op_label + " with a non-constant padding is not supported.");
                return false;
            }
            for (auto p : *pad_vals) {
                if (p != 0) {
                    add_exception_to_fw_node(
                        fw_node,
                        op_label + " with a non-constant kernel_size is only supported with zero padding.");
                    return false;
                }
            }
            auto dil_vals = fold_optional_list(fw_node, 4);
            if (!dil_vals) {
                add_exception_to_fw_node(fw_node, op_label + " with a non-constant dilation is not supported.");
                return false;
            }
            for (auto d : *dil_vals) {
                if (d != 1) {
                    add_exception_to_fw_node(
                        fw_node,
                        op_label + " with a non-constant kernel_size is only supported with dilation 1.");
                    return false;
                }
            }
            auto ceil_mode = fold_ceil_mode(fw_node, 5);
            if (!ceil_mode) {
                add_exception_to_fw_node(fw_node, op_label + " with a non-constant ceil_mode is not supported.");
                return false;
            }
            if (*ceil_mode) {
                add_exception_to_fw_node(
                    fw_node,
                    op_label + " with a non-constant kernel_size is only supported with ceil_mode=False.");
                return false;
            }
            if (static_cast<int>(elem_is_const.size()) != dims ||
                static_cast<int>(elem_runtime_val.size()) != dims) {
                add_exception_to_fw_node(fw_node,
                                         op_label + ": could not interpret the non-constant kernel_size (expected " +
                                             std::to_string(dims) + " spatial entries).");
                return false;
            }

            // Probe the stride the same way as the kernel (constant vs runtime, per spatial axis).
            bool stride_is_default = input_is_none_or_missing(fw_node, 2) || is_empty_list(fw_node->input_value(2));
            std::vector<bool> stride_is_const;
            std::vector<int64_t> stride_const_val;
            std::vector<Output<Node>> stride_runtime_val;
            if (!stride_is_default) {
                for (const auto& e : get_list_as_outputs(fw_node->input_value(2))) {
                    if (const auto c = ov::util::get_constant_from_source(e)) {
                        for (auto val : c->cast_vector<int64_t>()) {
                            stride_is_const.push_back(true);
                            stride_const_val.push_back(val);
                            stride_runtime_val.emplace_back();
                        }
                    } else {
                        stride_is_const.push_back(false);
                        stride_const_val.push_back(-1);
                        stride_runtime_val.push_back(e);
                    }
                }
                // A scalar stride applies to every axis (copy element 0 into locals first).
                if (stride_is_const.size() == 1 && dims > 1) {
                    const bool c0 = stride_is_const[0];
                    const int64_t v0 = stride_const_val[0];
                    const Output<Node> r0 = stride_runtime_val[0];
                    stride_is_const.assign(dims, c0);
                    stride_const_val.assign(dims, v0);
                    stride_runtime_val.assign(dims, r0);
                }
                if (static_cast<int>(stride_is_const.size()) != dims) {
                    add_exception_to_fw_node(fw_node,
                                             op_label + ": could not interpret the stride (expected " +
                                                 std::to_string(dims) + " spatial entries).");
                    return false;
                }
            }
            new_outputs = build_dynamic_kernel_max_pool(rg,
                                                        dims,
                                                        fw_node->input_value(0),
                                                        elem_is_const,
                                                        elem_const_val,
                                                        elem_runtime_val,
                                                        stride_is_const,
                                                        stride_const_val,
                                                        stride_runtime_val,
                                                        stride_is_default);
        }

        // copy_runtime_info (not _and_name): don't transplant the FrameworkNode's friendly name onto
        // the decomposed subgraph -- the new nodes keep their own generated names.
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
