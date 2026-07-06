// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "max_pool_dynamic_kernel_resolver.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"
// Included after utils.hpp: op/max_poolnd.hpp opens ns ov::frontend::pytorch::op, and utils.hpp uses
// unqualified op:: expecting ov::op, so utils.hpp must be parsed before that namespace is visible.
#include "op/max_poolnd.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::op;

namespace {

// The deferred max_pool op-type strings emitted as PtFrameworkNode by translate_max_pool_base
// (mirrors the op_table entries for TorchScript and FX).
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

// True when input `index` is absent (the node has fewer than index+1 inputs) or is an explicit
// `none` node (an omitted optional argument).
bool input_is_none_or_missing(const std::shared_ptr<ov::op::util::FrameworkNode>& fw, size_t index) {
    if (index >= fw->get_input_size())
        return true;
    return is_none_node(fw->input_value(index));
}

// Fold an optional list-valued attribute input (stride/padding/dilation) to i64 values. Returns an
// empty vector when the input is absent/None/empty (meaning "use the default"). Raises
// OpConversionFailure when the input is present but not constant-foldable: a runtime-computed
// stride/padding/dilation cannot be honored by the static MaxPool attributes or the ReduceMax
// decomposition, and silently treating it as the default would miscompute. `attr_name` names the
// attribute in the message.
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

// Fold the optional ceil_mode flag. Returns false (the default) when the input is absent/None;
// raises OpConversionFailure when it is present but not constant-foldable (a runtime ceil_mode has
// no safe conversion-time interpretation for either lowering).
bool fold_ceil_mode(const std::shared_ptr<ov::op::util::FrameworkNode>& fw, size_t index, int dims) {
    if (input_is_none_or_missing(fw, index))
        return false;
    auto ceil_c = ov::util::get_constant_from_source(fw->input_value(index));
    PYTORCH_OP_CONVERSION_CHECK(ceil_c, "aten::max_pool", dims, "d with a non-constant ceil_mode is not supported.");
    return ceil_c->cast_vector<bool>()[0];
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

        // Re-probe the kernel_size list now that shapes have propagated. get_list_as_outputs handles
        // both the SequenceMark form and a list already lowered to elementwise outputs.
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
        // A scalar kernel_size applies to every spatial axis in PyTorch. Copy element 0 into locals
        // first: assign(dims, elem_runtime_val[0]) would read a reference into the same vector while
        // it reallocates (grows 1 -> dims), which is UB for a self-aliased value.
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

        // Folding the optional stride/padding/dilation/ceil_mode inputs, and the ReduceMax guards,
        // raise OpConversionFailure for a configuration neither lowering can honor (a non-constant
        // attribute, a strided/padded/return_indices dynamic pool, ...). Route that to the standard
        // unconverted-ops reporter instead of letting it escape run_passes.
        try {
            if (kernel_is_static) {
                // The kernel is now fully constant -> build the ordinary static MaxPool.
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
                // The kernel is still dynamic -> the ReduceMax full-extent decomposition (with guards).
                bool stride_is_default =
                    input_is_none_or_missing(fw_node, 2) || is_empty_list(fw_node->input_value(2));
                auto pads = fold_optional_list(fw_node, 3, dims, "padding");
                auto dilations = fold_optional_list(fw_node, 4, dims, "dilation");
                bool ceil_mode = fold_ceil_mode(fw_node, 5, dims);
                new_outputs = op::build_dynamic_kernel_max_pool(rg,
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
            // Unsupported configuration (non-constant/strided/padded/return_indices/...). Annotate the
            // placeholder so the standard unconverted-ops reporter surfaces a clear message.
            add_exception_to_fw_node(fw_node, e.what());
            return false;
        }

        // copy_runtime_info (not _and_name) so the guard nodes keep their op-labeled friendly names
        // (e.g. aten::max_poolNd/require_full_extent, asserted by the runtime-guard test).
        ov::copy_runtime_info(fw_node, rg.get());
        replace_node(fw_node, new_outputs);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(
        fw_pattern,
        "ov::frontend::pytorch::pass::MaxPoolDynamicKernelResolver");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
