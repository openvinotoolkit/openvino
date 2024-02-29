// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fftn_complex_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/dft.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::pass;
using namespace ov::op;

FFTNComplexReplacer::FFTNComplexReplacer() {
    // Transformation used to replace combination of aten::fft_fftn -> aten::real torch operator.
    // Pattern: aten::fft_fftn -> aten::real
    auto fft_op = pattern::wrap_type<ov::op::util::FrameworkNode>();
    ov::matcher_pass_callback fftn_callback = [](pattern::Matcher& m) {
        // Schema: "aten::fft_fftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"
        auto fftn_op = cast_fw_node(m.get_match_root(), "aten::fft_fftn");
        if (!fftn_op) {
            return false;
        }
        auto const_neg_1 = v0::Constant::create(element::i32, Shape{}, {-1});
        auto const_0 = v0::Constant::create(element::i32, Shape{}, {0});
        auto const_1 = v0::Constant::create(element::i32, Shape{}, {1});

        auto input = fftn_op->input_value(0);
        auto input_shape = std::make_shared<v3::ShapeOf>(input, element::i32);
        auto input_rank = std::make_shared<v3::ShapeOf>(input_shape, element::i32);
        auto input_rank_scalar = std::make_shared<v0::Squeeze>(input_rank);

        // Inputs can be either none or ListConstruct. Check whether input values should be used or should be set to
        // default values.
        bool dim_use_default = is_none_node(fftn_op->input_value(2));
        bool s_use_default = is_none_node(fftn_op->input_value(1));
        // Can be None constant, when used check s_use_default.
        auto raw_s_input_maybe = concat_list_construct(fftn_op->input_value(1));

        // Handle dim parameter containing vector of intigers indicating dimensions to be transformed.
        std::shared_ptr<ov::Node> dim;
        if (!dim_use_default) {
            // Dim values is provided, load from input.
            dim = std::make_shared<v0::Convert>(concat_list_construct(fftn_op->input_value(2)), element::i32);
        } else if (!s_use_default) {
            // If dim is default and s is provided, use last s_len dimensions where s_len is length of s.
            auto s_len = std::make_shared<v3::ShapeOf>(raw_s_input_maybe, element::i32);
            auto slice_start = std::make_shared<v1::Subtract>(input_rank, s_len);
            auto slice_start_scalar = std::make_shared<v0::Squeeze>(slice_start);
            dim = std::make_shared<v4::Range>(slice_start_scalar, input_rank_scalar, const_1, element::i32);
        } else {
            // Dim and s are set to default, use all of dimensions.
            dim = std::make_shared<v4::Range>(const_0, input_rank_scalar, const_1, element::i32);
        }

        // Handle s parameter containing vector of integers indicating signal sizes for dimensions.
        std::shared_ptr<ov::Node> s;
        if (!s_use_default) {
            // Values for s were provided. Replace -1 values with default full size in given dimension.
            auto full_s_cond = std::make_shared<v1::Equal>(raw_s_input_maybe, const_neg_1);
            auto full_s_values = std::make_shared<v8::Gather>(input_shape, dim, const_0);
            s = std::make_shared<v1::Select>(full_s_cond, full_s_values, raw_s_input_maybe);
        } else {
            // Value for s was set to default, use full size for all dimensions.
            s = std::make_shared<v8::Gather>(input_shape, dim, const_0);
        }

        // Handle norm parameter indicating normalization mode to use. Defaults to "backward".
        std::string norm;
        if (const auto& fw_node_mode =
                std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(fftn_op->input_value(3).get_node_shared_ptr())) {
            const auto& attrs = fw_node_mode->get_attrs();
            if (attrs.find("string_value") != attrs.end()) {
                norm = attrs.at("string_value");
            } else {
                norm = "backward";
            }
        } else {
            add_exception_to_fw_node(fftn_op, "aten::fft_fftn: could not retrive value for norm attribute.");
            return false;
        }

        auto dft = std::make_shared<v7::DFT>(input, dim, s);

        // Apply normalizations
        auto n_int = std::make_shared<v1::ReduceProd>(s, const_0);
        auto n = std::make_shared<v1::ConvertLike>(n_int, dft);
        std::shared_ptr<ov::Node> normalized_fftn;
        if (norm == "forward") {
            // Normalize by 1/n
            normalized_fftn = std::make_shared<v1::Divide>(dft, n);
        } else if (norm == "backward") {
            // No normalization
            normalized_fftn = dft;
        } else if (norm == "ortho") {
            // Normalize by 1/sqrt(n)
            auto sqrt_n = std::make_shared<v0::Sqrt>(n);
            normalized_fftn = std::make_shared<v1::Divide>(dft, sqrt_n);
        } else {
            add_exception_to_fw_node(
                fftn_op,
                "aten::fft_fftn: unrecognized normalization mode. Only forward, backward and ortho are supported.");
            return false;
        }

        // Replace outputs that are either torch operators aten::real or aten::imag. Apply squeeze to remove last
        // dimension used to concatenate.
        auto normalized_rfftn_splitted = std::make_shared<v1::Split>(normalized_fftn, const_neg_1, 2);
        auto fftn_outs = fftn_op->get_users();
        bool rval = false;
        for (auto& out : fftn_outs) {
            if (auto real_op = cast_fw_node(out, "aten::real")) {
                auto squeezed = std::make_shared<v0::Squeeze>(normalized_rfftn_splitted->output(0), const_neg_1);
                copy_runtime_info({fftn_op, real_op}, squeezed);
                squeezed->set_friendly_name(real_op->get_friendly_name());
                replace_node(real_op, squeezed);
                rval = true;
            }
        }
        add_exception_to_fw_node(fftn_op, "aten::fft_fftn: Unsupported output node. Only aten::real is supported.");
        return rval;
    };

    auto m = std::make_shared<pattern::Matcher>(fft_op, "ov::frontend::pytorch::pass::FFTNComplexReplacer");
    this->register_matcher(m, fftn_callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
