// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rfftn_complex_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/rdft.hpp"
#include "openvino/op/reduce_prod.hpp"
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

RFFTNComplexReplacer::RFFTNComplexReplacer() {
    auto fft_op = pattern::wrap_type<ov::op::util::FrameworkNode>();
    ov::matcher_pass_callback rfftn_callback = [](pattern::Matcher& m) {
        // Pattern: aten::fft_rfftn -> {aten::real, aten::imag}
        // "aten::fft_rfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"
        auto rfftn_op = cast_fw_node(m.get_match_root(), "aten::fft_rfftn");
        if (!rfftn_op) {
            return false;
        }
        auto const_neg_1 = v0::Constant::create(element::i32, Shape{}, {-1});
        auto const_0 = v0::Constant::create(element::i32, Shape{}, {0});
        auto const_1 = v0::Constant::create(element::i32, Shape{}, {1});

        auto input = rfftn_op->input_value(0);
        auto input_shape = std::make_shared<v3::ShapeOf>(input, element::i32);
        auto input_rank = std::make_shared<v3::ShapeOf>(input_shape, element::i32);
        auto input_rank_scalar = std::make_shared<v0::Squeeze>(input_rank);

        auto node_s_input = concat_list_construct(rfftn_op->input_value(1)).get_node_shared_ptr();

        bool dim_use_default = is_none_node(rfftn_op->input_value(2).get_node_shared_ptr());
        bool s_use_default = is_none_node(rfftn_op->input_value(1).get_node_shared_ptr());

        std::shared_ptr<ov::Node> dim;
        if (!dim_use_default) {
            dim = std::make_shared<v0::Convert>(concat_list_construct(rfftn_op->input_value(2)), element::i32);
        } else if (!s_use_default) {
            auto s_len = std::make_shared<v3::ShapeOf>(node_s_input, element::i32);
            auto slice_start = std::make_shared<v1::Subtract>(input_rank, s_len);
            auto slice_start_scalar = std::make_shared<v0::Squeeze>(slice_start);
            dim = std::make_shared<v0::Range>(slice_start_scalar, input_rank_scalar, const_1);
        } else {
            dim = std::make_shared<v0::Range>(const_0, input_rank_scalar, const_1);
        }

        std::shared_ptr<ov::Node> s;
        if (!s_use_default) {
            auto full_s_cond = std::make_shared<v1::Equal>(node_s_input, const_neg_1);
            auto full_s_values = std::make_shared<v8::Gather>(input_shape, dim, const_0);
            s = std::make_shared<v1::Select>(full_s_cond, full_s_values, node_s_input);
        } else {
            s = std::make_shared<v8::Gather>(input_shape, dim, const_0);
        }

        std::string norm;
        if (const auto& fw_node_mode = std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(
                rfftn_op->input_value(3).get_node_shared_ptr())) {
            const auto& attrs = fw_node_mode->get_attrs();
            if (attrs.find("string_value") != attrs.end()) {
                norm = attrs.at("string_value");
            } else {
                norm = "backward";
            }
        } else {
            return false;
        }

        auto rdft = std::make_shared<v9::RDFT>(input, dim, s);

        auto n_int = std::make_shared<v1::ReduceProd>(s, const_0);
        auto n = std::make_shared<v1::ConvertLike>(n_int, rdft);
        std::shared_ptr<ov::Node> normalized_rfftn;
        if (norm == "forward") {
            normalized_rfftn = std::make_shared<v1::Divide>(rdft, n);
        } else if (norm == "backward") {
            normalized_rfftn = rdft;
        } else if (norm == "ortho") {
            auto sqrt_n = std::make_shared<v0::Sqrt>(n);
            normalized_rfftn = std::make_shared<v1::Divide>(rdft, sqrt_n);
        } else {
            return false;
        }
        auto normalized_rfftn_splitted = std::make_shared<v1::Split>(normalized_rfftn, const_neg_1, 2);
        auto rfftn_outs = rfftn_op->get_users();
        bool rval = false;
        for (auto out : rfftn_outs) {
            if (auto real_op = cast_fw_node(out, "aten::real")) {
                auto squeezed = std::make_shared<v0::Squeeze>(normalized_rfftn_splitted->output(0), const_neg_1);
                copy_runtime_info({rfftn_op, real_op}, squeezed);
                replace_node(real_op, squeezed);
                rval = true;
            }
            if (auto imag_op = cast_fw_node(out, "aten::imag")) {
                auto squeezed = std::make_shared<v0::Squeeze>(normalized_rfftn_splitted->output(1), const_neg_1);
                copy_runtime_info({rfftn_op, imag_op}, squeezed);
                replace_node(imag_op, squeezed);
                rval = true;
            }
        }
        return rval;
    };

    auto m = std::make_shared<pattern::Matcher>(fft_op, "ov::frontend::pytorch::pass::RFFTNComplexReplacer");
    this->register_matcher(m, rfftn_callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
