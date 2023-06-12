// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "irfftn_complex_replacer.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/irdft.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils.hpp"

using namespace ov::pass::pattern;

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::pass;
using namespace ov::op;

IRFFTNComplexReplacer::IRFFTNComplexReplacer() {
    auto fft_op = pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback irfftn_callback = [](pattern::Matcher& m) {
        // Pattern: aten::complex -> aten::fft_irfftn
        // "aten::fft_irfftn(Tensor self, int[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor"
        auto irfftn_op = cast_fw_node(m.get_match_root(), "aten::fft_irfftn");
        if (!irfftn_op) {
            return false;
        }
        auto const_neg_1 = v0::Constant::create(element::i32, Shape{1}, {-1});
        auto const_0 = v0::Constant::create(element::i32, Shape{1}, {0});
        auto const_scalar_0 = v0::Constant::create(element::i32, Shape{}, {0});
        auto const_1 = v0::Constant::create(element::i32, Shape{1}, {1});
        auto const_scalar_1 = v0::Constant::create(element::i32, Shape{}, {1});
        auto const_2 = v0::Constant::create(element::i32, Shape{1}, {2});
        auto fw_node_complex_input = cast_fw_node(irfftn_op->input_value(0).get_node_shared_ptr(), "aten::complex");
        if (!fw_node_complex_input) {
            return false;
        }
        auto real = std::make_shared<v0::Unsqueeze>(fw_node_complex_input->input(0).get_source_output(), const_neg_1);
        auto imag = std::make_shared<v0::Unsqueeze>(fw_node_complex_input->input(1).get_source_output(), const_neg_1);
        NodeVector complex = {real, imag};
        auto input = std::make_shared<v0::Concat>(complex, -1);

        auto complex_input_shape =
            std::make_shared<v3::ShapeOf>(fw_node_complex_input->input(0).get_source_output(), element::i32);
        auto input_rank = std::make_shared<v3::ShapeOf>(complex_input_shape, element::i32);
        auto input_rank_scalar = std::make_shared<v0::Squeeze>(input_rank);

        bool dim_use_default = is_none_node(irfftn_op->input_value(2).get_node_shared_ptr());
        bool s_use_default = is_none_node(irfftn_op->input_value(1).get_node_shared_ptr());
        // Can be None constant, when used check s_use_default.
        auto raw_s_input_maybe = concat_list_construct(irfftn_op->input_value(1)).get_node_shared_ptr();

        std::shared_ptr<ov::Node> dim;
        if (!dim_use_default) {
            auto dim_range = std::make_shared<v0::Range>(const_scalar_0, input_rank_scalar, const_scalar_1);
            dim = std::make_shared<v0::Convert>(concat_list_construct(irfftn_op->input_value(2)), element::i32);
            dim = std::make_shared<v8::Gather>(dim_range, dim, const_0);
        } else if (!s_use_default) {
            auto s_len = std::make_shared<v3::ShapeOf>(raw_s_input_maybe, element::i32);
            auto range_start = std::make_shared<v1::Subtract>(input_rank, s_len);
            auto range_start_scalar = std::make_shared<v0::Squeeze>(range_start);
            dim = std::make_shared<v0::Range>(range_start_scalar, input_rank_scalar, const_scalar_1);
        } else {
            dim = std::make_shared<v0::Range>(const_scalar_0, input_rank_scalar, const_scalar_1);
        }

        auto default_s_raw = std::make_shared<v8::Gather>(complex_input_shape, dim, const_0);
        auto last_s = std::make_shared<v8::Gather>(default_s_raw, const_neg_1, const_0);
        auto last_s_m_1 = std::make_shared<v1::Subtract>(last_s, const_1);
        auto s_upd = std::make_shared<v1::Multiply>(last_s_m_1, const_2);
        auto s_shape = std::make_shared<v3::ShapeOf>(default_s_raw, element::i32);
        auto last_s_idx = std::make_shared<v1::Subtract>(s_shape, const_1);
        auto default_s = std::make_shared<v3::ScatterUpdate>(default_s_raw, last_s_idx, s_upd, const_0);

        std::shared_ptr<ov::Node> s;
        if (!s_use_default) {
            auto full_s_cond = std::make_shared<v1::Equal>(raw_s_input_maybe, const_neg_1);
            s = std::make_shared<v1::Select>(full_s_cond, default_s, raw_s_input_maybe);
        } else {
            s = default_s;
        }

        std::string norm;
        if (const auto& fw_node_mode = std::dynamic_pointer_cast<ov::op::util::FrameworkNode>(
                irfftn_op->input_value(3).get_node_shared_ptr())) {
            const auto& attrs = fw_node_mode->get_attrs();
            if (attrs.find("string_value") != attrs.end()) {
                norm = attrs.at("string_value");
            } else {
                norm = "backward";
            }
        } else {
            return false;
        }

        auto irdft = std::make_shared<v9::IRDFT>(input, dim, s);

        auto n_int = std::make_shared<v1::ReduceProd>(s, const_0);
        auto n = std::make_shared<v1::ConvertLike>(n_int, irdft);
        std::shared_ptr<ov::Node> normalized_irfftn;
        if (norm == "forward") {
            normalized_irfftn = std::make_shared<v1::Multiply>(irdft, n);
        } else if (norm == "backward") {
            normalized_irfftn = irdft;
        } else if (norm == "ortho") {
            auto sqrt_n = std::make_shared<v0::Sqrt>(n);
            normalized_irfftn = std::make_shared<v1::Multiply>(irdft, sqrt_n);
        } else {
            return false;
        }

        copy_runtime_info({irfftn_op, fw_node_complex_input}, normalized_irfftn);
        replace_node(irfftn_op, normalized_irfftn);
        return true;
    };
    auto m = std::make_shared<pattern::Matcher>(fft_op, "ov::frontend::pytorch::pass::IRFFTNComplexReplacer");
    this->register_matcher(m, irfftn_callback);
};

}  // namespace pass
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
