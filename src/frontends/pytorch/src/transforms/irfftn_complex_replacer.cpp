// Copyright (C) 2018-2024 Intel Corporation
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

namespace ov {
namespace frontend {
namespace pytorch {
namespace pass {

using namespace ov::pass;
using namespace ov::op;

IRFFTNComplexReplacer::IRFFTNComplexReplacer() {
    // Transformation used to replace combination of aten::complex -> aten::fft_irfftn torch operators.
    // Pattern: aten::complex -> aten::fft_irfftn
    auto fft_op = pattern::wrap_type<ov::op::util::FrameworkNode>();

    ov::matcher_pass_callback irfftn_callback = [](pattern::Matcher& m) {
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

        // Check whether input node being aten::complex.
        auto fw_node_complex_input = cast_fw_node(irfftn_op->input_value(0).get_node_shared_ptr(), "aten::complex");
        if (!fw_node_complex_input) {
            return false;
        }

        // Concatenate real and imag parts over additional, last dimension.
        auto real = std::make_shared<v0::Unsqueeze>(fw_node_complex_input->input_value(0), const_neg_1);
        auto imag = std::make_shared<v0::Unsqueeze>(fw_node_complex_input->input_value(1), const_neg_1);
        NodeVector complex = {real, imag};
        auto input = std::make_shared<v0::Concat>(complex, -1);

        // Input shape of complex number (excluding dimension created by concatenation of real and imag)
        auto complex_input_shape = std::make_shared<v3::ShapeOf>(fw_node_complex_input->input_value(0), element::i32);
        auto input_rank = std::make_shared<v3::ShapeOf>(complex_input_shape, element::i32);
        auto input_rank_scalar = std::make_shared<v0::Squeeze>(input_rank);

        // Inputs can be either none or ListConstruct. Check whether input values should be used or should be set to
        // default values.
        bool dim_use_default = is_none_node(irfftn_op->input_value(2));
        bool s_use_default = is_none_node(irfftn_op->input_value(1));
        // Can be None constant, when used check s_use_default.
        auto raw_s_input_maybe = concat_list_construct(irfftn_op->input_value(1));
        raw_s_input_maybe = std::make_shared<v0::Convert>(raw_s_input_maybe, element::i32);

        // Handle dim parameter containing vector of integers indicating dimensions to be transformed.
        std::shared_ptr<ov::Node> dim;
        if (!dim_use_default) {
            // Dim values is provided, load from input.
            dim = std::make_shared<v0::Convert>(concat_list_construct(irfftn_op->input_value(2)), element::i32);
        } else if (!s_use_default) {
            // If dim is default and s is provided, use last s_len dimensions where s_len is length of s.
            auto s_len = std::make_shared<v3::ShapeOf>(raw_s_input_maybe, element::i32);
            auto range_start = std::make_shared<v1::Subtract>(input_rank, s_len);
            auto range_start_scalar = std::make_shared<v0::Squeeze>(range_start);
            dim = std::make_shared<v4::Range>(range_start_scalar, input_rank_scalar, const_scalar_1, element::i32);
        } else {
            // Dim and s are set to default, use all of dimensions.
            dim = std::make_shared<v4::Range>(const_scalar_0, input_rank_scalar, const_scalar_1, element::i32);
        }

        // Calculate default s values. Use full available size except last element, which is set to even value in last
        // dimension: s[-1] = 2 * (complex_input_shape[dim[-1]])
        auto default_s_raw = std::make_shared<v8::Gather>(complex_input_shape, dim, const_0);
        auto last_s = std::make_shared<v8::Gather>(default_s_raw, const_neg_1, const_0);
        auto last_s_m_1 = std::make_shared<v1::Subtract>(last_s, const_1);
        auto s_upd = std::make_shared<v1::Multiply>(last_s_m_1, const_2);
        auto s_shape = std::make_shared<v3::ShapeOf>(default_s_raw, element::i32);
        auto last_s_idx = std::make_shared<v1::Subtract>(s_shape, const_1);
        auto default_s = std::make_shared<v3::ScatterUpdate>(default_s_raw, last_s_idx, s_upd, const_0);

        // Handle s parameter containing vector of intigers indicating signal sizes for dimensions.
        std::shared_ptr<ov::Node> s;
        if (!s_use_default) {
            // Values for s were provided. Replace -1 values with default full size in given dimension.
            auto full_s_cond = std::make_shared<v1::Equal>(raw_s_input_maybe, const_neg_1);
            s = std::make_shared<v1::Select>(full_s_cond, default_s, raw_s_input_maybe);
        } else {
            // Value for s was set to default.
            s = default_s;
        }

        // Handle norm parameter indicating normalization mode to use. Defaults to "backward".
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
            add_exception_to_fw_node(irfftn_op, "aten::fft_irfftn: could not retrive value for norm attribute.");
            return false;
        }

        auto irdft = std::make_shared<v9::IRDFT>(input, dim, s);

        // Apply normalizations.
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
            add_exception_to_fw_node(
                irfftn_op,
                "aten::fft_irfftn: unrecognized normalization mode. Only forward, backward and ortho are supported.");
            return false;
        }

        copy_runtime_info({irfftn_op, fw_node_complex_input}, normalized_irfftn);
        normalized_irfftn->set_friendly_name(irfftn_op->get_friendly_name());
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
