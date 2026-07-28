// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/clamp_fp16_fc_output.hpp"

#include <limits>
#include <memory>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace {
// excludes a Constant and a Convert(Constant), e.g. a bias -- not a residual activation
bool is_not_constant_like(const std::shared_ptr<ov::Node>& node) {
    if (ov::is_type<ov::op::v0::Constant>(node)) {
        return false;
    }
    if (auto convert = ov::as_type_ptr<ov::op::v0::Convert>(node)) {
        return !ov::is_type<ov::op::v0::Constant>(convert->get_input_node_shared_ptr(0));
    }
    return true;
}
}  // namespace

namespace ov {
namespace pass {

ClampFP16FCOutput::ClampFP16FCOutput() {
    using namespace ov::op;
    using namespace ov::pass::pattern;
    using namespace ov::pass::pattern::op;

    auto activation_in = any_input(class_other_than<v0::Constant>());
    auto weight_in = wrap_type<v0::Constant>();
    auto matmul_m =
        wrap_type<v0::MatMul>({activation_in, weight_in}, type_matches(ov::element::f16) && consumers_count(1));
    auto fc_output_m = optional<v0::Convert>(matmul_m);
    auto residual_in = any_input(is_not_constant_like);
    auto residual_add_m = wrap_type<v1::Add>({fc_output_m, residual_in});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto add = ov::as_type_ptr<v1::Add>(pattern_map.at(residual_add_m).get_node_shared_ptr());
        if (!add || transformation_callback(add)) {
            return false;
        }
        auto matmul = pattern_map.at(matmul_m).get_node_shared_ptr();
        auto fc_output = pattern_map.at(fc_output_m);

        int fc_input_index = -1;
        for (size_t i = 0; i < add->get_input_size(); ++i) {
            if (add->input_value(i) == fc_output) {
                fc_input_index = static_cast<int>(i);
                break;
            }
        }
        if (fc_input_index < 0) {
            return false;
        }

        auto min = static_cast<double>(std::numeric_limits<ov::float16>::lowest());
        auto max = static_cast<double>(std::numeric_limits<ov::float16>::max());
        auto clamp = std::make_shared<v0::Clamp>(fc_output, min, max);
        clamp->set_friendly_name(matmul->get_friendly_name() + "/ClampFP16FCOutput");
        ov::copy_runtime_info({matmul, add}, clamp);
        add->input(fc_input_index).replace_source_output(clamp);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(residual_add_m, "ClampFP16FCOutput");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace ov
