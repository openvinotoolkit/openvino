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
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace {
using namespace ov;

// a Constant or a Convert(Constant), e.g. a bias -- not a residual activation
bool is_constant_like(const std::shared_ptr<Node>& node) {
    if (is_type<op::v0::Constant>(node)) {
        return true;
    }
    if (auto convert = as_type_ptr<op::v0::Convert>(node)) {
        return is_type<op::v0::Constant>(convert->get_input_node_shared_ptr(0));
    }
    return false;
}

// if `output` is (optionally, through one Convert) a constant-weight, f16, single-consumer
// MatMul, returns that MatMul; otherwise nullptr. Derived directly from the graph rather than
// via a pattern_value_map lookup: Optional's binding for a nested sub-pattern is not reliable
// when the outer op (Add) is commutative and triggers permutation-based matching.
std::shared_ptr<op::v0::MatMul> get_fc_matmul(const Output<Node>& output) {
    auto node = output.get_node_shared_ptr();
    if (auto convert = as_type_ptr<op::v0::Convert>(node)) {
        node = convert->get_input_node_shared_ptr(0);
    }
    auto matmul = as_type_ptr<op::v0::MatMul>(node);
    if (!matmul || matmul->get_output_element_type(0) != element::f16 ||
        matmul->get_output_target_inputs(0).size() != 1) {
        return nullptr;
    }
    if (!is_type<op::v0::Constant>(matmul->get_input_node_shared_ptr(1)) ||
        is_type<op::v0::Constant>(matmul->get_input_node_shared_ptr(0))) {
        return nullptr;
    }
    return matmul;
}
}  // namespace

namespace ov {
namespace pass {

ClampFP16FCOutput::ClampFP16FCOutput() {
    using namespace ov::op;

    auto add_m = ov::pass::pattern::wrap_type<v1::Add>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto add = ov::as_type_ptr<v1::Add>(m.get_match_root());
        if (!add || transformation_callback(add)) {
            return false;
        }

        for (size_t fc_idx = 0; fc_idx < 2; ++fc_idx) {
            auto fc_output = add->input_value(fc_idx);
            auto residual = add->input_value(1 - fc_idx);
            auto matmul = get_fc_matmul(fc_output);
            if (!matmul || is_constant_like(residual.get_node_shared_ptr())) {
                continue;
            }

            auto min = static_cast<double>(std::numeric_limits<ov::float16>::lowest());
            auto max = static_cast<double>(std::numeric_limits<ov::float16>::max());
            auto clamp = std::make_shared<v0::Clamp>(fc_output, min, max);
            clamp->set_friendly_name(matmul->get_friendly_name() + "/ClampFP16FCOutput");
            ov::copy_runtime_info({matmul, add}, clamp);
            add->input(fc_idx).replace_source_output(clamp);
            return true;
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(add_m, "ClampFP16FCOutput");
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace ov
