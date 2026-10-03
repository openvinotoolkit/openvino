// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/fp16_compression/mark_sin_cos_angles_to_keep_in_mixed_precision.hpp"

#include <unordered_set>
#include <vector>

#include "itt.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cos.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/util/convolution_base.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/utils/utils.hpp"

namespace v0 = ov::op::v0;
namespace op_util = ov::op::util;

ov::pass::MarkSinCosAnglesToKeepInMixedPrecision::MarkSinCosAnglesToKeepInMixedPrecision() {
    MATCHER_SCOPE(MarkSinCosAnglesToKeepInMixedPrecision);
    using namespace ov::pass::pattern;

    auto sin_or_cos = wrap_type<v0::Sin, v0::Cos>({any_input()});

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto root = m.get_match_root();
        if (transformation_callback(root)) {
            return false;
        }

        // Whether the argument depends on activations is a property of the whole input path,
        // so it cannot be expressed in the pattern.
        bool depends_on_activations = false;
        std::vector<ov::Node*> path;
        std::unordered_set<ov::Node*> visited;
        op_util::visit_path(
            root.get(),
            visited,
            [&path](ov::Node* node) {
                path.push_back(node);
            },
            [&depends_on_activations](ov::Node* node) {
                if (ov::is_type_any_of<v0::MatMul, op_util::ConvolutionBase>(node)) {
                    depends_on_activations = true;
                    return true;
                }
                return ov::is_type_any_of<v0::Constant, v0::Parameter, op_util::ShapeOfBase>(node);
            });
        if (depends_on_activations) {
            return false;
        }

        for (const auto& node : path) {
            ov::disable_conversion(node->shared_from_this(), element::f16);
        }
        // markup only: the graph itself is not changed
        return false;
    };

    auto m = std::make_shared<Matcher>(sin_or_cos, matcher_name);
    register_matcher(m, callback);
}
