// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/qdq_stripping.hpp"

#include <memory>

#include "itt.hpp"
#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/lpt_itt.hpp"
#include "low_precision/network_helper.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/util/log.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace pass {
namespace low_precision {

FQStrippingTransformation::FQStrippingTransformation(const std::set<size_t>& levels_to_strip)
    : levels_to_strip(levels_to_strip) {}

bool FQStrippingTransformation::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(FQStrippingTransformation);

    auto is_scalar_const = [](const std::shared_ptr<Node>& node) -> bool {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
        if (!constant) {
            return false;
        }
        return ov::shape_size(constant->get_shape()) == 1;
    };

    auto constants_are_equal = [](const std::shared_ptr<Node>& lhs, const std::shared_ptr<Node>& rhs) -> bool {
        auto equal =
            ov::as_type_ptr<ov::op::v0::Constant>(ov::op::util::make_try_fold<ov::op::v1::Equal>(lhs, rhs));
        OPENVINO_ASSERT(equal && ov::shape_size(equal->get_shape()) == 1,
                        "constants_are_equal expects scalar constant as a comparison result");
        return equal->get_vector<bool>()[0];
    };

    auto check_fq_constants = [&](const std::shared_ptr<ov::op::v0::FakeQuantize>& fq) -> bool {
        if (!is_scalar_const(fq->get_input_node_shared_ptr(1)) ||
            !is_scalar_const(fq->get_input_node_shared_ptr(2)) ||
            !is_scalar_const(fq->get_input_node_shared_ptr(3)) ||
            !is_scalar_const(fq->get_input_node_shared_ptr(4))) {
            return false;
        }

        if (!constants_are_equal(fq->get_input_node_shared_ptr(1), fq->get_input_node_shared_ptr(3)) ||
            !constants_are_equal(fq->get_input_node_shared_ptr(2), fq->get_input_node_shared_ptr(4))) {
            return false;
        }
        return true;
    };

    bool model_changed = false;
    for (const auto& node : f->get_ordered_ops()) {
        if (transformation_callback(node)) {
            continue;
        }
        auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node);
        if (!fq || !levels_to_strip.count(fq->get_levels()) || !check_fq_constants(fq)) {
            continue;
        }
        OPENVINO_ASSERT(replace_output_update_name(fq->output(0), fq->input_value(0)), "FQ stripping failed");
        model_changed = true;
        std::cout << "[ INFO ] QDQ Stripping: removed FakeQuantize " << fq << std::endl;
    }

    return model_changed;
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ov