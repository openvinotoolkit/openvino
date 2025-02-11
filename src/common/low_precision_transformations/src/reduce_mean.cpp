// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/reduce_mean.hpp"
#include <memory>

#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ov {
namespace pass {
namespace low_precision {

ReduceMeanTransformation::ReduceMeanTransformation(const Params& params) : ReduceBaseTransformation(params) {
    MATCHER_SCOPE(ReduceMeanTransformation);
    auto matcher = pattern::wrap_type<ov::opset1::ReduceMean>({ pattern::wrap_type<ov::opset1::Multiply>(), pattern::wrap_type<ov::opset1::Constant>() });

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool ReduceMeanTransformation::canBeTransformed(const std::shared_ptr<Node>& reduce) const {
    return ov::is_type<ov::opset1::ReduceMean>(reduce) ? ReduceBaseTransformation::canBeTransformed(reduce) : false;
}

bool ReduceMeanTransformation::isPrecisionPreserved(std::shared_ptr<Node> reduce) const noexcept {
    return false;
}

bool ReduceMeanTransformation::getUpdatePrecision(const std::shared_ptr<Node>& reduce) const {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
