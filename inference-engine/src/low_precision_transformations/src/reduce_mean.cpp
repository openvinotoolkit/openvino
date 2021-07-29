// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/reduce_mean.hpp"
#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::ReduceMeanTransformation, "ReduceMeanTransformation", 0);

ReduceMeanTransformation::ReduceMeanTransformation(const Params& params) : ReduceBaseTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::ReduceMean>({ pattern::wrap_type<opset1::Multiply>(), pattern::wrap_type<opset1::Constant>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "ReduceMeanTransformation");
    this->register_matcher(m, callback);
}

bool ReduceMeanTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> reduce) const {
    return is_type<opset1::ReduceMean>(reduce) ? ReduceBaseTransformation::canBeTransformed(context, reduce) : false;
}

bool ReduceMeanTransformation::isPrecisionPreserved(std::shared_ptr<Node> reduce) const noexcept {
    return false;
}

bool ReduceMeanTransformation::getUpdatePrecision(const std::shared_ptr<Node>& reduce) const {
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
