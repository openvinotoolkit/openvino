// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/filter_precision_attribute.hpp"

#include "low_precision/fake_quantize_decomposition.hpp"
#include "low_precision/network_helper.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov {
namespace pass {
namespace low_precision {

FilterPrecisionAttribute::FilterPrecisionAttribute() {
    auto matcher = pattern::wrap_type<opset1::FakeQuantize>();

    ov::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto fq = ov::as_type_ptr<opset1::FakeQuantize>(m.get_match_root());
        if (!fq) {
            return false;
        }

        if (NetworkHelper::isConstantPath(fq)) {
            return false;
        }

        auto attribute = getAttributeFromOutput<PrecisionsAttribute>(fq->output(0));
        if (attribute.empty()) {
            return false;
        }

        const auto& precisions = attribute.as<PrecisionsAttribute>().value();
        if (precisions.size() <= 1ul) {
            return false;
        }

        // Resolve multi-precision attribute to a single precision based on FQ output ranges
        fq_decomposition::getDataPrecisionByOutputPort(fq);
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, "FilterPrecisionAttribute");
    this->register_matcher(m, callback);
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ov
