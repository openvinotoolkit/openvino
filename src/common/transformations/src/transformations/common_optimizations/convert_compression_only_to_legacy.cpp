// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"

#include "itt.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

ov::pass::EnableDecompressionConvertConstantFolding::EnableDecompressionConvertConstantFolding() {
    MATCHER_SCOPE(EnableDecompressionConvertConstantFolding);
    auto convert = pattern::wrap_type<opset8::Convert>();

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& node = m.get_match_root();
        if (!ov::is_decompression(node))
            return false;
        enable_constant_folding(node);
        MATCHER_SCOPE_ENABLE(EnableDecompressionConvertConstantFolding);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(convert, matcher_name);
    this->register_matcher(m, callback);
}

bool ov::pass::ConvertCompressedOnlyToLegacy::run_on_model(const std::shared_ptr<ov::Model>& f) {
    if (ngraph::op::util::has_decompression_converts(f)) {
        Manager manager(get_pass_config());

        const precisions_array convert_precision_list{{ov::element::f32, ov::element::f16}};
        manager.register_pass<ngraph::pass::ConvertPrecision>(convert_precision_list);
        manager.register_pass<ov::pass::EnableDecompressionConvertConstantFolding>();
        manager.register_pass<ov::pass::ConstantFolding>();

        manager.run_passes(f);
    }
    return false;
}
