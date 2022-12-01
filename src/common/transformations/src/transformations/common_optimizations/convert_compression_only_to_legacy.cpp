// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_compression_only_to_legacy.hpp"

#include "itt.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/common_optimizations/mark_precision_sensitive_subgraphs.hpp"
#include "transformations/convert_precision.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
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
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(convert, matcher_name);
    this->register_matcher(m, callback);
}

bool ov::pass::ConvertCompressedOnlyToLegacy::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_MODEL_SCOPE(ConvertCompressedOnlyToLegacy);
    if (ngraph::op::util::has_decompression_converts(f)) {
        Manager manager(get_pass_config());
        // Skip precision sensitive nodes with marking and pass_callback:
        // callback skips (returns true) for nodes marked as precision sensitive/disabled_f16_compression.
        // Skipping was done by callback in order to impact behavior of ConvertPrecision as little as possible
        manager.register_pass<ov::pass::MarkPrecisionSensitiveSubgraphs>();
        get_pass_config()->set_callback<ngraph::pass::ConvertPrecision>(
            [](const std::shared_ptr<const Node>& node) -> bool {
                auto const const_node = std::dynamic_pointer_cast<const ov::opset8::Constant>(node);
                if (!const_node)
                    return false;
                return ov::fp16_compression_is_disabled(node) && const_node->get_output_element_type(0) == element::f32;
            });

        const precisions_array convert_precision_list{{ov::element::f32, ov::element::f16}};
        manager.register_pass<ngraph::pass::ConvertPrecision>(convert_precision_list);
        using namespace ov::pass;
        REGISTER_PASS(manager, EnableDecompressionConvertConstantFolding)
        REGISTER_PASS(manager, ConstantFolding)

        manager.run_passes(f);
    }
    return false;
}
