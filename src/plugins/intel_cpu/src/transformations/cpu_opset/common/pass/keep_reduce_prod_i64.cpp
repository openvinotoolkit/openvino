// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "keep_reduce_prod_i64.hpp"

#include <memory>

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

namespace ov::intel_cpu {

KeepReduceProdI64::KeepReduceProdI64() {
    MATCHER_SCOPE(KeepReduceProdI64);
    using namespace ov::pass::pattern;

    auto reduce_m = wrap_type<ov::op::v1::ReduceProd>({any_input(), any_input()}, type_matches(ov::element::i64));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto reduce = m.get_match_root();
        if (transformation_callback(reduce) || ov::is_conversion_disabled(reduce, ov::element::i64, ov::element::i32)) {
            return false;
        }

        // The data input must be produced by an i64 Convert that is kept in i64: otherwise ConvertPrecision
        // would lower the producer to i32 and ReduceProd would be re-inferred as i32.
        auto data = reduce->input_value(0);
        auto in_convert = ov::as_type_ptr<ov::op::v0::Convert>(data.get_node_shared_ptr());
        if (!in_convert || in_convert->get_output_target_inputs(0).size() != 1) {
            in_convert = std::make_shared<ov::op::v0::Convert>(data, ov::element::i64);
            in_convert->set_friendly_name(reduce->get_friendly_name() + "/convert_to_i64");
            ov::copy_runtime_info(reduce, in_convert);
            reduce->input(0).replace_source_output(in_convert);
        }
        ov::disable_conversion(in_convert, ov::element::i64, ov::element::i32);
        ov::disable_conversion(reduce, ov::element::i64, ov::element::i32);

        // Results keep the exact i64 value; every other consumer reads it through a Convert which
        // ConvertPrecision lowers to the plugin's native i32.
        std::shared_ptr<ov::op::v0::Convert> out_convert;
        for (auto& consumer : reduce->output(0).get_target_inputs()) {
            if (ov::is_type<ov::op::v0::Result>(consumer.get_node())) {
                continue;
            }
            if (!out_convert) {
                out_convert = std::make_shared<ov::op::v0::Convert>(reduce, ov::element::i64);
                out_convert->set_friendly_name(reduce->get_friendly_name() + "/convert_from_i64");
                ov::copy_runtime_info(reduce, out_convert);
            }
            consumer.replace_source_output(out_convert);
        }
        return true;
    };

    auto m = std::make_shared<Matcher>(reduce_m, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_cpu
