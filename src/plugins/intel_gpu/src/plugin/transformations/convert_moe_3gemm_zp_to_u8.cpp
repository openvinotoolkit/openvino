// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_moe_3gemm_zp_to_u8.hpp"

#include <array>
#include <memory>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

ConvertMOE3GemmZpToU8::ConvertMOE3GemmZpToU8() {
    using namespace ov::pass::pattern;

    auto moe_compressed_m = wrap_type<ov::op::internal::MOECompressed>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        auto moe_compressed = ov::as_type_ptr<ov::op::internal::MOECompressed>(m.get_match_root());
        if (!moe_compressed || transformation_callback(moe_compressed)) {
            return false;
        }
        const auto& config = moe_compressed->get_config();
        if (config.expert_type != ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU || !config.has_zp) {
            return false;
        }

        // Routed gate/up/down zp, then shared gate/up/down zp (input layout of GEMM3_SWIGLU MOECompressed).
        // Each zp directly follows its scale.
        constexpr std::array<size_t, 6> zp_indices{5, 8, 11, 14, 17, 20};
        bool changed = false;
        for (auto zp_idx : zp_indices) {
            if (zp_idx >= moe_compressed->get_input_size()) {
                continue;
            }
            ov::Output<ov::Node> zp = moe_compressed->input_value(zp_idx);
            const auto zp_orig = zp;
            if (zp.get_element_type() == ov::element::u3) {
                zp = ov::op::util::make_try_fold<ov::op::v0::Convert>(zp, ov::element::u8);
            }

            // A single zp shared by all experts (e.g. the symmetric u3 midpoint) is expanded to the scale shape:
            // the kernels and OneDNN grouped matmul take per-group zp only. Sub-byte zp is left as is.
            const auto& scale_pshape = moe_compressed->get_input_partial_shape(zp_idx - 1);
            const auto& zp_pshape = zp.get_partial_shape();
            const bool byte_zp = zp.get_element_type() == ov::element::u8 || zp.get_element_type() == ov::element::i8;
            if (byte_zp && zp_pshape.is_static() && scale_pshape.is_static() && ov::shape_size(zp_pshape.to_shape()) == 1 &&
                ov::shape_size(scale_pshape.to_shape()) > 1) {
                const auto& scale_shape = scale_pshape.to_shape();
                auto target_shape = ov::op::v0::Constant::create(ov::element::i64, {scale_shape.size()}, scale_shape);
                zp = ov::op::util::make_try_fold<ov::op::v3::Broadcast>(zp, target_shape);
            }

            if (zp != zp_orig) {
                ov::copy_runtime_info(zp_orig.get_node_shared_ptr(), zp.get_node_shared_ptr());
                moe_compressed->input(zp_idx).replace_source_output(zp);
                changed = true;
            }
        }
        return changed;
    };

    auto m = std::make_shared<Matcher>(moe_compressed_m, "ConvertMOE3GemmZpToU8");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
