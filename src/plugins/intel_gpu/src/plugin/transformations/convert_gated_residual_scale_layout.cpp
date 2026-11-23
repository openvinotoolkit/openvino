// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_gated_residual_scale_layout.hpp"

#include <cstdint>
#include <memory>
#include <vector>

#include "intel_gpu/op/gated_residual.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

namespace {

/// [N, G] -> [G, N]; nullptr when `src` is not a rank-2 Constant.
std::shared_ptr<ov::op::v0::Constant> transpose_to_gn(const ov::Output<ov::Node>& src) {
    const auto c = ov::as_type_ptr<ov::op::v0::Constant>(src.get_node_shared_ptr());
    if (!c || c->get_shape().size() != 2) {
        return nullptr;
    }
    const size_t N = c->get_shape()[0];
    const size_t G = c->get_shape()[1];
    const auto et = c->get_element_type();
    if (et == ov::element::f16) {
        const auto ng = c->cast_vector<ov::float16>();
        std::vector<ov::float16> gn(ng.size());
        for (size_t n = 0; n < N; ++n) {
            for (size_t g = 0; g < G; ++g) {
                gn[(g * N) + n] = ng[(n * G) + g];
            }
        }
        return ov::op::v0::Constant::create(et, ov::Shape{G, N}, gn);
    }
    // 4 / 8-bit zero points of the weight type (4-bit ones are unpacked here and packed again by Constant::create)
    const auto ng = c->cast_vector<int32_t>();
    std::vector<int32_t> gn(ng.size());
    for (size_t n = 0; n < N; ++n) {
        for (size_t g = 0; g < G; ++g) {
            gn[(g * N) + n] = ng[(n * G) + g];
        }
    }
    return ov::op::v0::Constant::create(et, ov::Shape{G, N}, gn);
}

}  // namespace

ConvertGatedResidualScaleLayout::ConvertGatedResidualScaleLayout() {
    using namespace ov::pass::pattern;
    // Only compressed READ ops that still carry the [N, G] scale / zero point layout.
    auto is_nong_compressed_read = [](const ov::Output<ov::Node>& output) {
        const auto gr = ov::as_type_ptr<ov::intel_gpu::op::GatedResidual>(output.get_node_shared_ptr());
        if (!gr || gr->get_scale_gn() || gr->get_mode() != ov::intel_gpu::op::GatedResidual::Mode::READ) {
            return false;
        }
        const auto layout = gr->get_read_layout();
        return layout && layout->compressed;
    };
    auto gr_m = wrap_type<ov::intel_gpu::op::GatedResidual>(is_nong_compressed_read);

    ov::matcher_pass_callback callback = [](Matcher& m) {
        const auto gr = ov::as_type_ptr<ov::intel_gpu::op::GatedResidual>(m.get_match_root());
        const auto layout = gr->get_read_layout();

        using Idx = ov::intel_gpu::op::GatedResidual::ReadInputIdx;
        std::vector<size_t> indices{Idx::SCALE_DOWN, Idx::SCALE_UP};
        if (layout->has_zp) {
            indices.push_back(Idx::ZP_DOWN);
            indices.push_back(Idx::ZP_UP);
        }
        if (layout->write_gate) {
            indices.push_back(layout->scale_inject);
            if (layout->has_zp) {
                indices.push_back(layout->zp_inject);
            }
        }

        // All or nothing: the op has one layout flag for every scale / zero point.
        ov::OutputVector inputs = gr->input_values();
        ov::NodeVector new_consts;
        for (const size_t i : indices) {
            const auto t = transpose_to_gn(inputs[i]);
            if (!t) {
                return false;
            }
            inputs[i] = t->output(0);
            new_consts.push_back(t);
        }

        auto new_gr = std::make_shared<ov::intel_gpu::op::GatedResidual>(inputs,
                                                                         gr->get_mode(),
                                                                         gr->get_hc_count(),
                                                                         gr->get_hidden_size(),
                                                                         gr->get_lowrank(),
                                                                         gr->get_epsilon(),
                                                                         gr->get_produce_write_gate(),
                                                                         /*scale_gn=*/true);
        new_gr->set_friendly_name(gr->get_friendly_name());
        ov::copy_runtime_info(gr, new_consts);
        ov::copy_runtime_info(gr, new_gr);
        ov::replace_node(gr, new_gr);
        return true;
    };

    auto m = std::make_shared<Matcher>(gr_m, "ConvertGatedResidualScaleLayout");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
