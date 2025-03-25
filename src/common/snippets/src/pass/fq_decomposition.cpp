// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/fq_decomposition.hpp"

#include <numeric>

#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/validate.hpp"
#include "transformations/utils/utils.hpp"

#include "openvino/reference/autobroadcast_binop.hpp"
#include "openvino/reference/broadcast.hpp"

#include "snippets/itt.hpp"
#include "snippets/utils/utils.hpp"
#include "snippets/op/convert_saturation.hpp"


ov::snippets::pass::FakeQuantizeDecomposition::FakeQuantizeDecomposition() {
    MATCHER_SCOPE(FakeQuantizeDecomposition);

    auto fake_quantize = ov::pass::pattern::wrap_type<ov::op::v0::FakeQuantize>(
                                        OutputVector{ov::pass::pattern::any_input(),
                                        ov::pass::pattern::wrap_type<ov::op::v0::Constant>(),
                                        ov::pass::pattern::wrap_type<ov::op::v0::Constant>(),
                                        ov::pass::pattern::wrap_type<ov::op::v0::Constant>(),
                                        ov::pass::pattern::wrap_type<ov::op::v0::Constant>()});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::FakeQuantizeDecomposition")
        auto& pattern_to_output = m.get_pattern_value_map();
        const auto fake_quantize_node = ov::as_type_ptr<ov::op::v0::FakeQuantize>(
            pattern_to_output.at(fake_quantize).get_node_shared_ptr());

        if (!fake_quantize_node || transformation_callback(fake_quantize_node)) {
            return false;
        }

        OPENVINO_ASSERT(CommonFakeQuantizeDecomposition::is_supported_fq(fake_quantize_node),
                        "FQ Decomposition got invalid FakeQuantize node with the name " + fake_quantize_node->get_friendly_name());

        Output<Node> data{fake_quantize_node->input_value(0)};
        const Output<Node> input_low{fake_quantize_node->input_value(1)};
        const Output<Node> input_high{fake_quantize_node->input_value(2)};
        const Output<Node> output_low{fake_quantize_node->input_value(3)};
        const Output<Node> output_high{fake_quantize_node->input_value(4)};
        auto input_type = data.get_element_type();
        auto broadcast_type = fake_quantize_node->get_auto_broadcast();

        std::vector<float> out_scales;
        std::vector<float> cl, ch, isc, ish, osc, osh;
        const bool status = getScalesAndShifts(fake_quantize_node, cl, ch, isc, ish, osc, osh);
        if (status) {
            out_scales = calculateScales(fake_quantize_node->get_output_element_type(0), cl, ch, isc, ish, osc, osh);
        }
        const bool do_dequantize = !(status && ((std::all_of(osc.cbegin(),
                                                             osc.cend(),
                                                             [](float val) {
                                                                 return val == 1.f;
                                                             }) &&
                                                 std::all_of(osh.cbegin(),
                                                             osh.cend(),
                                                             [](float val) {
                                                                 return val == 0.f;
                                                             })) ||
                                                out_scales.size() != 0));
        const bool do_rounding = do_dequantize || fake_quantize_node->get_output_element_type(0) == ov::element::f32 ||
                                 fake_quantize_node->get_output_element_type(0) == ov::element::f16;

        ov::NodeVector decomp_ops;
        if (input_type != input_low.get_element_type()) {
            input_type = input_low.get_element_type();
            data = std::make_shared<ov::snippets::op::ConvertSaturation>(data, input_type);
            decomp_ops.push_back(data.get_node_shared_ptr());
        }

        // if we set input_low or input_high in formula we got output = output_low and output = output_high
        // respectively so we just clamp x
        const auto max = std::make_shared<ov::op::v1::Maximum>(data, input_low);
        const auto min = std::make_shared<ov::op::v1::Minimum>(max, input_high);
        decomp_ops.push_back(max);
        decomp_ops.push_back(min);

        std::shared_ptr<ov::Node> result = nullptr;
        if (out_scales.size() != 0) {
            PartialShape scale_shape = input_low.get_partial_shape();
            ov::PartialShape::broadcast_merge_into(scale_shape,
                                                       input_high.get_partial_shape(),
                                                       broadcast_type);
            const auto scales = std::make_shared<ov::op::v0::Constant>(input_low.get_element_type(),
                                                                       scale_shape.get_shape(),
                                                                       out_scales);
            decomp_ops.push_back(scales);

            result = std::make_shared<ov::op::v1::Multiply>(min, scales);
            decomp_ops.push_back(result);
        } else {
            // (levels-1)
            const auto levels_minus_one = std::make_shared<ov::op::v0::Constant>(input_low.get_element_type(),
                                                                                 Shape{},
                                                                                 fake_quantize_node->get_levels() - 1);
            decomp_ops.push_back(levels_minus_one);
            // (input_high - input_low)
            const auto subInHighLow = std::make_shared<ov::op::v1::Subtract>(input_high, input_low);
            // (levels-1) / (input_high - input_low)
            const auto isc = std::make_shared<ov::op::v1::Divide>(levels_minus_one, subInHighLow);
            // input_low * (levels-1) / (input_high - input_low)
            const auto ish = std::make_shared<ov::op::v1::Multiply>(input_low, isc);
            decomp_ops.push_back(subInHighLow);
            decomp_ops.push_back(isc);
            decomp_ops.push_back(ish);

            // x * (levels-1) / (input_high - input_low)
            const auto after_isc_apply = std::make_shared<ov::op::v1::Multiply>(min, isc);
            // x * (levels-1) / (input_high - input_low) - input_low * (levels-1) / (input_high - input_low)
            result = std::make_shared<ov::op::v1::Subtract>(after_isc_apply, ish);
            decomp_ops.push_back(after_isc_apply);
            decomp_ops.push_back(result);
        }

        if (do_rounding) {
            // round(x * (levels-1) / (input_high - input_low) - input_low * (levels-1) / (input_high - input_low))
            result = std::make_shared<ov::op::v5::Round>(result, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
            decomp_ops.push_back(result);
        }

        if (do_dequantize) {
            // (levels-1)
            const auto levels_minus_one = std::make_shared<ov::op::v0::Constant>(output_high.get_element_type(),
                                                                                 Shape{},
                                                                                 fake_quantize_node->get_levels() - 1);
            // (output_high - output_low)
            const auto sub_out_high_low = std::make_shared<ov::op::v1::Subtract>(output_high, output_low);
            // (output_high - output_low) / (levels-1)
            const auto osc = std::make_shared<ov::op::v1::Divide>(sub_out_high_low, levels_minus_one);
            decomp_ops.push_back(sub_out_high_low);
            decomp_ops.push_back(osc);

            // round(x * (levels-1) / (input_high - input_low) - input_low * (levels-1) / (input_high - input_low)) *
            // (output_high - output_low) / (levels-1)
            const auto after_osc_apply = std::make_shared<ov::op::v1::Multiply>(result, osc);
            // round(x * (levels-1) / (input_high - input_low) - input_low * (levels-1) / (input_high - input_low)) *
            // (output_high - output_low) / (levels-1) + output_low
            result = std::make_shared<ov::op::v1::Add>(after_osc_apply, output_low);
            decomp_ops.push_back(after_osc_apply);
            decomp_ops.push_back(result);
        }

        if (result->get_output_element_type(0) != fake_quantize_node->get_output_element_type(0)) {
            result = std::make_shared<snippets::op::ConvertSaturation>(result, fake_quantize_node->get_output_element_type(0));
            decomp_ops.push_back(result);
        }

        result->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(fake_quantize_node, decomp_ops);
        ov::replace_node(m.get_match_root(), result);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(fake_quantize, matcher_name);
    register_matcher(m, callback);
}

bool ov::snippets::pass::FakeQuantizeDecomposition::getScalesAndShifts(
    const std::shared_ptr<const ov::op::v0::FakeQuantize>& fq_node,
    std::vector<float>& cl,
    std::vector<float>& ch,
    std::vector<float>& isc,
    std::vector<float>& ish,
    std::vector<float>& osc,
    std::vector<float>& osh) {
    auto input_low_constant = ov::as_type_ptr<ov::op::v0::Constant>(fq_node->get_input_node_shared_ptr(1));
    auto input_high_constant = ov::as_type_ptr<ov::op::v0::Constant>(fq_node->get_input_node_shared_ptr(2));
    auto output_low_constant = ov::as_type_ptr<ov::op::v0::Constant>(fq_node->get_input_node_shared_ptr(3));
    auto output_high_constant = ov::as_type_ptr<ov::op::v0::Constant>(fq_node->get_input_node_shared_ptr(4));
    if (!input_low_constant || !input_high_constant || !output_low_constant || !output_high_constant)
        return false;

    const auto input_low_shape = input_low_constant->get_shape();
    const auto input_high_shape = input_high_constant->get_shape();
    const auto output_low_shape = output_low_constant->get_shape();
    const auto output_high_shape = output_high_constant->get_shape();

    auto input_low = input_low_constant->cast_vector<float>();
    auto input_high = input_high_constant->cast_vector<float>();
    auto output_low = output_low_constant->cast_vector<float>();
    auto output_high = output_high_constant->cast_vector<float>();
    auto levels = fq_node->get_levels();
    auto broadcast_type = fq_node->get_auto_broadcast();

    // We have two ways for computations of scales and shifts to avoid model compilation time growth
    // because common function "ov::reference::autobroadcast_binop()" is expensive:
    //  - A usual case (weights with the same shapes or scalars) - optimal calculations without large broadcasting
    //  - A rare case ("general broadcasting") - common computations using autobroadcast_binop() call with broadcasting
    //  support

    // Calculations of input scales and shift:
    //   - isc := (levels-1) / (ih - il)
    //   - ish := -il * isc
    if (input_low_shape == input_high_shape || shape_size(input_low_shape) == 1 || shape_size(input_high_shape) == 1) {
        const auto input_size = std::max(input_low.size(), input_high.size());
        isc.resize(input_size, 0);
        ish.resize(input_size, 0);
        for (size_t i = 0; i < input_size; i++) {
            float il = input_low[input_low.size() == 1 ? 0 : i];
            float ih = input_high[input_high.size() == 1 ? 0 : i];

            isc[i] = (levels - 1) / (ih - il);
            ish[i] = -il * isc[i];
        }
        cl = input_low;
        ch = input_high;
    } else {  // general broadcasting
        PartialShape scale_pshape = input_low_constant->get_output_partial_shape(0);
        PartialShape::broadcast_merge_into(scale_pshape, input_high_shape, broadcast_type);
        const auto scale_shape = scale_pshape.get_shape();
        const auto input_size = ov::shape_size(scale_shape);
        isc.resize(input_size, 0);
        ish.resize(input_size, 0);
        ov::reference::autobroadcast_binop(input_high.data(),
                                           input_low.data(),
                                           isc.data(),
                                           input_high_shape,
                                           input_low_shape,
                                           broadcast_type,
                                           [levels](float x, float y) -> float {
                                               return (levels - 1) / (x - y);
                                           });
        ov::reference::autobroadcast_binop(input_low.data(),
                                           isc.data(),
                                           ish.data(),
                                           input_low_shape,
                                           scale_shape,
                                           broadcast_type,
                                           [](float x, float y) -> float {
                                               return -x * y;
                                           });
        auto broadcast = [](const std::vector<float>& original_data, std::vector<float>& out_data,
                            const ov::Shape& original_shape, const ov::Shape& out_shape, size_t size) -> void {
            out_data.resize(size, 0);
            std::vector<size_t> broadcast_axes(out_shape.size() - original_shape.size());
            std::iota(broadcast_axes.begin(), broadcast_axes.end(), 0);
            ov::reference::broadcast(reinterpret_cast<const char*>(original_data.data()),
                                     reinterpret_cast<char*>(out_data.data()),
                                     original_shape,
                                     out_shape,
                                     broadcast_axes,
                                     sizeof(float));
        };
        broadcast(input_low, cl, input_low_shape, scale_shape, input_size);
        broadcast(input_high, ch, input_high_shape, scale_shape, input_size);
    }

    // Calculations of output scales and shift:
    //   - osc := (oh - ol) / (levels-1)
    //   - osh := ol
    if (output_low_shape == output_high_shape || shape_size(output_low_shape) == 1 || shape_size(output_high_shape) == 1) {
        const auto output_size = std::max(output_low.size(), output_high.size());
        osc.resize(output_size, 0);
        osh.resize(output_size, 0);
        for (size_t i = 0; i < output_size; i++) {
            float ol = output_low[output_low.size() == 1 ? 0 : i];
            float oh = output_high[output_high.size() == 1 ? 0 : i];

            osc[i] = (oh - ol) / (levels - 1);
            osh[i] = ol;
        }
    } else {  // general broadcasting
        PartialShape scale_pshape = output_low_constant->get_output_partial_shape(0);
        PartialShape::broadcast_merge_into(scale_pshape, output_high_constant->get_output_partial_shape(0), broadcast_type);
        const auto output_size = ov::shape_size(scale_pshape.get_shape());
        osc.resize(output_size, 0);
        ov::reference::autobroadcast_binop(output_high.data(),
                                           output_low.data(),
                                           osc.data(),
                                           output_high_shape,
                                           output_low_shape,
                                           broadcast_type,
                                           [levels](float x, float y) -> float {
                                               return (x - y) / (levels - 1);
                                           });
        osh = output_low;
    }

    return true;
}

std::vector<float> ov::snippets::pass::FakeQuantizeDecomposition::calculateScales(const ov::element::Type& out_type,
                                                                                  const std::vector<float>& cl,
                                                                                  const std::vector<float>& ch,
                                                                                  const std::vector<float>& isc,
                                                                                  const std::vector<float>& ish,
                                                                                  const std::vector<float>& osc,
                                                                                  const std::vector<float>& osh) {
    std::vector<float> out_scales;
    if (out_type == ov::element::u8 &&
        std::all_of(cl.cbegin(),
                    cl.cend(),
                    [](float val) {
                        return val == 0.0f;
                    }) &&
        std::all_of(ish.cbegin(),
                    ish.cend(),
                    [](float val) {
                        return val == 0.0f;
                    }) &&
        std::all_of(osc.cbegin(),
                    osc.cend(),
                    [](float val) {
                        return val == 1.0f;
                    }) &&
        std::all_of(osh.cbegin(), osh.cend(), [](float val) {
            return val == 0.0f;
        })) {
        out_scales = isc;
    }

    static const float thr = 0.0001f;
    if (out_type == ov::element::i8 &&
        std::all_of(ish.cbegin(), ish.cend(), [](float val) { return std::abs(val - 128.f) < thr; }) &&
        std::all_of(osc.cbegin(), osc.cend(), [](float val) { return val == 1.f; }) &&
        std::all_of(osh.cbegin(), osh.cend(), [](float val) { return std::abs(val + 128.f) < thr; })) {
        bool is_crop_aligned = true;
        for (size_t i = 0; i < std::max(cl.size(), isc.size()); i++) {
            if (std::abs(cl[cl.size() == 1 ? 0 : i] * isc[isc.size() == 1 ? 0 : i] + 128.f) > thr) {
                is_crop_aligned = false;
            }
        }

        for (size_t i = 0; i < std::max(ch.size(), isc.size()); i++) {
            if (std::abs(ch[ch.size() == 1 ? 0 : i] * isc[isc.size() == 1 ? 0 : i] - 127.f) > thr) {
                is_crop_aligned = false;
            }
        }

        if (is_crop_aligned) {
            out_scales = isc;
        }
    }

    return out_scales;
}

bool ov::snippets::pass::CommonFakeQuantizeDecomposition::is_supported_fq(const std::shared_ptr<const ov::op::v0::FakeQuantize>& fq) {
    // TODO [92179]: Add support of FakeQuantize with non-constants inputs and with binarization algorithm.
    auto is_valid_range_values = [](const std::shared_ptr<const ov::op::v0::FakeQuantize>& fq) {
        const auto il = fq->input_value(1);
        const auto ih = fq->input_value(2);
        const auto greater_equal = std::make_shared<ov::op::v1::Greater>(il, ih);

        ov::OutputVector result(1);
        if (!greater_equal->constant_fold(result, greater_equal->input_values()))
            return false;

        const auto res_node = ov::as_type_ptr<const ov::op::v0::Constant>(result[0].get_node_shared_ptr());
        const auto comp_result = res_node->cast_vector<bool>();
        return !std::any_of(comp_result.begin(), comp_result.end(), [](const bool value) {
            return value;
        });
    };
    return fq && fq->get_levels() != 2 &&
           ov::is_type<ov::op::v0::Constant>(fq->get_input_node_shared_ptr(1)) &&
           ov::is_type<ov::op::v0::Constant>(fq->get_input_node_shared_ptr(2)) &&
           ov::is_type<ov::op::v0::Constant>(fq->get_input_node_shared_ptr(3)) &&
           ov::is_type<ov::op::v0::Constant>(fq->get_input_node_shared_ptr(4)) &&
           utils::one_of(fq->get_auto_broadcast(), ov::op::AutoBroadcastType::NUMPY, ov::op::AutoBroadcastType::NONE) &&
           is_valid_range_values(fq);
}

bool ov::snippets::pass::CommonFakeQuantizeDecomposition::run_on_model(const std::shared_ptr<ov::Model>& f) {
    RUN_ON_FUNCTION_SCOPE(CommonFakeQuantizeDecomposition);
    ov::pass::Manager manager("Snippets:CommonFakeQuantizeDecomposition");
    manager.set_per_pass_validation(false);
    manager.register_pass<ov::snippets::pass::FakeQuantizeDecomposition>();
    manager.register_pass<ov::pass::ConstantFolding>();
    manager.register_pass<ov::pass::Validate>();
    manager.run_passes(f);
    return false;
}
