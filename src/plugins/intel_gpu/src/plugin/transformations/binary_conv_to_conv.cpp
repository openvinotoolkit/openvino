// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "binary_conv_to_conv.hpp"
#include <memory>

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/binary_convolution.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

namespace {
template <typename DST_T>
void convert_packed_bin_to_fp(const uint8_t* src_ptr, DST_T* dst_ptr, size_t size) {
    for (size_t i = 0; i < size; i++) {
        auto val = (src_ptr[i / 8] >> (i % 8)) & 0x01;
        dst_ptr[i] = static_cast<DST_T>(val == 0 ? -1.0f : 1.0f);
    }
}
}  // namespace

ConvertBinaryConvolutionToConvolution::ConvertBinaryConvolutionToConvolution() {
    using namespace ov::pass::pattern;

    auto binary_fq = [](const Output<Node>& node) {
        auto fq = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node.get_node_shared_ptr());
        if (!fq)
            return false;

        return fq->get_levels() == 2;
    };

    auto activations_input_m = any_input();
    auto in_lo_m = wrap_type<ov::op::v0::Constant>();
    auto in_hi_m = wrap_type<ov::op::v0::Constant>();
    auto out_lo_m = wrap_type<ov::op::v0::Constant>();
    auto out_hi_m = wrap_type<ov::op::v0::Constant>();
    auto fq_m = wrap_type<ov::op::v0::FakeQuantize>({activations_input_m, in_lo_m, in_hi_m, out_lo_m, out_hi_m}, binary_fq);
    auto weights_input_m = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u1));
    auto binary_conv_m = wrap_type<ov::op::v1::BinaryConvolution>({fq_m, weights_input_m});


    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto binary_conv = ov::as_type_ptr<ov::op::v1::BinaryConvolution>(pattern_map.at(binary_conv_m).get_node_shared_ptr());
        auto activations = pattern_map.at(activations_input_m);
        auto weights = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(weights_input_m).get_node_shared_ptr());
        auto fp_element_type = activations.get_element_type();

        ov::Tensor new_weights_data(fp_element_type, weights->get_output_shape(0));
        auto src_ptr = static_cast<const uint8_t*>(weights->get_data_ptr());
        auto size = ov::shape_size(weights->get_shape());
        switch (fp_element_type) {
            case ov::element::f16: convert_packed_bin_to_fp(src_ptr, static_cast<ov::float16*>(new_weights_data.data()), size); break;
            case ov::element::f32: convert_packed_bin_to_fp(src_ptr, static_cast<float*>(new_weights_data.data()), size); break;
            default: return false;
        }

        auto new_weights_const = std::make_shared<ov::op::v0::Constant>(new_weights_data);
        auto rank = activations.get_partial_shape().size();

        auto in_lo = pattern_map.at(in_lo_m);
        auto in_hi = pattern_map.at(in_hi_m);
        auto out_lo = std::make_shared<ov::op::v0::Constant>(fp_element_type, ov::Shape(rank, 1), std::vector<float>{-1.0f});
        auto out_hi = std::make_shared<ov::op::v0::Constant>(fp_element_type, ov::Shape(rank, 1), std::vector<float>{1.0f});

        auto new_fq = std::make_shared<ov::op::v0::FakeQuantize>(activations, in_lo, in_hi, out_lo, out_hi, 2);
        std::vector<std::shared_ptr<ov::Node>> result_nodes = { new_fq };

        std::shared_ptr<ov::Node> conv_input = new_fq;
        auto pb = binary_conv->get_pads_begin();
        auto pe = binary_conv->get_pads_end();
        if (binary_conv->get_pad_value() != 0.0f) {
            pb.insert(pb.begin(), rank - pb.size(), 0);
            pe.insert(pe.begin(), rank - pe.size(), 0);
            auto pad_b = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{pb.size()}, pb);
            auto pad_e = std::make_shared<ov::op::v0::Constant>(ov::element::i32, ov::Shape{pe.size()}, pe);
            auto pad_v = std::make_shared<ov::op::v0::Constant>(fp_element_type, ov::Shape{}, std::vector<float>{binary_conv->get_pad_value()});
            auto pad = std::make_shared<ov::op::v1::Pad>(new_fq, pad_b, pad_e, pad_v, ov::op::PadMode::CONSTANT);
            conv_input = pad;

            pb = ov::CoordinateDiff(binary_conv->get_pads_begin().size(), 0);
            pe = ov::CoordinateDiff(binary_conv->get_pads_end().size(), 0);
            result_nodes.push_back(pad);
        }
        auto convolution = std::make_shared<ov::op::v1::Convolution>(conv_input,
                                                                     new_weights_const,
                                                                     binary_conv->get_strides(),
                                                                     pb,
                                                                     pe,
                                                                     binary_conv->get_dilations(),
                                                                     ov::op::PadType::EXPLICIT);

        result_nodes.push_back(convolution);
        convolution->set_friendly_name(binary_conv->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), result_nodes);
        ov::replace_node(binary_conv, convolution);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(binary_conv_m, "ConvertBinaryConvolutionToConvolution");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
