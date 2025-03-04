// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bcast_and_pad_zp_buffers.hpp"

#include "intel_gpu/op/convolution.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>

using namespace ov::pass::pattern;

namespace ov::intel_gpu {
namespace {

template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type align_to(T size, size_t align) {
    return static_cast<T>((size % align == 0) ? size : size - size % align + align);
}

template <typename T>
void bcast_and_pad(const T* qp_ptr, T* new_qp_ptr, size_t prev_size, size_t new_size, size_t aligned_size) {
    for (size_t i = 0; i < aligned_size; i++) {
        new_qp_ptr[i] = i < new_size ? qp_ptr[i % prev_size] : static_cast<T>(0);
    }
}

std::shared_ptr<ov::Node> pad_quantization_parameter(std::shared_ptr<ov::op::v0::Constant> qp, size_t channels_count, size_t channel_idx, size_t alignment) {
    auto type = qp->get_element_type();
    auto bcasted_shape = qp->get_shape();
    bcasted_shape[channel_idx] = channels_count;

    auto bcasted_and_padded_shape = qp->get_shape();
    bcasted_and_padded_shape[channel_idx] = align_to(channels_count, alignment);
    ov::Tensor new_qp(type, bcasted_and_padded_shape);

    size_t prev_size = ov::shape_size(qp->get_shape());
    size_t new_size = ov::shape_size(bcasted_shape);
    size_t aligned_size = ov::shape_size(bcasted_and_padded_shape);

    OPENVINO_ASSERT(prev_size <= new_size && new_size <= aligned_size);

    switch (type) {
        case ov::element::u8:
            bcast_and_pad(static_cast<const uint8_t*>(qp->get_data_ptr()), static_cast<uint8_t*>(new_qp.data()), prev_size, new_size, aligned_size);
            break;
        case ov::element::i8:
            bcast_and_pad(static_cast<const int8_t*>(qp->get_data_ptr()), static_cast<int8_t*>(new_qp.data()), prev_size, new_size, aligned_size);
            break;
        case ov::element::f16:
            bcast_and_pad(static_cast<const ov::float16*>(qp->get_data_ptr()), static_cast<ov::float16*>(new_qp.data()), prev_size, new_size, aligned_size);
            break;
        case ov::element::f32:
            bcast_and_pad(static_cast<const float*>(qp->get_data_ptr()), static_cast<float*>(new_qp.data()), prev_size, new_size, aligned_size);
            break;
        default: OPENVINO_THROW("[GPU] Can't pad quantization parameter for ", type, " element type");
    }

    return std::make_shared<ov::op::v0::Constant>(new_qp);
}

template <typename T>
bool all_same_value(const T* qp_ptr, size_t size) {
    return std::all_of(qp_ptr, qp_ptr + size, [qp_ptr](const T& val) {
        return val == qp_ptr[0];
    });
}

template <typename T>
std::shared_ptr<ov::op::v0::Constant>
create_scalar_constant(const std::shared_ptr<ov::op::v0::Constant>& qp) {
    auto type = qp->get_element_type();
    auto shape = qp->get_shape();
    if (all_same_value(static_cast<const T*>(qp->get_data_ptr()), ov::shape_size(shape))) {
        ov::Shape new_shape(shape.size(), 1);
        ov::Tensor new_tensor(type, new_shape);
        auto new_qp = std::make_shared<ov::op::v0::Constant>(new_tensor);
        auto val = qp->get_vector<T>()[0];
        new_qp->fill_data(type, val);
        return new_qp;
    }
    return nullptr;
}

std::shared_ptr<ov::Node> scalar_parameter(std::shared_ptr<ov::op::v0::Constant> qp) {
    auto type = qp->get_element_type();
    std::shared_ptr<ov::op::v0::Constant> new_qp = nullptr;

    if (type == ov::element::u8) {
        new_qp = create_scalar_constant<uint8_t>(qp);
    } else if (type == ov::element::i8) {
        new_qp = create_scalar_constant<int8_t>(qp);
    } else if (type == ov::element::f16) {
        new_qp = create_scalar_constant<ov::float16>(qp);
    } else if (type == ov::element::f32) {
        new_qp = create_scalar_constant<float>(qp);
    } else {
        OPENVINO_THROW("[GPU] Can't pad quantization parameter for ", type, " element type");
    }

    return new_qp;
}

}  // namespace

BroadcastAndPadZeroPointBuffers::BroadcastAndPadZeroPointBuffers(size_t pad_size, bool supports_immad) {
    auto input_m = any_input(type_matches_any({ov::element::u8, ov::element::i8}));
    auto weights_m = any_input(type_matches_any({ov::element::u8, ov::element::i8}));
    auto bias_m = any_input();
    auto azp_m = wrap_type<ov::op::v0::Constant, op::Placeholder>();
    auto wzp_m = wrap_type<ov::op::v0::Constant, op::Placeholder>();
    auto cmp_m = wrap_type<ov::op::v0::Constant, op::Placeholder>();

    auto convolution_m = wrap_type<op::Convolution>({ input_m, weights_m, bias_m, azp_m, wzp_m, cmp_m });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto conv = ov::as_type_ptr<op::Convolution>(m.get_match_root());
        auto in_shape = conv->get_input_partial_shape(0);
        auto out_shape = conv->get_output_partial_shape(0);
        const size_t channel_idx = 1;

        if (in_shape[channel_idx].is_dynamic() || out_shape[channel_idx].is_dynamic()) {
            return false;
        }

        if (auto azp = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(azp_m).get_node_shared_ptr())) {
            auto target_shape = azp->get_shape();
            const size_t azp_c_idx = target_shape.size() == in_shape.size() ? 1 : 0;
            auto aligned_azp = pad_quantization_parameter(azp, in_shape[channel_idx].get_length(), azp_c_idx, pad_size);
            aligned_azp->set_friendly_name(conv->get_friendly_name() + "_azp");
            ov::copy_runtime_info(azp, aligned_azp);
            conv->input(op::Convolution::Args::AZP).replace_source_output(aligned_azp);
        }

        if (auto wzp = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(wzp_m).get_node_shared_ptr())) {
            auto target_shape = wzp->get_shape();

            std::shared_ptr<ov::Node> aligned_wzp;
            if (supports_immad && !(conv->get_groups() > 1)) {
                // OneDNN supports scalar wzp. If wzp data are identical, replace it with scalar value for OneDNN
                aligned_wzp = scalar_parameter(wzp);
            }

            if (aligned_wzp == nullptr) {
                const size_t wzp_c_idx = 0;
                aligned_wzp = pad_quantization_parameter(wzp, out_shape[channel_idx].get_length(), wzp_c_idx, pad_size);
            }

            aligned_wzp->set_friendly_name(conv->get_friendly_name() + "_wzp");
            ov::copy_runtime_info(wzp, aligned_wzp);
            conv->input(op::Convolution::Args::WZP).replace_source_output(aligned_wzp);
        }

        if (auto cmp = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(cmp_m).get_node_shared_ptr())) {
            auto target_shape = cmp->get_shape();
            const size_t cmp_c_idx = target_shape.size() == out_shape.size() ? 1 : 0;
            auto aligned_cmp = pad_quantization_parameter(cmp, out_shape[channel_idx].get_length(), cmp_c_idx, pad_size);
            aligned_cmp->set_friendly_name(conv->get_friendly_name() + "_compensation");
            ov::copy_runtime_info(cmp, aligned_cmp);
            conv->input(op::Convolution::Args::COMPENSATION).replace_source_output(aligned_cmp);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(convolution_m, "BroadcastAndPadZeroPointBuffers");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
