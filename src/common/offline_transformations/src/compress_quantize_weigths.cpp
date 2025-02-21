// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compress_quantize_weights.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/fake_convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/reference/autobroadcast_binop.hpp"
#include "openvino/reference/convert.hpp"
#include "openvino/reference/fake_quantize.hpp"
#include "transformations/utils/utils.hpp"

static bool has_dequantization_subgraph(const std::shared_ptr<ov::Node>& fq,
                                        std::shared_ptr<ov::Node>& convert_to_low_precision,
                                        std::shared_ptr<ov::Node>& convert_to_high_precision,
                                        std::shared_ptr<ov::Node>& zero_point);

static bool compute_scale_and_zero_point(const std::shared_ptr<ov::op::v0::Constant>& output_low,
                                         const std::shared_ptr<ov::op::v0::Constant>& output_high,
                                         size_t levels,
                                         ov::Tensor& scale_tensor,
                                         ov::Tensor& zero_point_tensor,
                                         bool& zero_point_is_zero,
                                         ov::element::Type& low_precision_type);

static std::shared_ptr<ov::op::v0::Constant> compress_quantized_weights(
    const std::shared_ptr<ov::op::v0::Constant>& weights,
    const std::shared_ptr<ov::op::v0::FakeQuantize>& fq,
    const std::shared_ptr<ov::op::v0::Constant>& input_low,
    const std::shared_ptr<ov::op::v0::Constant>& input_high,
    const std::shared_ptr<ov::op::v0::Constant>& output_low,
    const std::shared_ptr<ov::op::v0::Constant>& output_high,
    const std::shared_ptr<ov::Node>& convert,
    const std::shared_ptr<ov::Node>& zero_point,
    bool& can_fuse_zero_point);

static std::shared_ptr<ov::op::v0::Constant> compress_quantized_weights(
    const std::shared_ptr<ov::op::v0::Constant>& weights,
    const std::shared_ptr<ov::op::v0::Constant>& input_low,
    const std::shared_ptr<ov::op::v0::Constant>& input_high,
    const ov::element::Type& low_precision_type,
    size_t levels,
    bool zero_point_is_zero,
    const ov::Tensor& zero_point_tensor,
    bool& can_fuse_zero_point);

static void replace_with_dequantize_subgraph(const std::shared_ptr<ov::op::v0::FakeQuantize>& fq,
                                             const std::shared_ptr<ov::op::v0::Constant>& new_weights,
                                             const ov::element::Type& high_precision_type,
                                             const ov::Shape& scale_or_zero_point_shape,
                                             const ov::Tensor& scale_tensor,
                                             bool zero_point_is_zero,
                                             const ov::Tensor& zero_point_tensor = {});

template <typename T>
bool is_non_negative(const T* data, const ov::Shape& shape) {
    bool non_neg = true;
    for (size_t i = 0; i < ov::shape_size(shape); ++i) {
        non_neg &= (data[i] >= 0);
        if (!non_neg) {
            return false;
        }
    }
    return true;
}

ov::pass::CompressWeightsWithFakeQuantize::CompressWeightsWithFakeQuantize() {
    auto weights_const_pattern = pattern::wrap_type<op::v0::Constant>();
    auto weights_convert_pattern = pattern::wrap_type<op::v0::Convert>({weights_const_pattern});
    OutputVector weights_options{weights_const_pattern, weights_convert_pattern};
    auto weights_pattern = std::make_shared<pattern::op::Or>(weights_options);
    auto input_low_pattern = pattern::wrap_type<op::v0::Constant>();
    auto input_high_pattern = pattern::wrap_type<op::v0::Constant>();
    auto output_low_pattern = pattern::wrap_type<op::v0::Constant>();
    auto output_high_pattern = pattern::wrap_type<op::v0::Constant>();
    auto fq_pattern = pattern::wrap_type<op::v0::FakeQuantize>(
        {weights_pattern, input_low_pattern, input_high_pattern, output_low_pattern, output_high_pattern});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto fq = ov::as_type_ptr<op::v0::FakeQuantize>(m.get_match_root());
        if (!fq)
            return false;
        const auto& high_precision_type = fq->get_element_type();

        auto weights = ov::util::constantfold_subgraph(fq->get_input_node_shared_ptr(0));
        if (!weights)
            return false;
        auto input_low = ov::as_type_ptr<op::v0::Constant>(fq->get_input_node_shared_ptr(1));
        if (!input_low)
            return false;
        auto input_high = ov::as_type_ptr<op::v0::Constant>(fq->get_input_node_shared_ptr(2));
        if (!input_high)
            return false;
        auto output_low = ov::as_type_ptr<op::v0::Constant>(fq->get_input_node_shared_ptr(3));
        if (!output_low)
            return false;
        auto output_high = ov::as_type_ptr<op::v0::Constant>(fq->get_input_node_shared_ptr(4));
        if (!output_high)
            return false;

        // skip dequantize part if there is already dequantization subgraph after FakeQuantize
        std::shared_ptr<Node> convert_to_low_precision;
        std::shared_ptr<Node> convert_to_high_precision;
        std::shared_ptr<Node> zero_point;
        if (has_dequantization_subgraph(fq, convert_to_low_precision, convert_to_high_precision, zero_point)) {
            bool can_fuse_zero_point = false;
            auto new_weights = compress_quantized_weights(weights,
                                                          fq,
                                                          input_low,
                                                          input_high,
                                                          output_low,
                                                          output_high,
                                                          convert_to_low_precision,
                                                          zero_point,
                                                          can_fuse_zero_point);
            if (!new_weights)
                return false;

            new_weights->set_friendly_name(convert_to_low_precision->get_friendly_name());
            replace_node(convert_to_low_precision, new_weights);
            copy_runtime_info({fq, convert_to_low_precision}, new_weights);
            // preserve dequantization subgraph for LP transformations
            ov::pass::disable_constant_folding(convert_to_high_precision);
            if (can_fuse_zero_point) {
                auto subtract = convert_to_high_precision->get_users()[0];
                auto subtract_consumers = subtract->output(0).get_target_inputs();
                auto multiply = *(subtract_consumers.begin());
                multiply.replace_source_output(convert_to_high_precision);
            }
            return true;
        } else {
            /*
               Quantize part

               Prepare new FakeQuantize that performs weights quantization.
               In this case input_low/high stays the same, but we need new output_low/high:
                 output_low = -levels / 2
                 output_high = levels - 1 + output_low
               The FakeQuantize result is converted to low precision type and then constant folded

               Dequantize part is performed by Convert(from low to high precision)->Subtract->Multiply subgraph.

                                 +-------------------------+
                                 |         Convert         |
                                 | (from low to high prec) |
                                 +-------------------------+
                                              |
                                              v
                        +----------+    +------------+
                        |zero point|--->|  Subtract  |
                        +----------+    +-----+------+
                                              |
                                              v
                         +---------+    +------------+
                         |  scale  |--->|  Multiply  |
                         +---------+    +-----+------+
                                              |
                                              v

                where:
                    scale = (output_high - output_low) / (new_output_high - new_output_low)
                    zero_point = new_output_low - output_low / scale
            */

            auto levels = fq->get_levels();
            if (levels <= 2 || levels > 256)
                return false;
            auto low_precision_type = element::dynamic;
            // Currently we support two weights quantize types: i4, u4, i8, u8
            // we determine that the weights should be cast to u4, u8 inside compute_scale_and_zero_point
            if (levels <= 16) {
                low_precision_type = element::i4;
            } else if (levels <= 256) {
                low_precision_type = element::i8;
            }

            bool zero_point_is_zero = true;
            PartialShape merged_shape{output_low->get_shape()};
            PartialShape::broadcast_merge_into(merged_shape, output_high->get_shape(), op::AutoBroadcastType::NUMPY);
            Shape scale_or_zero_point_shape = merged_shape.to_shape();
            Tensor scale_tensor(high_precision_type, scale_or_zero_point_shape);
            Tensor zero_point_tensor(high_precision_type, scale_or_zero_point_shape);

            if (!compute_scale_and_zero_point(output_low,
                                              output_high,
                                              levels,
                                              scale_tensor,
                                              zero_point_tensor,
                                              zero_point_is_zero,
                                              low_precision_type)) {
                return false;
            }

            bool can_fuse_zero_point = false;
            auto new_weights = compress_quantized_weights(weights,
                                                          input_low,
                                                          input_high,
                                                          low_precision_type,
                                                          levels,
                                                          zero_point_is_zero,
                                                          zero_point_tensor,
                                                          can_fuse_zero_point);
            if (!new_weights) {
                return false;
            }

            if (zero_point_is_zero || can_fuse_zero_point) {
                replace_with_dequantize_subgraph(fq,
                                                 new_weights,
                                                 high_precision_type,
                                                 scale_or_zero_point_shape,
                                                 scale_tensor,
                                                 true);
            } else {
                replace_with_dequantize_subgraph(fq,
                                                 new_weights,
                                                 high_precision_type,
                                                 scale_or_zero_point_shape,
                                                 scale_tensor,
                                                 zero_point_is_zero,
                                                 zero_point_tensor);
            }

            return true;
        }
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(fq_pattern, "CompressWeightsWithFakeQuantize");
    register_matcher(m, callback);
}

static std::shared_ptr<ov::op::v0::Constant> get_fake_convert_shift(
    const std::shared_ptr<ov::op::v13::FakeConvert>& fake_convert) {
    if (fake_convert->get_input_size() < 3)
        return nullptr;
    const auto shift = ov::as_type_ptr<ov::op::v0::Constant>(fake_convert->get_input_node_shared_ptr(2));
    if (!shift)
        return nullptr;
    float value = -1.0f;
    if (!ov::op::util::get_single_value(shift, value) || value != 0.0f)
        return shift;
    return nullptr;
}

ov::pass::CompressWeightsWithFakeConvert::CompressWeightsWithFakeConvert() {
    auto weights_const_pattern = pattern::wrap_type<op::v0::Constant>();
    auto weights_convert_pattern = pattern::wrap_type<op::v0::Convert>({weights_const_pattern});
    OutputVector weights_options{weights_const_pattern, weights_convert_pattern};
    auto weights_pattern = std::make_shared<pattern::op::Or>(weights_options);
    auto fake_convert_pattern = pattern::wrap_type<op::v13::FakeConvert>(
        {weights_pattern, pattern::wrap_type<op::v0::Constant>(), pattern::wrap_type<op::v0::Constant>()});
    auto fake_convert_pattern2 =
        pattern::wrap_type<op::v13::FakeConvert>({weights_pattern, pattern::wrap_type<op::v0::Constant>()});
    auto root = std::make_shared<pattern::op::Or>(OutputVector{fake_convert_pattern, fake_convert_pattern2});

    matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_map();
        const auto fake_convert = ov::as_type_ptr<op::v13::FakeConvert>(m.get_match_root());
        auto weights = pattern_map.at(weights_const_pattern);

        NodeVector from{weights, fake_convert, fake_convert->get_input_node_shared_ptr(1)};
        NodeRegistry node_registry;

        if (weights->get_output_element_type(0) != fake_convert->get_input_element_type(0)) {
            weights = std::make_shared<op::v0::Convert>(weights, fake_convert->get_input_element_type(0));
        }
        const auto scale = fake_convert->input_value(1);
        weights = std::make_shared<op::v1::Multiply>(weights, scale);
        const auto shift = get_fake_convert_shift(fake_convert);
        if (shift) {
            from.push_back(shift);
            weights = std::make_shared<op::v1::Subtract>(weights, shift);
        }
        const auto destination_type = element::Type(fake_convert->get_destination_type());
        const auto weights_convert = std::make_shared<op::v0::Convert>(weights, destination_type);
        auto compressed_weights = ov::util::constantfold_subgraph(weights_convert);
        if (!compressed_weights) {
            return false;
        }
        node_registry.add(compressed_weights);

        const auto convert =
            node_registry.make<op::v0::Convert>(compressed_weights, fake_convert->get_input_element_type(0));
        const auto inv_scale = ov::util::constantfold_subgraph(
            std::make_shared<op::v1::Power>(scale,
                                            op::v0::Constant::create(scale.get_element_type(), Shape{}, {-1.0f})));
        if (!inv_scale)
            return false;
        node_registry.add(inv_scale);
        std::shared_ptr<op::v1::Multiply> multiply;
        if (shift) {
            // TODO: check if shift can be fused to weights and eliminate it
            const auto neg_shift = ov::util::constantfold_subgraph(std::make_shared<op::v0::Negative>(shift));
            if (!neg_shift)
                return false;
            node_registry.add(neg_shift);
            const auto subtract = node_registry.make<op::v1::Subtract>(convert, neg_shift);
            multiply = node_registry.make<op::v1::Multiply>(subtract, inv_scale);
        } else {
            multiply = node_registry.make<op::v1::Multiply>(convert, inv_scale);
        }

        compressed_weights->set_friendly_name(weights->get_friendly_name());
        multiply->set_friendly_name(fake_convert->get_friendly_name());

        copy_runtime_info(from, node_registry.get());

        replace_node(fake_convert, multiply);

        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(root, "CompressWeightsWithFakeConvert");
    register_matcher(m, callback);
}

static ov::Tensor tensor_from_constant(const std::shared_ptr<ov::op::v0::Constant>& constant) {
    return ov::Tensor(constant->get_element_type(), constant->get_shape(), const_cast<void*>(constant->get_data_ptr()));
}

static bool evaluate_node(const std::shared_ptr<ov::Node>& node,
                          const ov::TensorVector& input_tensors,
                          ov::Tensor& output_tensor) {
    if (node->get_output_size() != 1)
        return false;

    ov::TensorVector output_tensors{ov::Tensor(node->get_output_element_type(0), node->get_output_shape(0))};
    if (!node->evaluate(output_tensors, input_tensors))
        return false;

    output_tensor = output_tensors[0];

    return true;
}

static ov::TensorVector get_fake_quantize_input_tensors(const std::shared_ptr<ov::Node>& fq) {
    ov::Tensor weights_tensor;

    auto fq_input = fq->get_input_node_shared_ptr(0);
    auto fq_input_constant = ov::as_type_ptr<ov::op::v0::Constant>(fq_input);

    if (!fq_input_constant) {
        auto weights = ov::as_type_ptr<ov::op::v0::Constant>(fq_input->get_input_node_shared_ptr(0));
        if (!evaluate_node(fq_input, ov::TensorVector{tensor_from_constant(weights)}, weights_tensor))
            return {};
    } else {
        weights_tensor = tensor_from_constant(fq_input_constant);
    }

    auto in_low = ov::as_type_ptr<ov::op::v0::Constant>(fq->get_input_node_shared_ptr(1));
    auto in_high = ov::as_type_ptr<ov::op::v0::Constant>(fq->get_input_node_shared_ptr(2));
    auto out_low = ov::as_type_ptr<ov::op::v0::Constant>(fq->get_input_node_shared_ptr(3));
    auto out_high = ov::as_type_ptr<ov::op::v0::Constant>(fq->get_input_node_shared_ptr(4));

    return ov::TensorVector{weights_tensor,
                            tensor_from_constant(in_low),
                            tensor_from_constant(in_high),
                            tensor_from_constant(out_low),
                            tensor_from_constant(out_high)};
}

template <typename T>
static std::shared_ptr<ov::Node> get_single_consumer_of_type(const std::shared_ptr<ov::Node>& node) {
    auto target_inputs = node->output(0).get_target_inputs();
    if (target_inputs.size() != 1)
        return nullptr;
    auto consumer = ov::as_type<T>(target_inputs.begin()->get_node());
    if (!consumer)
        return nullptr;
    return consumer->shared_from_this();
}

bool has_dequantization_subgraph(const std::shared_ptr<ov::Node>& fq,
                                 std::shared_ptr<ov::Node>& convert_to_low_precision,
                                 std::shared_ptr<ov::Node>& convert_to_high_precision,
                                 std::shared_ptr<ov::Node>& zero_point) {
    convert_to_low_precision = get_single_consumer_of_type<ov::op::v0::Convert>(fq);
    if (!convert_to_low_precision)
        return false;
    convert_to_high_precision = get_single_consumer_of_type<ov::op::v0::Convert>(convert_to_low_precision);
    if (!convert_to_high_precision)
        return false;
    auto subtract = get_single_consumer_of_type<ov::op::v1::Subtract>(convert_to_high_precision);
    if (subtract) {
        zero_point = subtract->get_input_node_shared_ptr(1);
        return get_single_consumer_of_type<ov::op::v1::Multiply>(subtract) != nullptr;
    } else {
        return get_single_consumer_of_type<ov::op::v1::Multiply>(convert_to_high_precision) != nullptr;
    }
}

static std::shared_ptr<ov::op::v0::Constant> evaluate_fake_quantize(const std::shared_ptr<ov::Node>& quantize,
                                                                    const std::shared_ptr<ov::Node>& convert) {
    ov::Tensor quantize_output_tensor;
    if (!evaluate_node(quantize, get_fake_quantize_input_tensors(quantize), quantize_output_tensor))
        return nullptr;
    ov::Tensor new_weights_tensor;
    if (!evaluate_node(convert, {quantize_output_tensor}, new_weights_tensor))
        return nullptr;
    return std::make_shared<ov::op::v0::Constant>(new_weights_tensor);
}

void replace_with_dequantize_subgraph(const std::shared_ptr<ov::op::v0::FakeQuantize>& fq,
                                      const std::shared_ptr<ov::op::v0::Constant>& new_weights,
                                      const ov::element::Type& high_precision_type,
                                      const ov::Shape& scale_or_zero_point_shape,
                                      const ov::Tensor& scale_tensor,
                                      bool zero_point_is_zero,
                                      const ov::Tensor& zero_point_tensor) {
    ov::pass::NodeRegistry node_registry;
    auto convert = node_registry.make<ov::op::v0::Convert>(new_weights, high_precision_type);
    ov::pass::disable_constant_folding(convert);
    std::shared_ptr<ov::op::v1::Multiply> mul;
    auto scale = node_registry.make<ov::op::v0::Constant>(scale_tensor);
    if (!zero_point_is_zero) {
        auto zero_point = node_registry.make<ov::op::v0::Constant>(zero_point_tensor);
        auto sub = node_registry.make<ov::op::v1::Subtract>(convert, zero_point);
        mul = node_registry.make<ov::op::v1::Multiply>(sub, scale);
    } else {
        mul = node_registry.make<ov::op::v1::Multiply>(convert, scale);
    }
    mul->set_friendly_name(fq->get_friendly_name());
    copy_runtime_info(fq, node_registry.get());
    replace_node(fq, mul);
}

template <typename T>
static void compute_scale_and_zero_point_internal(const std::shared_ptr<ov::op::v0::Constant>& output_low,
                                                  const std::shared_ptr<ov::op::v0::Constant>& output_high,
                                                  size_t levels,
                                                  ov::Tensor& scale_tensor,
                                                  ov::Tensor& zero_point_tensor,
                                                  bool& zero_point_is_zero,
                                                  ov::element::Type& low_precision_type) {
    // we consider that if all output low and output high values are non negative then unsigned int (u8, u4) need to be
    // used.
    bool out_low_non_neg = is_non_negative(output_low->get_data_ptr<T>(), output_low->get_shape()) &&
                           is_non_negative(output_high->get_data_ptr<T>(), output_high->get_shape());
    zero_point_is_zero = true;
    float new_output_low;
    if (out_low_non_neg) {
        new_output_low = 0;
        low_precision_type = (low_precision_type == ov::element::i8) ? ov::element::u8 : ov::element::u4;
    } else {
        new_output_low = -static_cast<float>(levels / 2);
    }

    float input_range = static_cast<float>(levels - 1);
    T* zero_point = zero_point_tensor.data<T>();
    T* scale = scale_tensor.data<T>();
    ov::reference::autobroadcast_binop(
        output_low->get_data_ptr<T>(),
        output_high->get_data_ptr<T>(),
        scale,
        output_low->get_shape(),
        output_high->get_shape(),
        ov::op::AutoBroadcastType::NUMPY,
        [input_range, new_output_low, zero_point, &zero_point_is_zero](float output_low_value,
                                                                       float output_high_value) mutable {
            float output_range = output_high_value - output_low_value;
            float scale = output_range / input_range;
            float zero_point_value = (scale != 0) ? (new_output_low - output_low_value / scale) : 0;
            zero_point_is_zero =
                zero_point_is_zero && std::fabs(zero_point_value) < std::numeric_limits<float>::epsilon();
            *zero_point++ = zero_point_value;
            return scale;
        });
}

bool compute_scale_and_zero_point(const std::shared_ptr<ov::op::v0::Constant>& output_low,
                                  const std::shared_ptr<ov::op::v0::Constant>& output_high,
                                  size_t levels,
                                  ov::Tensor& scale_tensor,
                                  ov::Tensor& zero_point_tensor,
                                  bool& zero_point_is_zero,
                                  ov::element::Type& low_precision_type) {
    const auto type = output_low->get_element_type();
    switch (type) {
    case ov::element::Type_t::f32: {
        compute_scale_and_zero_point_internal<float>(output_low,
                                                     output_high,
                                                     levels,
                                                     scale_tensor,
                                                     zero_point_tensor,
                                                     zero_point_is_zero,
                                                     low_precision_type);
        break;
    }
    case ov::element::f16: {
        compute_scale_and_zero_point_internal<ov::float16>(output_low,
                                                           output_high,
                                                           levels,
                                                           scale_tensor,
                                                           zero_point_tensor,
                                                           zero_point_is_zero,
                                                           low_precision_type);
        break;
    }
    case ov::element::bf16: {
        compute_scale_and_zero_point_internal<ov::bfloat16>(output_low,
                                                            output_high,
                                                            levels,
                                                            scale_tensor,
                                                            zero_point_tensor,
                                                            zero_point_is_zero,
                                                            low_precision_type);
        break;
    }

    default:
        return false;
    }

    return true;
}

template <typename T, typename U, typename F>
static void
transform(const T* first1, const T* const last1, const T* first2, const T* first3, const T* first4, U* out, F& f) {
    while (first1 < last1) {
        *out++ = f(*first1++, *first2++, *first3++, *first4++);
    }
}

template <typename T, typename U, typename F>
static void transform(const T* first1,
                      const T* const last1,
                      const T* first2,
                      const T* first3,
                      const T* first4,
                      const T* first5,
                      const T* first6,
                      U* out,
                      F& f) {
    while (first1 < last1) {
        *out++ = f(*first1++, *first2++, *first3++, *first4++, *first5++, *first6++);
    }
}

template <typename T, typename U, typename F>
static void numpy_broadcast_4inputs(const T* weights,
                                    const ov::Shape& weights_shape,
                                    const T* in_low,
                                    const ov::Shape& in_low_shape,
                                    const T* in_high,
                                    const ov::Shape& in_high_shape,
                                    const T* zero_point,
                                    const ov::Shape& zero_point_shape,
                                    U* new_weights,
                                    F& f) {
    using namespace ov::reference::fake_quantize_details;

    std::vector<size_t> output_strides = compute_strides(weights_shape, weights_shape);
    std::vector<size_t> in_low_strides = compute_strides(weights_shape, in_low_shape);
    std::vector<size_t> in_high_strides = compute_strides(weights_shape, in_high_shape);
    std::vector<size_t> zero_point_strides = compute_strides(weights_shape, zero_point_shape);

    size_t num_elements = shape_size(weights_shape);

    size_t weights_inner_stride = num_elements;
    size_t in_low_inner_stride = 0;
    size_t in_high_inner_stride = 0;
    size_t zero_point_inner_stride = 0;

    std::tie(in_low_inner_stride, weights_inner_stride) =
        get_inner_stride(num_elements, weights_shape, in_low_shape, weights_inner_stride);
    std::tie(in_high_inner_stride, weights_inner_stride) =
        get_inner_stride(num_elements, weights_shape, in_high_shape, weights_inner_stride);
    std::tie(zero_point_inner_stride, weights_inner_stride) =
        get_inner_stride(num_elements, weights_shape, zero_point_shape, weights_inner_stride);

    auto get_outer_strides =
        [&output_strides, &in_low_strides, &in_high_strides, &zero_point_strides](size_t flat_index) {
            size_t in_low_stride = 0;
            size_t in_high_stride = 0;
            size_t zero_point_stride = 0;

            for (size_t i = 0; i < output_strides.size(); i++) {
                size_t div = flat_index / output_strides[i];
                flat_index = flat_index % output_strides[i];
                in_low_stride += div * in_low_strides[i];
                in_high_stride += div * in_high_strides[i];
                zero_point_stride += div * zero_point_strides[i];
            }

            return std::tuple<size_t, size_t, size_t>{in_low_stride, in_high_stride, zero_point_stride};
        };

    size_t in_low_stride = 0;
    size_t in_high_stride = 0;
    size_t zero_point_stride = 0;

    if (in_low_inner_stride * in_high_inner_stride * zero_point_inner_stride == 1) {
        for (size_t i = 0; i < shape_size(weights_shape); i += weights_inner_stride) {
            std::tie(in_low_stride, in_high_stride, zero_point_stride) = get_outer_strides(i);
            T in_low_scalar = *(in_low + in_low_stride);
            T in_high_scalar = *(in_high + in_high_stride);
            T zero_point_scalar = *(zero_point + zero_point_stride);
            std::transform(weights,
                           weights + weights_inner_stride,
                           new_weights,
                           [in_low_scalar, in_high_scalar, zero_point_scalar, &f](T w) {
                               return f(w, in_low_scalar, in_high_scalar, zero_point_scalar);
                           });
            weights += weights_inner_stride;
            new_weights += weights_inner_stride;
        }
    } else if (in_low_inner_stride > 1 && in_high_inner_stride > 1 && zero_point_inner_stride > 1) {
        for (size_t i = 0; i < shape_size(weights_shape); i += weights_inner_stride) {
            std::tie(in_low_stride, in_high_stride, zero_point_stride) = get_outer_strides(i);
            transform(weights,
                      weights + weights_inner_stride,
                      in_low + in_low_stride,
                      in_high + in_high_stride,
                      zero_point + zero_point_stride,
                      new_weights,
                      f);
            weights += weights_inner_stride;
            new_weights += weights_inner_stride;
        }
    } else {
        for (size_t i = 0; i < shape_size(weights_shape); i++) {
            std::tie(in_low_stride, in_high_stride, zero_point_stride) = get_outer_strides(i);
            *new_weights++ = f(*weights++,
                               *(in_low + in_low_stride),
                               *(in_high + in_high_stride),
                               *(zero_point + zero_point_stride));
        }
    }
}

template <typename T, typename U, typename F>
static void numpy_broadcast_6inputs(const T* weights,
                                    const ov::Shape& weights_shape,
                                    const T* in_low,
                                    const ov::Shape& in_low_shape,
                                    const T* in_high,
                                    const ov::Shape& in_high_shape,
                                    const T* out_low,
                                    const ov::Shape& out_low_shape,
                                    const T* out_high,
                                    const ov::Shape& out_high_shape,
                                    const T* zero_point,
                                    const ov::Shape& zero_point_shape,
                                    U* new_weights,
                                    F& f) {
    using namespace ov::reference::fake_quantize_details;

    std::vector<size_t> output_strides = compute_strides(weights_shape, weights_shape);
    std::vector<size_t> in_low_strides = compute_strides(weights_shape, in_low_shape);
    std::vector<size_t> in_high_strides = compute_strides(weights_shape, in_high_shape);
    std::vector<size_t> out_low_strides = compute_strides(weights_shape, out_low_shape);
    std::vector<size_t> out_high_strides = compute_strides(weights_shape, out_high_shape);
    std::vector<size_t> zero_point_strides = compute_strides(weights_shape, zero_point_shape);

    auto get_outer_strides =
        [&output_strides, &in_low_strides, &in_high_strides, &out_low_strides, &out_high_strides, &zero_point_strides](
            size_t flat_index) {
            size_t in_low_stride = 0;
            size_t in_high_stride = 0;
            size_t out_low_stride = 0;
            size_t out_high_stride = 0;
            size_t zero_point_stride = 0;

            for (size_t i = 0; i < output_strides.size(); i++) {
                size_t div = flat_index / output_strides[i];
                flat_index = flat_index % output_strides[i];
                in_low_stride += div * in_low_strides[i];
                in_high_stride += div * in_high_strides[i];
                out_low_stride += div * out_low_strides[i];
                out_high_stride += div * out_high_strides[i];
                zero_point_stride += div * zero_point_strides[i];
            }

            return std::tuple<size_t, size_t, size_t, size_t, size_t>{in_low_stride,
                                                                      in_high_stride,
                                                                      out_low_stride,
                                                                      out_high_stride,
                                                                      zero_point_stride};
        };

    size_t in_low_stride = 0;
    size_t in_high_stride = 0;
    size_t out_low_stride = 0;
    size_t out_high_stride = 0;
    size_t zero_point_stride = 0;

    for (size_t i = 0; i < shape_size(weights_shape); i++) {
        std::tie(in_low_stride, in_high_stride, out_low_stride, out_high_stride, zero_point_stride) =
            get_outer_strides(i);
        *new_weights = f(*weights++,
                         *(in_low + in_low_stride),
                         *(in_high + in_high_stride),
                         *(out_low + out_low_stride),
                         *(out_high + out_high_stride),
                         *(zero_point + zero_point_stride));
        new_weights++;
    }
}

static inline uint8_t convert_to_uint8(float val) {
    return static_cast<uint8_t>(std::nearbyint(val));
}

static inline uint8_t convert_to_uint4(float val) {
    return static_cast<uint8_t>(std::nearbyint(val)) & 0x0f;
}

static inline int8_t convert_to_int8(float val) {
    return static_cast<int8_t>(std::nearbyint(val));
}

static inline int8_t convert_to_int4(float val) {
    return static_cast<int8_t>(std::nearbyint(val)) & 0x0f;
}

static std::shared_ptr<ov::op::v0::Constant> create_weights_constant(const ov::Tensor& weights_tensor,
                                                                     const ov::element::Type& type) {
    auto weights = std::make_shared<ov::op::v0::Constant>(weights_tensor);
    if (weights->get_element_type() != type) {
        return ov::util::constantfold_subgraph(std::make_shared<ov::op::v0::Convert>(weights, type));
    }
    return weights;
}

template <typename T>
static std::shared_ptr<ov::op::v0::Constant> compress_quantized_weights_internal(
    const ov::element::Type& low_precision_type,
    const T* weights,
    const ov::Shape& weights_shape,
    const T* input_low,
    const ov::Shape& input_low_shape,
    const T* input_high,
    const ov::Shape& input_high_shape,
    const T* output_low,
    const ov::Shape& output_low_shape,
    const T* output_high,
    const ov::Shape& output_high_shape,
    const T* zero_point,
    const ov::Shape& zero_point_shape,
    size_t levels,
    bool& can_fuse_zero_point) {
    ov::element::Type new_low_precision_type = low_precision_type;
    bool out_not_neg = is_non_negative(output_low, output_low_shape) && is_non_negative(output_high, output_high_shape);

    if (low_precision_type == ov::element::i8 && out_not_neg) {
        new_low_precision_type = ov::element::u8;
    } else if (low_precision_type == ov::element::i4 && out_not_neg) {
        new_low_precision_type = ov::element::u4;
    }

    ov::element::Type tensor_el_type;
    if (new_low_precision_type == ov::element::i8 || new_low_precision_type == ov::element::i4) {
        tensor_el_type = ov::element::i8;
    } else if (new_low_precision_type == ov::element::u8 || new_low_precision_type == ov::element::u4) {
        tensor_el_type = ov::element::u8;
    }

    ov::Tensor compressed_weights_tensor(tensor_el_type, weights_shape);
    ov::Tensor compressed_weights_with_fused_zero_point_tensor(tensor_el_type, weights_shape);

    // TODO: reuse the common code parts
    if (tensor_el_type == ov::element::u8) {
        auto* compressed_weights = compressed_weights_tensor.data<uint8_t>();
        auto* compressed_weights_with_fused_zero_point =
            compressed_weights_with_fused_zero_point_tensor.data<uint8_t>();
        T levels_minus_one = static_cast<T>(levels - 1);
        can_fuse_zero_point = true;
        const auto convert_to_low_precision =
            low_precision_type == ov::element::u4 ? convert_to_uint4 : convert_to_uint8;

        auto f = [compressed_weights_with_fused_zero_point,
                  levels_minus_one,
                  convert_to_low_precision,
                  &can_fuse_zero_point](T weights_value,
                                        T input_low,
                                        T input_high,
                                        T output_low,
                                        T output_high,
                                        T zero_point) mutable {
            uint8_t compressed_weights_value =
                convert_to_low_precision(ov::reference::fake_quantize_details::quantize(weights_value,
                                                                                        input_low,
                                                                                        input_high,
                                                                                        output_low,
                                                                                        output_high,
                                                                                        levels_minus_one));
            T weights_minus_zero_point = static_cast<T>(compressed_weights_value) - zero_point;
            uint8_t compressed_weights_with_fused_zero_point_value = convert_to_low_precision(weights_minus_zero_point);
            can_fuse_zero_point &=
                std::fabs(compressed_weights_with_fused_zero_point_value - weights_minus_zero_point) < 1e-4;
            *compressed_weights_with_fused_zero_point++ = compressed_weights_with_fused_zero_point_value;
            return compressed_weights_value;
        };

        numpy_broadcast_6inputs(weights,
                                weights_shape,
                                input_low,
                                input_low_shape,
                                input_high,
                                input_high_shape,
                                output_low,
                                output_low_shape,
                                output_high,
                                output_high_shape,
                                zero_point,
                                zero_point_shape,
                                compressed_weights,
                                f);
    } else if (tensor_el_type == ov::element::i8) {
        auto* compressed_weights = compressed_weights_tensor.data<int8_t>();
        auto* compressed_weights_with_fused_zero_point = compressed_weights_with_fused_zero_point_tensor.data<int8_t>();
        T levels_minus_one = static_cast<T>(levels - 1);
        can_fuse_zero_point = true;
        const auto convert_to_low_precision = low_precision_type == ov::element::i4 ? convert_to_int4 : convert_to_int8;

        auto f = [compressed_weights_with_fused_zero_point,
                  levels_minus_one,
                  convert_to_low_precision,
                  &can_fuse_zero_point](T weights_value,
                                        T input_low,
                                        T input_high,
                                        T output_low,
                                        T output_high,
                                        T zero_point) mutable {
            int8_t compressed_weights_value =
                convert_to_low_precision(ov::reference::fake_quantize_details::quantize(weights_value,
                                                                                        input_low,
                                                                                        input_high,
                                                                                        output_low,
                                                                                        output_high,
                                                                                        levels_minus_one));
            T weights_minus_zero_point = static_cast<T>(compressed_weights_value) - zero_point;
            int8_t compressed_weights_with_fused_zero_point_value = convert_to_low_precision(weights_minus_zero_point);
            can_fuse_zero_point &=
                std::fabs(compressed_weights_with_fused_zero_point_value - weights_minus_zero_point) < 1e-4;
            *compressed_weights_with_fused_zero_point++ = compressed_weights_with_fused_zero_point_value;
            return compressed_weights_value;
        };

        numpy_broadcast_6inputs(weights,
                                weights_shape,
                                input_low,
                                input_low_shape,
                                input_high,
                                input_high_shape,
                                output_low,
                                output_low_shape,
                                output_high,
                                output_high_shape,
                                zero_point,
                                zero_point_shape,
                                compressed_weights,
                                f);
    }

    return create_weights_constant(
        can_fuse_zero_point ? compressed_weights_with_fused_zero_point_tensor : compressed_weights_tensor,
        new_low_precision_type);
}

std::shared_ptr<ov::op::v0::Constant> compress_quantized_weights(
    const std::shared_ptr<ov::op::v0::Constant>& weights,
    const std::shared_ptr<ov::op::v0::FakeQuantize>& fq,
    const std::shared_ptr<ov::op::v0::Constant>& input_low,
    const std::shared_ptr<ov::op::v0::Constant>& input_high,
    const std::shared_ptr<ov::op::v0::Constant>& output_low,
    const std::shared_ptr<ov::op::v0::Constant>& output_high,
    const std::shared_ptr<ov::Node>& convert,
    const std::shared_ptr<ov::Node>& zero_point,
    bool& can_fuse_zero_point) {
    const auto& weights_shape = weights->get_shape();
    const auto& type = weights->get_element_type();
    const auto& low_precision_type = convert->get_output_element_type(0);

    if (zero_point == nullptr)
        return evaluate_fake_quantize(fq, convert);

    auto zero_point_constant = ov::util::constantfold_subgraph(zero_point);
    if (!zero_point_constant)
        return nullptr;

    switch (type) {
    case ov::element::f32: {
        return compress_quantized_weights_internal(low_precision_type,
                                                   weights->get_data_ptr<float>(),
                                                   weights_shape,
                                                   input_low->get_data_ptr<float>(),
                                                   input_low->get_shape(),
                                                   input_high->get_data_ptr<float>(),
                                                   input_low->get_shape(),
                                                   output_low->get_data_ptr<float>(),
                                                   output_low->get_shape(),
                                                   output_high->get_data_ptr<float>(),
                                                   output_low->get_shape(),
                                                   zero_point_constant->get_data_ptr<float>(),
                                                   zero_point_constant->get_shape(),
                                                   fq->get_levels(),
                                                   can_fuse_zero_point);
    }
    case ov::element::f16: {
        return compress_quantized_weights_internal(low_precision_type,
                                                   weights->get_data_ptr<ov::float16>(),
                                                   weights_shape,
                                                   input_low->get_data_ptr<ov::float16>(),
                                                   input_low->get_shape(),
                                                   input_high->get_data_ptr<ov::float16>(),
                                                   input_low->get_shape(),
                                                   output_low->get_data_ptr<ov::float16>(),
                                                   output_low->get_shape(),
                                                   output_high->get_data_ptr<ov::float16>(),
                                                   output_low->get_shape(),
                                                   zero_point_constant->get_data_ptr<ov::float16>(),
                                                   zero_point_constant->get_shape(),
                                                   fq->get_levels(),
                                                   can_fuse_zero_point);
    }
    case ov::element::bf16: {
        return compress_quantized_weights_internal(low_precision_type,
                                                   weights->get_data_ptr<ov::bfloat16>(),
                                                   weights_shape,
                                                   input_low->get_data_ptr<ov::bfloat16>(),
                                                   input_low->get_shape(),
                                                   input_high->get_data_ptr<ov::bfloat16>(),
                                                   input_low->get_shape(),
                                                   output_low->get_data_ptr<ov::bfloat16>(),
                                                   output_low->get_shape(),
                                                   output_high->get_data_ptr<ov::bfloat16>(),
                                                   output_low->get_shape(),
                                                   zero_point_constant->get_data_ptr<ov::bfloat16>(),
                                                   zero_point_constant->get_shape(),
                                                   fq->get_levels(),
                                                   can_fuse_zero_point);
    }
    default:
        return nullptr;
    }
    return nullptr;
}

template <typename T>
static std::shared_ptr<ov::op::v0::Constant> compress_quantized_weights_internal(
    const ov::element::Type& low_precision_type,
    const T* weights,
    const ov::Shape& weights_shape,
    const T* input_low,
    const ov::Shape& input_low_shape,
    const T* input_high,
    const ov::Shape& input_high_shape,
    const T* zero_point,
    const ov::Shape& zero_point_shape,
    size_t levels,
    bool zero_point_is_zero,
    bool& can_fuse_zero_point) {
    using namespace ov::reference::fake_quantize_details;

    ov::element::Type tensor_el_type;
    if (low_precision_type == ov::element::i8 || low_precision_type == ov::element::i4) {
        tensor_el_type = ov::element::i8;
    } else if (low_precision_type == ov::element::u8 || low_precision_type == ov::element::u4) {
        tensor_el_type = ov::element::u8;
    }

    ov::Tensor compressed_weights_tensor(tensor_el_type, weights_shape);
    ov::Tensor compressed_weights_with_fused_zero_point_tensor;
    if (!zero_point_is_zero) {
        compressed_weights_with_fused_zero_point_tensor = ov::Tensor(tensor_el_type, weights_shape);
    }

    if (tensor_el_type == ov::element::i8) {
        int8_t* compressed_weights = compressed_weights_tensor.data<int8_t>();
        int8_t* compressed_weights_with_fused_zero_point = nullptr;

        if (!zero_point_is_zero) {
            compressed_weights_with_fused_zero_point = compressed_weights_with_fused_zero_point_tensor.data<int8_t>();
        }
        T levels_minus_one = static_cast<T>(levels - 1);
        T output_low = -static_cast<T>(levels / 2);
        T output_high = levels_minus_one + output_low;
        can_fuse_zero_point = !zero_point_is_zero;
        const auto convert_to_low_precision = low_precision_type == ov::element::i4 ? convert_to_int4 : convert_to_int8;

        auto f = [compressed_weights_with_fused_zero_point,
                  levels_minus_one,
                  output_low,
                  output_high,
                  zero_point_is_zero,
                  convert_to_low_precision,
                  &can_fuse_zero_point](T weights_value, T input_low, T input_high, T zero_point) mutable {
            int8_t compressed_weights_value = convert_to_low_precision(
                quantize(weights_value, input_low, input_high, output_low, output_high, levels_minus_one));
            if (!zero_point_is_zero && can_fuse_zero_point) {
                T weights_minus_zero_point = static_cast<T>(compressed_weights_value) - zero_point;
                int8_t compressed_weights_with_fused_zero_point_value =
                    convert_to_low_precision(weights_minus_zero_point);
                can_fuse_zero_point &=
                    std::fabs(compressed_weights_with_fused_zero_point_value - weights_minus_zero_point) < 1e-4;
                *compressed_weights_with_fused_zero_point++ = compressed_weights_with_fused_zero_point_value;
            }
            return compressed_weights_value;
        };

        numpy_broadcast_4inputs(weights,
                                weights_shape,
                                input_low,
                                input_low_shape,
                                input_high,
                                input_high_shape,
                                zero_point,
                                zero_point_shape,
                                compressed_weights,
                                f);
    } else if (tensor_el_type == ov::element::u8) {
        uint8_t* compressed_weights = compressed_weights_tensor.data<uint8_t>();
        uint8_t* compressed_weights_with_fused_zero_point = nullptr;

        if (!zero_point_is_zero) {
            compressed_weights_with_fused_zero_point = compressed_weights_with_fused_zero_point_tensor.data<uint8_t>();
        }
        T levels_minus_one = static_cast<T>(levels - 1);
        T output_low = 0;
        T output_high = levels_minus_one + output_low;
        can_fuse_zero_point = !zero_point_is_zero;
        const auto convert_to_low_precision =
            low_precision_type == ov::element::u4 ? convert_to_uint4 : convert_to_uint8;

        auto f = [compressed_weights_with_fused_zero_point,
                  levels_minus_one,
                  output_low,
                  output_high,
                  zero_point_is_zero,
                  convert_to_low_precision,
                  &can_fuse_zero_point](T weights_value, T input_low, T input_high, T zero_point) mutable {
            uint8_t compressed_weights_value = convert_to_low_precision(
                quantize(weights_value, input_low, input_high, output_low, output_high, levels_minus_one));
            if (!zero_point_is_zero && can_fuse_zero_point) {
                T weights_minus_zero_point = static_cast<T>(compressed_weights_value) - zero_point;
                uint8_t compressed_weights_with_fused_zero_point_value =
                    convert_to_low_precision(weights_minus_zero_point);
                can_fuse_zero_point &=
                    std::fabs(compressed_weights_with_fused_zero_point_value - weights_minus_zero_point) < 1e-4;
                *compressed_weights_with_fused_zero_point++ = compressed_weights_with_fused_zero_point_value;
            }
            return compressed_weights_value;
        };

        numpy_broadcast_4inputs(weights,
                                weights_shape,
                                input_low,
                                input_low_shape,
                                input_high,
                                input_high_shape,
                                zero_point,
                                zero_point_shape,
                                compressed_weights,
                                f);
    }

    return create_weights_constant(
        can_fuse_zero_point ? compressed_weights_with_fused_zero_point_tensor : compressed_weights_tensor,
        low_precision_type);
}

std::shared_ptr<ov::op::v0::Constant> compress_quantized_weights(
    const std::shared_ptr<ov::op::v0::Constant>& weights,
    const std::shared_ptr<ov::op::v0::Constant>& input_low,
    const std::shared_ptr<ov::op::v0::Constant>& input_high,
    const ov::element::Type& low_precision_type,
    size_t levels,
    bool zero_point_is_zero,
    const ov::Tensor& zero_point_tensor,
    bool& can_fuse_zero_point) {
    const auto& weights_shape = weights->get_shape();
    const auto& type = weights->get_element_type();
    switch (type) {
    case ov::element::f32: {
        return compress_quantized_weights_internal(low_precision_type,
                                                   weights->get_data_ptr<float>(),
                                                   weights_shape,
                                                   input_low->get_data_ptr<float>(),
                                                   input_low->get_shape(),
                                                   input_high->get_data_ptr<float>(),
                                                   input_low->get_shape(),
                                                   zero_point_tensor.data<float>(),
                                                   zero_point_tensor.get_shape(),
                                                   levels,
                                                   zero_point_is_zero,
                                                   can_fuse_zero_point);
    }
    case ov::element::f16: {
        return compress_quantized_weights_internal(low_precision_type,
                                                   weights->get_data_ptr<ov::float16>(),
                                                   weights_shape,
                                                   input_low->get_data_ptr<ov::float16>(),
                                                   input_low->get_shape(),
                                                   input_high->get_data_ptr<ov::float16>(),
                                                   input_low->get_shape(),
                                                   zero_point_tensor.data<ov::float16>(),
                                                   zero_point_tensor.get_shape(),
                                                   levels,
                                                   zero_point_is_zero,
                                                   can_fuse_zero_point);
    }
    case ov::element::bf16: {
        return compress_quantized_weights_internal(low_precision_type,
                                                   weights->get_data_ptr<ov::bfloat16>(),
                                                   weights_shape,
                                                   input_low->get_data_ptr<ov::bfloat16>(),
                                                   input_low->get_shape(),
                                                   input_high->get_data_ptr<ov::bfloat16>(),
                                                   input_low->get_shape(),
                                                   zero_point_tensor.data<ov::bfloat16>(),
                                                   zero_point_tensor.get_shape(),
                                                   levels,
                                                   zero_point_is_zero,
                                                   can_fuse_zero_point);
    }
    default:
        return nullptr;
    }
    return nullptr;
}

ov::pass::CompressQuantizeWeights::CompressQuantizeWeights() {
    add_matcher<CompressWeightsWithFakeQuantize>();
    add_matcher<CompressWeightsWithFakeConvert>();
}
