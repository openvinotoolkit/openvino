// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preprocess_steps_impl.hpp"

#include "color_utils.hpp"
#include "layout_utils.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/nv12_to_bgr.hpp"
#include "openvino/op/nv12_to_rgb.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/util/common_util.hpp"
#include "transformations/rt_info/preprocessing_attribute.hpp"

namespace ov {
namespace preprocess {

static Shape construct_mean_scale_shape(const Output<Node>& node,
                                        size_t values_size,
                                        const PreprocessingContext& context) {
    auto node_shape = node.get_partial_shape();
    auto node_rank = node_shape.rank();
    auto channels_index = get_and_check_channels_idx(context.layout(), node_shape);
    std::vector<std::size_t> v(node_rank.get_length(), 1);

    OPENVINO_ASSERT(node_shape[channels_index].is_dynamic() || node_shape[channels_index] == values_size,
                    "Number of channels and mean/values size mismatch: Channels = ",
                    node_shape[channels_index].get_length(),
                    ", mean/scale = ",
                    values_size,
                    ", shape = ",
                    node_shape,
                    ", layout = ",
                    context.layout().to_string());
    v[channels_index] = values_size;
    return {v};
}

template <typename T>
static std::string vector_to_string(const std::vector<T>& values) {
    if (values.empty()) {
        return {};
    }
    std::stringstream s;
    s << "(" << values[0];
    for (size_t i = 1; i < values.size(); i++) {
        s << "," << values[i];
    }
    s << ")";
    return s.str();
}
namespace {
std::shared_ptr<ov::Node> grey_from_yuv_single_plane(const std::vector<Output<Node>>& nodes) {
    using namespace ov::opset8;
    const auto axis = Constant::create(element::i32, {1}, {1});
    const auto yuv_shape_of = std::make_shared<ShapeOf>(nodes[0]);
    const auto get_height = std::make_shared<Gather>(yuv_shape_of, axis, Constant::create(element::i32, {}, {0}));

    const auto start = Constant::create(element::i32, {1}, {0});
    // slice stop is input height * (2/3)
    auto mul_height =
        std::make_shared<Multiply>(get_height, Constant::create(get_height->get_element_type(), {1}, {2}));
    auto stop = std::make_shared<Divide>(mul_height, Constant::create(get_height->get_element_type(), {1}, {3}));
    const auto step = Constant::create(element::i32, {1}, {1});
    //
    return std::make_shared<ov::op::v8::Slice>(nodes[0], start, stop, step, axis);
}
}  // namespace

void PreStepsList::add_scale_impl(const std::vector<float>& values) {
    m_actions.emplace_back(
        [values](const std::vector<Output<Node>>& nodes,
                 const std::shared_ptr<ov::Model>& function,
                 PreprocessingContext& context) -> std::tuple<std::vector<Output<Node>>, bool> {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't apply scale preprocessing for empty input.");
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't apply scale preprocessing for multi-plane input. Suggesting to convert current "
                            "image to RGB/BGR color format using 'convert_color'");
            Shape shape;
            if (values.size() == 1) {
                shape = Shape{1};
            } else {
                shape = construct_mean_scale_shape(nodes[0], values.size(), context);
            }
            auto element_type = nodes[0].get_element_type();
            OPENVINO_ASSERT(element_type.is_real(),
                            "Scale preprocessing can be applied to 'float' inputs. Consider using of "
                            "'convert_element_type' before scaling. Current type is: ",
                            element_type);

            auto constant = op::v0::Constant::create(element_type, shape, values);

            auto new_op = std::make_shared<op::v1::Divide>(nodes[0], constant);
            set_is_preprocessing_node(new_op);
            return std::make_tuple(std::vector<Output<Node>>{new_op}, false);
        },
        "scale " + vector_to_string(values));
}

void PreStepsList::add_mean_impl(const std::vector<float>& values) {
    m_actions.emplace_back(
        [values](const std::vector<Output<Node>>& nodes,
                 const std::shared_ptr<ov::Model>& function,
                 PreprocessingContext& context) {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't apply mean preprocessing for empty input.");
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't apply scale preprocessing for multi-plane input. Suggesting to convert current "
                            "image to RGB/BGR color format using 'convert_color'");
            Shape shape;
            if (values.size() == 1) {
                shape = Shape{1};
            } else {
                shape = construct_mean_scale_shape(nodes[0], values.size(), context);
            }
            auto element_type = nodes[0].get_element_type();
            OPENVINO_ASSERT(
                element_type.is_real(),
                "Mean preprocessing can be applied to 'float' inputs. Consider using of 'convert_element_type' "
                "before scaling. Current type is: ",
                element_type);

            auto constant = op::v0::Constant::create(element_type, shape, values);

            auto new_op = std::make_shared<op::v1::Subtract>(nodes[0], constant);
            set_is_preprocessing_node(new_op);
            return std::make_tuple(std::vector<Output<Node>>{new_op}, false);
        },
        "mean " + vector_to_string(values));
}

void PreStepsList::add_pad_impl(const std::vector<int>& pads_begin,
                                const std::vector<int>& pads_end,
                                const std::vector<float>& pad_values,
                                PaddingMode mode) {
    std::string name;
    name = "pad(begin " + vector_to_string(pads_begin) + ", end " + vector_to_string(pads_end);
    switch (mode) {
    case PaddingMode::CONSTANT:
        name += ", with " + vector_to_string(pad_values) + ")";
        break;
    case PaddingMode::EDGE:
        name += ", copied from edge)";
        break;
    case PaddingMode::REFLECT:
        name += ", reflected from tensor)";
        break;
    case PaddingMode::SYMMETRIC:
        name += ", symmetrically added from tensor)";
        break;
    }

    m_actions.emplace_back(
        [pads_begin, pads_end, pad_values, mode](const std::vector<Output<Node>>& nodes,
                                                 const std::shared_ptr<Model>& function,
                                                 PreprocessingContext& ctxt) {
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't pad multi-plane input. Suggesting to convert current image to "
                            "RGB/BGR color format using 'PreProcessSteps::convert_color'");

            const auto& node = nodes[0];
            auto element_type = nodes[0].get_element_type();
            OPENVINO_ASSERT(element_type.is_real(),
                            "Pad preprocessing can be applied to 'float' inputs. Consider using of "
                            "'convert_element_type' before padding. Current type is: ",
                            element_type);

            auto pad_value = opset8::Constant::create(node.get_element_type(), Shape{}, pad_values);

            auto npads_begin = opset8::Constant::create(element::i64, Shape{pads_begin.size()}, pads_begin);
            auto npads_end = opset8::Constant::create(element::i64, Shape{pads_end.size()}, pads_end);
            auto npad_value = opset8::Constant::create(element_type, Shape{}, pad_values);

            auto pad = std::make_shared<opset8::Pad>(node, npads_begin, npads_end, npad_value, mode);
            return std::make_tuple(std::vector<Output<Node>>{pad}, true);
        },
        name);
}

void PreStepsList::add_convert_impl(const element::Type& type) {
    m_actions.emplace_back(
        [type](const std::vector<Output<Node>>& nodes,
               const std::shared_ptr<Model>& function,
               PreprocessingContext& ctxt) {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't set element type for empty input.");
            std::vector<Output<Node>> res;
            element::Type t = type;
            if (t == element::Type{}) {
                t = ctxt.target_element_type();
            }
            for (const auto& node : nodes) {
                OPENVINO_ASSERT(node.get_element_type().is_static(),
                                "Can't insert 'convert_element_type' for dynamic source tensor type.");
                if (t != node.get_element_type()) {
                    auto convert = std::make_shared<op::v0::Convert>(node, t);
                    res.emplace_back(convert);
                } else {
                    res.emplace_back(node);
                }
            }
            // return false to avoid excess function revalidations as conversion of types
            // doesn't require shape or type propagation.
            return std::make_tuple(res, false);
        },
        "convert type (" + type.to_string() + ")");
}

void PreStepsList::add_resize_impl(ResizeAlgorithm alg, int dst_height, int dst_width) {
    using InterpolateMode = op::util::InterpolateBase::InterpolateMode;
    std::string name;
    if (dst_width > 0 && dst_height > 0) {
        name = "resize to (" + std::to_string(dst_height) + ", " + std::to_string(dst_width) + ")";
    } else {
        name = "resize to model width/height";
    }
    m_actions.emplace_back(
        [alg, dst_width, dst_height](const std::vector<Output<Node>>& nodes,
                                     const std::shared_ptr<Model>& function,
                                     PreprocessingContext& ctxt) {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't add resize for empty input.");
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't resize multi-plane input. Suggesting to convert current image to "
                            "RGB/BGR color format using 'PreProcessSteps::convert_color'");
            const auto to_mode = [](const ResizeAlgorithm alg) -> InterpolateMode {
                switch (alg) {
                case ResizeAlgorithm::RESIZE_NEAREST:
                    return InterpolateMode::NEAREST;
                case ResizeAlgorithm::RESIZE_CUBIC:
                    return InterpolateMode::CUBIC;
                case ResizeAlgorithm::RESIZE_BILINEAR_PILLOW:
                    return InterpolateMode::BILINEAR_PILLOW;
                case ResizeAlgorithm::RESIZE_BICUBIC_PILLOW:
                    return InterpolateMode::BICUBIC_PILLOW;
                case ResizeAlgorithm::RESIZE_LINEAR:
                default:
                    return InterpolateMode::LINEAR;
                }
            };
            const auto& layout = ctxt.layout();
            OPENVINO_ASSERT(ov::layout::has_height(layout) && ov::layout::has_width(layout),
                            "Can't add resize for layout without W/H specified. Use 'set_layout' API to define layout "
                            "of image data, like `NCHW`");
            const auto& node = nodes.front();
            OPENVINO_ASSERT(node.get_partial_shape().rank().is_static(),
                            "Resize operation is not supported for fully dynamic shape");

            const auto height_idx = static_cast<int64_t>(get_and_check_height_idx(layout, node.get_partial_shape()));
            const auto width_idx = static_cast<int64_t>(get_and_check_width_idx(layout, node.get_partial_shape()));
            if (dst_height < 0 || dst_width < 0) {
                OPENVINO_ASSERT(ctxt.model_shape().rank().is_static(),
                                "Resize is not fully specified while target model shape is dynamic");
            }
            const int new_image_width = dst_width < 0 ? static_cast<int>(ctxt.get_model_width_for_resize()) : dst_width;
            const int new_image_height =
                dst_height < 0 ? static_cast<int>(ctxt.get_model_height_for_resize()) : dst_height;

            const auto target_spatial_shape =
                op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {new_image_height, new_image_width});
            // In future consider replacing this to set of new OV operations like `getDimByName(node, "H")`
            // This is to allow specifying layout on 'evaluation' stage
            const auto axes = op::v0::Constant::create<int64_t>(element::i64, Shape{2}, {height_idx, width_idx});

            op::util::InterpolateBase::InterpolateAttrs attrs(to_mode(alg),
                                                              op::util::InterpolateBase::ShapeCalcMode::SIZES,
                                                              {0, 0},
                                                              {0, 0});

            const auto interp = std::make_shared<op::v11::Interpolate>(node, target_spatial_shape, axes, attrs);
            return std::make_tuple(OutputVector{interp}, true);
        },
        name);
}

void PreStepsList::add_crop_impl(const std::vector<int>& begin, const std::vector<int>& end) {
    std::stringstream name_str;
    name_str << "Crop (" << ov::util::vector_to_string(begin) << "," << ov::util::vector_to_string(end) << ")";
    OPENVINO_ASSERT(begin.size() == end.size(),
                    name_str.str(),
                    " begin/end coordinates must have the same size. Begin size=",
                    begin.size(),
                    ", end size=",
                    end.size());
    m_actions.emplace_back(
        [begin, end](const std::vector<Output<Node>>& nodes,
                     const std::shared_ptr<Model>& function,
                     PreprocessingContext& ctxt) {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't add resize for empty input.");
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't crop multi-plane input. Suggesting to convert current image to "
                            "RGB/BGR color format using 'PreProcessSteps::convert_color'");
            auto node = nodes.front();
            auto start = opset8::Constant::create(element::i32, {begin.size()}, begin);
            auto stop = opset8::Constant::create(element::i32, {end.size()}, end);
            auto step = opset8::Constant::create(element::i32, {begin.size()}, std::vector<int32_t>(begin.size(), 1));
            auto slice = std::make_shared<opset8::Slice>(node, start, stop, step);
            return std::make_tuple(std::vector<Output<Node>>{slice}, true);
        },
        name_str.str());
}

Layout PreStepsList::propagate_layout(const Layout& tensor_layout) const {
    auto res = m_last_explicit_layout_set ? m_last_explicit_layout : tensor_layout;
    for (const auto& convert : m_forward_layout_converts) {
        res = layout::utils::apply_permutation(res, convert);
    }
    return res;
}

void PreStepsList::add_convert_layout_impl(const Layout& layout) {
    m_forward_layout_converts.clear();
    m_last_explicit_layout = layout;
    m_last_explicit_layout_set = true;
    m_actions.emplace_back(
        [layout](const std::vector<Output<Node>>& nodes,
                 const std::shared_ptr<Model>& function,
                 PreprocessingContext& context) {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't convert layout for empty input.");
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't convert layout for multi-plane input. Suggesting to convert current image to "
                            "RGB/BGR color format using 'convert_color'");
            Layout dst_layout = layout.empty() ? context.target_layout() : layout;
            auto node = nodes[0];
            auto shape = node.get_partial_shape();
            size_t add_cnt;
            Layout unsqueeze_layout;
            std::tie(shape, unsqueeze_layout, add_cnt) =
                layout::utils::find_unsqueeze(context.layout(), shape, dst_layout);
            if (add_cnt) {
                std::vector<size_t> dims;
                dims.push_back(add_cnt);
                Shape const_shape(dims);
                std::vector<int64_t> vals(add_cnt);
                for (size_t i = 0; i < add_cnt; i++) {
                    vals[i] = i;
                }
                auto axes = op::v0::Constant::create<int64_t>(element::i64, const_shape, vals);
                // Add unsqueeze on top
                node = std::make_shared<opset8::Unsqueeze>(node, axes);
            }
            auto permutation = layout::utils::find_permutation(unsqueeze_layout, shape, dst_layout);
            if (permutation.empty()) {
                // No transpose is needed, just update layout
                if (!layout.empty()) {
                    context.layout() = layout;
                }
                return std::make_tuple(nodes, false);
            }
            auto perm_constant =
                op::v0::Constant::create<int64_t>(element::i64, Shape{permutation.size()}, permutation);
            auto transpose = std::make_shared<op::v1::Transpose>(node, perm_constant);
            context.layout() = std::move(dst_layout);  // Update context's current layout
            // return false to avoid excess function revalidations as layout conversion
            // doesn't require shape or type propagation.
            return std::make_tuple(std::vector<Output<Node>>{transpose}, false);
        },
        "convert layout " + layout.to_string());
}

void PreStepsList::add_convert_layout_impl(const std::vector<uint64_t>& dims) {
    if (dims.empty()) {
        return;
    }
    m_layout_converts.emplace_front(dims);
    m_forward_layout_converts.emplace_back(dims);
    m_actions.emplace_back(
        [dims](const std::vector<Output<Node>>& nodes,
               const std::shared_ptr<Model>& function,
               PreprocessingContext& context) {
            OPENVINO_ASSERT(!nodes.empty(), "Internal error: Can't convert layout for empty input.");
            OPENVINO_ASSERT(nodes.size() == 1,
                            "Can't convert layout for multi-plane input. Suggesting to convert current image to "
                            "RGB/BGR color format using 'convert_color'");
            auto new_layout = layout::utils::apply_permutation(context.layout(), dims);
            auto perm_constant = op::v0::Constant::create<uint64_t>(element::u64, Shape{dims.size()}, dims);
            auto transpose = std::make_shared<op::v1::Transpose>(nodes[0], perm_constant);
            context.layout() = std::move(new_layout);  // Update context's current layout
            // return false to avoid excess function revalidations as layout conversion
            // doesn't require shape or type propagation.
            return std::make_tuple(std::vector<Output<Node>>{transpose}, false);
        },
        "convert layout " + vector_to_string(dims));
}

std::tuple<PartialShape, Layout> PreStepsList::calculate_param_shape(const PartialShape& model_shape,
                                                                     const Layout& model_layout) const {
    if (model_shape.rank().is_dynamic()) {
        return std::tuple<PartialShape, Layout>{model_shape, model_layout};
    }
    Layout res_layout = model_layout;
    std::vector<Dimension> old_dims(model_shape.rank().get_length());
    std::vector<Dimension> dims(model_shape.rank().get_length());
    for (size_t i = 0; i < static_cast<size_t>(model_shape.rank().get_length()); i++) {
        dims[i] = model_shape[i];
    }
    for (const auto& convert : m_layout_converts) {
        old_dims = dims;
        dims = std::vector<Dimension>(model_shape.rank().get_length());
        auto back_convert = convert;
        for (size_t i = 0; i < convert.size(); i++) {
            OPENVINO_ASSERT(convert[i] < dims.size(), "Convert dimension ", convert[i], " is out of bounds.");
            dims[convert[i]] = old_dims[i];
            back_convert[convert[i]] = i;
        }
        res_layout = layout::utils::apply_permutation(res_layout, back_convert);
    }
    return std::tuple<PartialShape, Layout>{dims, res_layout};
}

void PreStepsList::add_convert_color_impl(const ColorFormat& dst_format) {
    m_actions.emplace_back(
        [dst_format](const std::vector<Output<Node>>& nodes,
                     const std::shared_ptr<Model>& function,
                     PreprocessingContext& context) {
            if (context.color_format() == dst_format) {
                return std::make_tuple(nodes, false);
            }
            if (context.color_format() == ColorFormat::NV12_SINGLE_PLANE) {
                OPENVINO_ASSERT(nodes.size() == 1,
                                "Internal error: single plane NV12 image can't have multiple inputs");
                std::shared_ptr<Node> convert;
                switch (dst_format) {
                case ColorFormat::RGB:
                    convert = std::make_shared<op::v8::NV12toRGB>(nodes[0]);
                    break;
                case ColorFormat::BGR:
                    convert = std::make_shared<op::v8::NV12toBGR>(nodes[0]);
                    break;
                case ColorFormat::GRAY:
                    convert = grey_from_yuv_single_plane(nodes);
                    break;
                default:
                    OPENVINO_ASSERT(false,
                                    "Unsupported conversion from NV12 to '",
                                    color_format_name(dst_format),
                                    "' format:");
                }
                context.color_format() = dst_format;
                return std::make_tuple(std::vector<Output<Node>>{convert}, true);
            } else if (context.color_format() == ColorFormat::NV12_TWO_PLANES) {
                OPENVINO_ASSERT(nodes.size() == 2, "Internal error: two-plane NV12 image must have exactly two inputs");
                std::shared_ptr<Node> convert;
                switch (dst_format) {
                case ColorFormat::RGB:
                    convert = std::make_shared<op::v8::NV12toRGB>(nodes[0], nodes[1]);
                    break;
                case ColorFormat::BGR:
                    convert = std::make_shared<op::v8::NV12toBGR>(nodes[0], nodes[1]);
                    break;
                case ColorFormat::GRAY:
                    convert = nodes[0].get_node_shared_ptr();
                    break;
                default:
                    OPENVINO_ASSERT(false,
                                    "Unsupported conversion from NV12 to '",
                                    color_format_name(dst_format),
                                    "' format:");
                }
                context.color_format() = dst_format;
                return std::make_tuple(std::vector<Output<Node>>{convert}, true);
            } else if (context.color_format() == ColorFormat::I420_SINGLE_PLANE) {
                OPENVINO_ASSERT(nodes.size() == 1,
                                "Internal error: single plane I420 image can't have multiple inputs");
                std::shared_ptr<Node> convert;
                switch (dst_format) {
                case ColorFormat::RGB:
                    convert = std::make_shared<op::v8::I420toRGB>(nodes[0]);
                    break;
                case ColorFormat::BGR:
                    convert = std::make_shared<op::v8::I420toBGR>(nodes[0]);
                    break;
                case ColorFormat::GRAY:
                    convert = grey_from_yuv_single_plane(nodes);
                    break;
                default:
                    OPENVINO_ASSERT(false,
                                    "Unsupported conversion from I420 to '",
                                    color_format_name(dst_format),
                                    "' format:");
                }
                context.color_format() = dst_format;
                return std::make_tuple(std::vector<Output<Node>>{convert}, true);
            } else if (context.color_format() == ColorFormat::I420_THREE_PLANES) {
                OPENVINO_ASSERT(nodes.size() == 3,
                                "Internal error: three-plane I420 image must have exactly three inputs");
                std::shared_ptr<Node> convert;
                switch (dst_format) {
                case ColorFormat::RGB:
                    convert = std::make_shared<op::v8::I420toRGB>(nodes[0], nodes[1], nodes[2]);
                    break;
                case ColorFormat::BGR:
                    convert = std::make_shared<op::v8::I420toBGR>(nodes[0], nodes[1], nodes[2]);
                    break;
                case ColorFormat::GRAY:
                    convert = nodes[0].get_node_shared_ptr();
                    break;
                default:
                    OPENVINO_ASSERT(false,
                                    "Unsupported conversion from I420 to '",
                                    color_format_name(dst_format),
                                    "' format:");
                }
                context.color_format() = dst_format;
                return std::make_tuple(std::vector<Output<Node>>{convert}, true);
            }
            if ((context.color_format() == ColorFormat::RGB || context.color_format() == ColorFormat::BGR) &&
                (dst_format == ColorFormat::RGB || dst_format == ColorFormat::BGR)) {
                auto res = reverse_channels(nodes, function, context);
                context.color_format() = dst_format;
                return res;
            }
            if ((context.color_format() == ColorFormat::RGB || context.color_format() == ColorFormat::BGR) &&
                (dst_format == ColorFormat::GRAY)) {
                auto node = nodes[0];
                auto elem_type = node.get_element_type();
                auto shape = node.get_partial_shape();
                OPENVINO_ASSERT(shape.size() == 4,
                                "Input shape size should be equal to 4, actual size: ",
                                shape.size());
                auto channels_idx = get_and_check_channels_idx(context.layout(), shape);
                OPENVINO_ASSERT(shape[channels_idx] == 3,
                                "Channels dimesion should be equal to 3, actual value: ",
                                shape[channels_idx]);

                auto is_transposed = false, is_converted = false;
                if (channels_idx + 1 == shape.size()) {
                    // Transpose N...C  to NC...
                    auto permutation = layout::utils::find_permutation(context.layout(), shape, ov::Layout{"NC..."});
                    auto perm_constant =
                        op::v0::Constant::create<int64_t>(element::i64, Shape{permutation.size()}, permutation);
                    node = std::make_shared<op::v1::Transpose>(node, perm_constant);
                    is_transposed = true;
                }
                if (elem_type.is_integral_number()) {
                    // Compute in floats due weights are floats
                    node = std::make_shared<op::v0::Convert>(node, element::f32);
                    is_converted = true;
                }

                // RGB coefficients were used from https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html
                auto weights_data = context.color_format() == ColorFormat::RGB
                                        ? std::vector<float>{0.299f, 0.587f, 0.114f}
                                        : std::vector<float>{0.114f, 0.587f, 0.299f};
                auto weights_shape = ov::Shape(shape.size(), 1);
                weights_shape[1] = 3;  // Set kernel layout to [1, 3, 1, ...]
                auto weights_node = std::make_shared<ov::op::v0::Constant>(element::f32, weights_shape, weights_data);
                node = std::make_shared<ov::op::v1::Convolution>(node,
                                                                 weights_node,
                                                                 ov::Strides(weights_shape.size() - 2, 1),
                                                                 ov::CoordinateDiff(weights_shape.size() - 2, 0),
                                                                 ov::CoordinateDiff(weights_shape.size() - 2, 0),
                                                                 ov::Strides(weights_shape.size() - 2, 1));

                if (is_converted) {
                    // Roundp values according to OpenCV rule before converting to integral values
                    auto round_val =
                        std::make_shared<ov::op::v5::Round>(node, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
                    node = std::make_shared<op::v0::Convert>(round_val, elem_type);
                }
                if (is_transposed) {
                    // Return NC... to N...C
                    auto permutation = layout::utils::find_permutation(ov::Layout{"NC..."}, shape, context.layout());
                    auto perm_constant =
                        op::v0::Constant::create<int64_t>(element::i64, Shape{permutation.size()}, permutation);
                    node = std::make_shared<op::v1::Transpose>(node, perm_constant);
                }
                context.color_format() = dst_format;
                return std::make_tuple(std::vector<Output<Node>>{std::move(node)}, true);
            }
            if (context.color_format() == ColorFormat::RGBX) {
                if (dst_format == ColorFormat::RGB) {
                    auto res = cut_last_channel(nodes, function, context);
                    context.color_format() = dst_format;
                    return res;
                } else if (dst_format == ColorFormat::BGR) {
                    auto cut = cut_last_channel(nodes, function, context);
                    auto reverse = reverse_channels(std::get<0>(cut), function, context);
                    bool updated = std::get<1>(cut) || std::get<1>(reverse);
                    context.color_format() = dst_format;
                    return std::make_tuple(std::get<0>(reverse), updated);
                }
            }
            if (context.color_format() == ColorFormat::BGRX) {
                if (dst_format == ColorFormat::BGR) {
                    auto res = cut_last_channel(nodes, function, context);
                    context.color_format() = dst_format;
                    return res;
                } else if (dst_format == ColorFormat::RGB) {
                    auto cut = cut_last_channel(nodes, function, context);
                    auto reverse = reverse_channels(std::get<0>(cut), function, context);
                    bool updated = std::get<1>(cut) || std::get<1>(reverse);
                    context.color_format() = dst_format;
                    return std::make_tuple(std::get<0>(reverse), updated);
                }
            }
            OPENVINO_ASSERT(false,
                            "Source color format '",
                            color_format_name(context.color_format()),
                            "' is not convertible to '",
                            color_format_name(dst_format),
                            "'");
        },
        "convert color (" + color_format_name(dst_format) + ")");
}

void PreStepsList::add_reverse_channels() {
    m_actions.emplace_back(
        [](const std::vector<Output<Node>>& nodes,
           const std::shared_ptr<Model>& function,
           PreprocessingContext& context) {
            auto resp = reverse_channels(nodes, function, context);
            auto outputs = std::get<0>(resp);
            OPENVINO_ASSERT(outputs.size() == 1,
                            "Internal error: reverse_channels returned unexpected number of outputs");
            set_is_preprocessing_node(outputs.at(0).get_node_shared_ptr());
            return resp;
        },
        "reverse channels");
}

std::tuple<std::vector<Output<Node>>, bool> PreStepsList::reverse_channels(const std::vector<Output<Node>>& nodes,
                                                                           const std::shared_ptr<Model>& function,
                                                                           PreprocessingContext& context) {
    OPENVINO_ASSERT(nodes.size() == 1, "Internal error: can't reverse channels for multi-plane inputs");
    OPENVINO_ASSERT(ov::layout::has_channels(context.layout()),
                    "Layout ",
                    context.layout().to_string(),
                    " doesn't have `channels` dimension");
    auto shape = nodes[0].get_partial_shape();
    if (shape.rank().is_static()) {
        // This block of code is to preserve output shape if it contains dynamic dimensions
        // Otherwise, dynamic version will transform shape {?,3,?,?} to {?,?,?,?} which is still ok but not desired
        auto channels_idx = get_and_check_channels_idx(context.layout(), shape);
        if (shape[channels_idx].is_static()) {
            auto channels_count = shape[channels_idx].get_length();
            // Add range from constants
            auto range_from = op::v0::Constant::create(element::i64, {}, {channels_count - 1});
            auto range_to = op::v0::Constant::create(element::i64, {}, {-1});
            auto range_step = op::v0::Constant::create(element::i64, {}, {-1});
            auto range = std::make_shared<op::v4::Range>(range_from, range_to, range_step, element::i32);

            auto constant_axis = op::v0::Constant::create(element::i32, {1}, {channels_idx});
            auto convert = std::make_shared<op::v8::Gather>(nodes[0], range, constant_axis);
            return std::make_tuple(std::vector<Output<Node>>{convert}, false);
        }
    }

    auto channels_idx = ov::layout::channels_idx(context.layout());
    // Get shape of user's input tensor (e.g. Tensor[1, 3, 224, 224] -> {1, 3, 224, 224})
    auto shape_of = std::make_shared<ov::op::v0::ShapeOf>(nodes[0]);  // E.g. {1, 3, 224, 224}

    auto constant_chan_idx = op::v0::Constant::create(element::i32, {}, {channels_idx});  // E.g. 1
    auto constant_chan_axis = op::v0::Constant::create(element::i32, {}, {0});
    // Gather will return scalar with number of channels (e.g. 3)
    auto gather_channels_num = std::make_shared<op::v8::Gather>(shape_of, constant_chan_idx, constant_chan_axis);

    // Create Range from channels_num-1 to 0 (e.g. {2, 1, 0})
    auto const_minus1 = op::v0::Constant::create(element::i64, {}, {-1});
    auto channels_num_minus1 = std::make_shared<op::v1::Add>(gather_channels_num, const_minus1);  // E.g. 3-1=2
    // Add range
    auto range_to = op::v0::Constant::create(element::i64, {}, {-1});
    auto range_step = op::v0::Constant::create(element::i64, {}, {-1});
    // E.g. {2, 1, 0}
    auto range = std::make_shared<op::v4::Range>(channels_num_minus1, range_to, range_step, element::i32);

    // Gather slices in reverse order (indexes are specified by 'range' operation)
    auto constant_axis = op::v0::Constant::create(element::i32, {1}, {channels_idx});
    auto gather = std::make_shared<op::v8::Gather>(nodes[0], range, constant_axis);
    return std::make_tuple(std::vector<Output<Node>>{gather}, false);
}

std::tuple<std::vector<Output<Node>>, bool> PreStepsList::cut_last_channel(const std::vector<Output<Node>>& nodes,
                                                                           const std::shared_ptr<Model>& function,
                                                                           PreprocessingContext& context) {
    OPENVINO_ASSERT(nodes.size() == 1, "Internal error: can't cut X channel for multi-plane inputs");
    OPENVINO_ASSERT(ov::layout::has_channels(context.layout()),
                    "Layout ",
                    context.layout().to_string(),
                    " doesn't have `channels` dimension");
    auto channels_idx = ov::layout::channels_idx(context.layout());

    auto start = opset8::Constant::create(element::i32, {1}, {0});
    auto stop = opset8::Constant::create(element::i32, {1}, {-1});  // Everything except last channel
    auto step = opset8::Constant::create(element::i32, {1}, {1});
    auto axis = opset8::Constant::create(element::i32, {1}, {channels_idx});  // E.g. 3
    auto slice = std::make_shared<ov::op::v8::Slice>(nodes[0], start, stop, step, axis);
    return std::make_tuple(std::vector<Output<Node>>{slice}, false);
}

//------------- Post processing ------
void PostStepsList::add_convert_impl(const element::Type& type) {
    m_actions.emplace_back(
        [type](const Output<Node>& node, PostprocessingContext& ctxt) {
            element::Type t = type;
            if (t == element::Type{}) {
                t = ctxt.target_element_type();
            }
            if (t == node.get_element_type()) {
                return std::make_tuple(node, false);
            }
            OPENVINO_ASSERT(
                !t.is_dynamic() && t != element::undefined,
                "Can't convert to dynamic/unknown element type, consider using of InputTensorInfo::set_element_type");
            auto convert = std::make_shared<op::v0::Convert>(node, t);
            return std::make_tuple(Output<Node>(convert), true);
        },
        "convert type (" + type.to_string() + ")");
}

void PostStepsList::add_convert_layout_impl(const Layout& layout) {
    m_actions.emplace_back(
        [layout](const Output<Node>& node, PostprocessingContext& context) {
            Layout dst_layout = layout.empty() ? context.target_layout() : layout;
            auto permutation = layout::utils::find_permutation(context.layout(), node.get_partial_shape(), dst_layout);
            if (permutation.empty()) {
                // No transpose is needed, just update layout
                if (!layout.empty()) {
                    context.layout() = layout;
                }
                return std::make_tuple(node, false);
            }
            auto perm_constant =
                op::v0::Constant::create<int64_t>(element::i64, Shape{permutation.size()}, permutation);
            auto transpose = std::make_shared<op::v1::Transpose>(node, perm_constant);
            context.layout() = std::move(dst_layout);  // Update context's current layout
            return std::make_tuple(Output<Node>(transpose), true);
        },
        "convert layout " + layout.to_string());
}

void PostStepsList::add_convert_layout_impl(const std::vector<uint64_t>& dims) {
    if (dims.empty()) {
        return;
    }
    m_actions.emplace_back(
        [dims](const Output<Node>& node, PostprocessingContext& context) {
            auto perm_constant = op::v0::Constant::create<uint64_t>(element::u64, Shape{dims.size()}, dims);
            auto new_layout = layout::utils::apply_permutation(context.layout(), dims);
            auto transpose = std::make_shared<op::v1::Transpose>(node, perm_constant);
            auto res = std::make_tuple(Output<Node>(transpose), true);
            context.layout() = std::move(new_layout);  // Update context's current layout
            return res;
        },
        "convert layout " + vector_to_string(dims));
}

void PostStepsList::add_convert_color_impl(const ColorFormat& dst_format) {
    m_actions.emplace_back(
        [dst_format](const Output<Node>& node, PostprocessingContext& context) {
            if (context.color_format() == dst_format) {
                return std::make_tuple(node, false);
            } else if ((context.color_format() == ColorFormat::RGB || context.color_format() == ColorFormat::BGR) &&
                       (dst_format == ColorFormat::RGB || dst_format == ColorFormat::BGR)) {
                auto res = reverse_channels({node}, context);
                context.color_format() = dst_format;
                return res;
            } else {
                OPENVINO_THROW("Source color format '",
                               color_format_name(context.color_format()),
                               "' is not convertible to '",
                               color_format_name(dst_format),
                               "'");
            }
        },
        "convert color (" + color_format_name(dst_format) + ")");
}

std::tuple<Output<Node>, bool> PostStepsList::reverse_channels(const Output<Node>& node,
                                                               PostprocessingContext& context) {
    OPENVINO_ASSERT(ov::layout::has_channels(context.layout()),
                    "Layout ",
                    context.layout().to_string(),
                    " doesn't have `channels` dimension");
    const auto& shape = node.get_partial_shape();
    if (shape.rank().is_static()) {
        // This block of code is to preserve output shape if it contains dynamic dimensions
        // Otherwise, dynamic version will transform shape {?,3,?,?} to {?,?,?,?} which is still ok but not desired
        auto channels_idx = get_and_check_channels_idx(context.layout(), shape);
        if (shape[channels_idx].is_static()) {
            auto channels_count = shape[channels_idx].get_length();
            // Add range from constants
            auto range_from = op::v0::Constant::create(element::i64, {}, {channels_count - 1});
            auto range_to = op::v0::Constant::create(element::i64, {}, {-1});
            auto range_step = op::v0::Constant::create(element::i64, {}, {-1});
            auto range = std::make_shared<op::v4::Range>(range_from, range_to, range_step, element::i32);

            auto constant_axis = op::v0::Constant::create(element::i32, {1}, {channels_idx});
            auto convert = std::make_shared<op::v8::Gather>(node, range, constant_axis);
            return std::make_tuple(convert, false);
        }
    }

    auto channels_idx = ov::layout::channels_idx(context.layout());
    // Get shape of user's input tensor (e.g. Tensor[1, 3, 224, 224] -> {1, 3, 224, 224})
    auto shape_of = std::make_shared<ov::op::v0::ShapeOf>(node);  // E.g. {1, 3, 224, 224}

    auto constant_chan_idx = op::v0::Constant::create(element::i32, {}, {channels_idx});  // E.g. 1
    auto constant_chan_axis = op::v0::Constant::create(element::i32, {}, {0});
    // Gather will return scalar with number of channels (e.g. 3)
    auto gather_channels_num = std::make_shared<op::v8::Gather>(shape_of, constant_chan_idx, constant_chan_axis);

    // Create Range from channels_num-1 to 0 (e.g. {2, 1, 0})
    auto const_minus1 = op::v0::Constant::create(element::i64, {}, {-1});
    auto channels_num_minus1 = std::make_shared<op::v1::Add>(gather_channels_num, const_minus1);  // E.g. 3-1=2
    // Add range
    auto range_to = op::v0::Constant::create(element::i64, {}, {-1});
    auto range_step = op::v0::Constant::create(element::i64, {}, {-1});
    // E.g. {2, 1, 0}
    auto range = std::make_shared<op::v4::Range>(channels_num_minus1, range_to, range_step, element::i32);

    // Gather slices in reverse order (indexes are specified by 'range' operation)
    auto constant_axis = op::v0::Constant::create(element::i32, {1}, {channels_idx});
    auto gather = std::make_shared<op::v8::Gather>(node, range, constant_axis);
    return std::make_tuple(gather, false);
}

}  // namespace preprocess
}  // namespace ov
