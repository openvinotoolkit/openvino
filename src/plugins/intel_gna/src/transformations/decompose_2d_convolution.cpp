// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/decompose_2d_convolution.hpp"

#include <gna/gna_config.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/cc/ngraph/itt.hpp>
#include <transformations/utils/utils.hpp>

#include "backend/gna_limitations.hpp"
#include "layers/gna_convolution_layer.hpp"
#include "utils/transformation_helper.hpp"

namespace ov {
namespace intel_gna {
using namespace target;
namespace pass {
using namespace helper;

struct GraphData {
    std::shared_ptr<ngraph::opset7::Transpose> leading_transpose;
    std::shared_ptr<ngraph::opset7::FakeQuantize> fq_filters;
    std::shared_ptr<ngraph::opset7::Convolution> conv;
    std::shared_ptr<ngraph::opset7::Transpose> trailing_transpose;
    std::shared_ptr<ngraph::opset7::FakeQuantize> fq_conv;
    std::shared_ptr<ngraph::opset7::FakeQuantize> fq_bias;
    std::shared_ptr<ngraph::opset7::MaxPool> max_pool;
    std::shared_ptr<ngraph::op::util::UnaryElementwiseArithmetic> af;
    std::shared_ptr<ngraph::opset7::FakeQuantize> fq_af;
    std::shared_ptr<ngraph::Node> last_op_in_sequence_for_replacement;
    std::shared_ptr<ngraph::Node> bias_const;
    size_t conv_count;
    size_t pool_size_width;
    size_t pool_stride_width;
    // TODO: currently 2D max pool is not supported
    // size_t pool_size_height;
    // size_t pool_stride_height;
};

static bool VerifyAndGetConvData(std::shared_ptr<ngraph::opset7::Convolution> conv, ConvData& conv_data) {
    const auto& input = conv->input_value(0);
    const auto& filters = conv->input_value(1);

    // We support only batch == 1
    if (input.get_shape()[0] != 1) {
        return false;
    }

    size_t filter_height = filters.get_shape()[2];
    size_t filter_width = filters.get_shape()[3];

    if (filter_width > limitations::copyMaxGrouping || filter_height > limitations::copyMaxGrouping) {
        return false;
    }

    GetConvData(conv, conv_data);

    IE_ASSERT(conv_data.output_channel_count == conv->get_output_shape(0)[1]);

    return true;
}

static bool VerifyMaxPool(GraphData& graph_data, std::shared_ptr<ngraph::opset7::MaxPool> max_pool) {
    auto pool_filter = max_pool->get_kernel();
    auto pool_strides = max_pool->get_strides();

    // Check Max Pool padding and limitations
    // Allow only Max Pool 1D (2D is currently not supported by this transformation)
    if ((max_pool->get_auto_pad() != ngraph::op::PadType::VALID &&
         (max_pool->get_auto_pad() != ngraph::op::PadType::EXPLICIT ||
          max_pool->get_pads_begin() != ngraph::Shape({0, 0}) || max_pool->get_pads_end() != ngraph::Shape({0, 0}))) ||
        pool_filter.size() != 2 || pool_strides.size() != 2 || pool_filter[0] > 1 || pool_strides[0] > 1 ||
        pool_filter[0] > limitations::maxPoolMaxWindowSize)
        return false;

    graph_data.pool_size_width = pool_filter[1];
    graph_data.pool_stride_width = pool_strides[1];
    return true;
}

static bool GNA30SupportedConv(const DeviceVersion& compile_target,
                               const InferenceEngine::Precision& gnaPrecision,
                               const GraphData& graph_data,
                               const ConvData& conv_data) {
    const auto cnn2dValidatorPtr = limitations::cnn2d::AbstractValidator::Create(compile_target);
    if (!cnn2dValidatorPtr) {
        return false;
    }
    const auto& cnn2dValidator = *cnn2dValidatorPtr;
    const auto cnnIsValid = cnn2dValidator.ValidateCnn2D(graph_data.conv->get_friendly_name(),
                                                         conv_data.input_height,
                                                         conv_data.input_width,
                                                         conv_data.input_channel_count,
                                                         conv_data.filter_height,
                                                         conv_data.filter_width,
                                                         conv_data.filter_channel_count,
                                                         conv_data.filter_stride_height,
                                                         conv_data.filter_stride_width,
                                                         conv_data.filter_dilation_height,
                                                         conv_data.filter_dilation_width,
                                                         OvGnaTypeIntFromBytes(gnaPrecision.size()),
                                                         false);
    if (!cnnIsValid) {
        return false;
    }
    if (!graph_data.max_pool) {
        return true;
    }
    const auto poolingValid = cnn2dValidator.ValidatePooling2D(graph_data.conv->get_friendly_name(),
                                                               graph_data.max_pool->get_kernel()[0],
                                                               graph_data.max_pool->get_kernel()[1],
                                                               graph_data.max_pool->get_strides()[0],
                                                               graph_data.max_pool->get_strides()[1],
                                                               false);
    return poolingValid;
}

static size_t CalculateConvCount(const ConvData& conv_data) {
    // Check if split of plane due to GNA HW limitations of 768 filter elements is possible
    size_t conv_count = 1;
    size_t total_factorized_conv_channel_count =
        (conv_data.input_channel_count * conv_data.filter_height * conv_data.filter_width);
    while (total_factorized_conv_channel_count / conv_count > limitations::convFilterMaxSize ||
           total_factorized_conv_channel_count % conv_count != 0 || conv_data.filter_channel_count % conv_count != 0)
        conv_count++;

    return conv_count;
}

static bool ShouldDecompose(GraphData& graph_data, const ConvData& conv_data) {
    // Calculate the number of splits required
    graph_data.conv_count = CalculateConvCount(conv_data);

    // Concat (copy) layer limitation allows to split up to a certain limit
    // Currently we are able to split only convolutions without pooling in horizontal dimension
    if (graph_data.conv_count > limitations::copyMaxGrouping ||
        ((graph_data.pool_size_width > 1 || graph_data.pool_stride_width > 1) && graph_data.conv_count > 1))
        return false;

    // GNA supported features or handled otherwise - there is no need to decompose such convolution
    if (graph_data.conv_count == 1 &&
        (((conv_data.input_height == 1 || conv_data.input_width == 1) && conv_data.filter_dilation_width == 1 &&
          conv_data.filter_dilation_height == 1) ||
         gna_convolution_layer::isMappableFrom2DTo1D(conv_data.input_height,
                                                     conv_data.input_width,
                                                     conv_data.input_channel_count,
                                                     conv_data.filter_height,
                                                     conv_data.filter_width,
                                                     conv_data.filter_stride_height,
                                                     conv_data.filter_stride_width)))
        return false;

    return true;
}

static std::vector<std::shared_ptr<ngraph::Node>> Split2DConvFilters(std::shared_ptr<ngraph::opset7::Constant>& filters,
                                                                     const bool& vertical_permute,
                                                                     const bool& horizontal_permute,
                                                                     const size_t& split_channels) {
    if (!horizontal_permute && !vertical_permute && split_channels == 1)
        return {filters};

    std::vector<std::shared_ptr<ngraph::Node>> result;
    ngraph::Shape reshape_shape;
    auto flat_filters = filters->outputs();
    const auto filter_shape = filters->get_output_shape(0);
    IE_ASSERT(filter_shape.size() == 4);

    if (split_channels > 1) {
        const auto axis_node = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        const auto split = std::make_shared<ngraph::opset7::Split>(filters, axis_node, split_channels);
        flat_filters = split->outputs();
    }

    for (size_t split_index = 0; split_index < split_channels; split_index++) {
        ngraph::Output<ngraph::Node>& flat_filter = flat_filters[split_index];
        if (horizontal_permute && !vertical_permute) {
            result.push_back(std::make_shared<ngraph::opset7::Transpose>(
                flat_filter,
                ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, ngraph::Shape{0, 1, 3, 2})));
        } else {
            result.push_back(flat_filter.get_node_shared_ptr());
        }
    }

    if (vertical_permute && horizontal_permute) {
        reshape_shape =
            ngraph::Shape{filter_shape[0], filter_shape[1] * filter_shape[2] * filter_shape[3] / split_channels, 1, 1};
    } else if (vertical_permute && !horizontal_permute) {
        reshape_shape =
            ngraph::Shape{filter_shape[0], filter_shape[1] * filter_shape[2] / split_channels, 1, filter_shape[3]};
    } else if (!vertical_permute && horizontal_permute) {
        reshape_shape =
            ngraph::Shape{filter_shape[0], filter_shape[1] * filter_shape[3] / split_channels, filter_shape[2], 1};
    } else {
        reshape_shape =
            ngraph::Shape{filter_shape[0], filter_shape[1] / split_channels, filter_shape[2], filter_shape[3]};
    }

    for (auto& new_filter : result)
        new_filter = ov::op::util::make_try_fold<ngraph::opset7::Reshape>(
            new_filter,
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, reshape_shape),
            false);

    return result;
}

static ngraph::OutputVector SplitInput(const GraphData& graph_data, ConvData& conv_data) {
    // We need to have proper input shape first
    ngraph::OutputVector split_planes;
    auto padded_input_plane = std::make_shared<ngraph::opset7::Reshape>(
        graph_data.leading_transpose->input_value(0),
        ngraph::opset7::Constant::create(
            ngraph::element::i64,
            ngraph::Shape{2},
            ngraph::Shape{1, shape_size(graph_data.leading_transpose->input_value(0).get_shape())}),
        false);
    copy_runtime_info(graph_data.conv, padded_input_plane);

    if (graph_data.conv_count > 1) {
        // If we have split input plane and convolutions due to GNA limitation - we must sum their results at the end
        conv_data.input_channel_count /= graph_data.conv_count;

        auto reshape_before_transpose = std::make_shared<ngraph::opset7::Reshape>(
            padded_input_plane,
            ngraph::opset7::Constant::create(
                ngraph::element::i64,
                ngraph::Shape{2},
                {shape_size(padded_input_plane->get_shape()) / graph_data.conv_count, graph_data.conv_count}),
            false);

        auto transpose_before_channel_wise_split = std::make_shared<ngraph::opset7::Transpose>(
            reshape_before_transpose,
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 0})->output(0));

        const auto axis_node = ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{}, {0});
        const auto split = std::make_shared<ngraph::opset7::Split>(transpose_before_channel_wise_split,
                                                                   axis_node,
                                                                   graph_data.conv_count);
        split_planes = split->outputs();
    } else {
        split_planes.push_back(padded_input_plane);
    }

    return split_planes;
}

static std::vector<std::shared_ptr<ngraph::Node>> SplitFilters(const GraphData& graph_data, ConvData& conv_data) {
    // If the input plane exceeds GNA limits and we have split into several convolutions, then we need to split filter
    // data as well; we also need to take filter height and potential dilation into account when modifying the filters

    // Take account of fake quantize when getting filter values
    auto filter_values = std::dynamic_pointer_cast<ngraph::opset7::Constant>(
        graph_data.fq_filters == nullptr ? graph_data.conv->input_value(1).get_node_shared_ptr()
                                         : graph_data.fq_filters->input_value(0).get_node_shared_ptr());
    bool vertical_permute = (conv_data.filter_height > 1);
    bool horizontal_permute = (conv_data.filter_dilation_width > 1);
    std::vector<std::shared_ptr<ngraph::Node>> h_1_filters{};

    h_1_filters = Split2DConvFilters(filter_values, vertical_permute, horizontal_permute, graph_data.conv_count);

    for (auto& filter : h_1_filters)
        copy_runtime_info(graph_data.conv, filter);

    return h_1_filters;
}

static void TransformInput(const GraphData& graph_data,
                           const ConvData& conv_data,
                           ngraph::Output<ngraph::Node>& split_input_plane) {
    /*
     *              Padded row - NHWC order
     *                  |
     *        Split in vertical dim (filter height)
     *                / | \
     *                Concat
     *                  |
     *              Transpose
     */

    // First we need to prepare flat (height = 1) slices of input data proper for flattened (height = 1) filters created
    // later on; the input data is overlapping (duplicated)
    ngraph::OutputVector dilated_input_planes;
    for (size_t filter_height = 0; filter_height < conv_data.filter_height; filter_height++) {
        size_t offset;

        if (conv_data.filter_stride_height > 1) {
            // Prepare strided slices of input data
            for (size_t output_height = 0; output_height < conv_data.output_height; output_height++) {
                offset = (filter_height * conv_data.filter_dilation_height +
                          output_height * conv_data.filter_stride_height) *
                         conv_data.input_width * conv_data.input_channel_count;
                auto slice = FlatCrop(split_input_plane, offset, conv_data.input_width * conv_data.input_channel_count);
                copy_runtime_info(graph_data.conv, slice);
                dilated_input_planes.push_back(slice);
            }
        } else {
            offset = filter_height * conv_data.filter_dilation_height * conv_data.input_width *
                     conv_data.input_channel_count;
            auto slice = FlatCrop(split_input_plane,
                                  offset,
                                  conv_data.input_width * conv_data.input_channel_count * conv_data.output_height);
            copy_runtime_info(graph_data.conv, slice);
            dilated_input_planes.push_back(slice);
        }
    }

    // Interleaving dilated input planes
    std::shared_ptr<ngraph::Node> dilated_chunks_concat =
        std::make_shared<ngraph::opset7::Concat>(dilated_input_planes, 0);

    // Additional reshape is required for strided slices of input intended for each filter row
    if (conv_data.filter_stride_height > 1) {
        dilated_chunks_concat = std::make_shared<ngraph::opset7::Reshape>(
            dilated_chunks_concat,
            ngraph::opset7::Constant::create(
                ngraph::element::i64,
                ngraph::Shape{2},
                {conv_data.filter_height,
                 conv_data.input_width * conv_data.input_channel_count * conv_data.output_height}),
            false);
    }

    auto transposed_dilated_chunks = std::make_shared<ngraph::opset7::Transpose>(
        dilated_chunks_concat,
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 0})->output(0));

    // Flattening of interleaved input planes
    auto flattened_dilated_transposed_input = std::make_shared<ngraph::opset7::Reshape>(
        transposed_dilated_chunks,
        ngraph::opset7::Constant::create(ngraph::element::i64,
                                         ngraph::Shape{2},
                                         {(size_t)1,
                                          conv_data.input_width * conv_data.input_channel_count *
                                              conv_data.output_height * conv_data.filter_height}),
        false);

    copy_runtime_info(graph_data.conv,
                      {dilated_chunks_concat, flattened_dilated_transposed_input, transposed_dilated_chunks});
    split_input_plane = flattened_dilated_transposed_input;
}

// Valid 1D (decomposed 2D) convolution wrapped with transposes NHWC => NCHW => conv => NCHW => NHWC
static std::shared_ptr<ngraph::Node> Create1DConv(const GraphData& graph_data,
                                                  const ConvData& conv_data,
                                                  const ngraph::Output<ngraph::Node>& input,
                                                  std::shared_ptr<ngraph::Node> filters,
                                                  const size_t conv_index,
                                                  const size_t h_index) {
    // Transpose NHWC => NCHW
    std::shared_ptr<ngraph::Node> nchw_input = std::make_shared<ngraph::opset7::Transpose>(
        input,
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 3, 1, 2})->output(0));

    // Fake quantize
    filters = InsertFQLayer(graph_data.fq_filters, filters);

    // 1D Convolution & fake quantize
    auto conv = std::make_shared<ngraph::opset7::Convolution>(nchw_input,
                                                              filters,
                                                              ngraph::Strides{1, conv_data.filter_stride_width},
                                                              ngraph::CoordinateDiff{0, 0},
                                                              ngraph::CoordinateDiff{0, 0},
                                                              ngraph::Strides{1, 1},
                                                              ngraph::op::PadType::VALID);
    std::string conv_name =
        graph_data.conv->get_friendly_name() + "_H_" + std::to_string(h_index) + "_CH_" + std::to_string(0);
    conv->set_friendly_name(conv_name);
    std::shared_ptr<ngraph::Node> last_conv_block_op = conv;
    last_conv_block_op = InsertFQLayer(graph_data.fq_conv, last_conv_block_op);

    // Bias & fake quantize
    if (graph_data.bias_const && conv_index == 0) {
        auto bias_size = shape_size(graph_data.bias_const->get_shape());
        auto reshaped_bias_const = ov::op::util::make_try_fold<ngraph::opset7::Reshape>(
            graph_data.bias_const,
            ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, ngraph::Shape{1, bias_size, 1, 1}),
            false);
        last_conv_block_op = std::make_shared<ngraph::opset7::Add>(conv, reshaped_bias_const);
        copy_runtime_info(graph_data.conv, last_conv_block_op);
        last_conv_block_op = InsertFQLayer(graph_data.fq_bias, last_conv_block_op);
    }

    // Max pooling
    if (graph_data.max_pool && (graph_data.pool_size_width > 1 || graph_data.pool_stride_width > 1)) {
        last_conv_block_op = std::make_shared<ngraph::opset7::MaxPool>(last_conv_block_op,
                                                                       ngraph::Strides{1, graph_data.pool_stride_width},
                                                                       ngraph::Shape{0, 0},
                                                                       ngraph::Shape{0, 0},
                                                                       ngraph::Shape{1, graph_data.pool_size_width},
                                                                       graph_data.max_pool->get_rounding_type(),
                                                                       ngraph::op::PadType::VALID);
    }

    // Activation function & fake quantize
    if (graph_data.af && graph_data.conv_count == 1) {
        last_conv_block_op = graph_data.af->clone_with_new_inputs({last_conv_block_op});
        copy_runtime_info(conv, last_conv_block_op);
        last_conv_block_op = InsertFQLayer(graph_data.fq_af, last_conv_block_op);
    }

    // Transpose NCHW => NHWC
    auto nhwc_output = std::make_shared<ngraph::opset7::Transpose>(
        last_conv_block_op,
        ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{4}, {0, 2, 3, 1})->output(0));
    copy_runtime_info(graph_data.conv, {nchw_input, conv, nhwc_output});
    return nhwc_output;
}

static std::shared_ptr<ngraph::Node> CreateDecomposedConv(const GraphData& graph_data,
                                                          ConvData& conv_data,
                                                          ngraph::Output<ngraph::Node>& reduced_input_plane,
                                                          const std::vector<std::shared_ptr<ngraph::Node>>& h_1_filters,
                                                          const size_t conv_index) {
    ngraph::OutputVector result_chunks;
    std::shared_ptr<ngraph::Node> last_op;
    bool horizontal_permute = (conv_data.filter_dilation_width > 1);
    size_t h_1_filter_channel_count = (conv_data.input_channel_count * conv_data.filter_height);

    for (size_t output_height = 0; output_height < conv_data.output_height; output_height++) {
        size_t offset = output_height * conv_data.input_width * h_1_filter_channel_count;
        auto row = (conv_data.output_height == 1)
                       ? reduced_input_plane
                       : FlatCrop(reduced_input_plane, offset, conv_data.input_width * h_1_filter_channel_count);
        /*
         *              Padded row
         *                  |
         *        ??? <Dilation !=1> ???
         *                  |
         *         Split in vertical dim
         *                / | \
         *                Concat
         *                  |
         *               Permute
         *                  |
         *              Transpose (NHWC => NCHW)
         *                  |
         *                1D Conv (Bias | MaxPooling)
         *                  |
         *              Transpose (NCHW => NHWC)
         */
        auto nhwc_conv_y_input = row;

        if (horizontal_permute) {
            // Horizontal split - transform input accordingly
            ngraph::OutputVector dilated_chunks;
            std::shared_ptr<ngraph::Node> dilated_chunks_concat = nhwc_conv_y_input.get_node_shared_ptr();

            // We need to calculate some parameters in case horizontal stride > 1 is used, because if we use the ones
            // available from the original convolution we won't take into account the fact horizontal strides will be
            // supported by the newly created 1D convolution, and not by decomposition
            size_t filter_dilation_width = conv_data.filter_width > 1 ? conv_data.filter_dilation_width : 1;
            size_t output_width = (conv_data.input_width - (filter_dilation_width * (conv_data.filter_width - 1)));

            if (conv_data.filter_width > 1) {
                for (size_t filter_width = 0; filter_width < conv_data.filter_width; filter_width++) {
                    size_t offset = filter_width * conv_data.filter_dilation_width * h_1_filter_channel_count;
                    auto slice = FlatCrop(row, offset, h_1_filter_channel_count * output_width);
                    copy_runtime_info(graph_data.conv, slice);
                    dilated_chunks.push_back(slice);
                }

                dilated_chunks_concat = std::make_shared<ngraph::opset7::Concat>(dilated_chunks, 0);
            }

            auto transposed_dilated_chunks = std::make_shared<ngraph::opset7::Transpose>(
                dilated_chunks_concat,
                ngraph::opset7::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 0})->output(0));

            auto flattened_dilated_conv_input = std::make_shared<ngraph::opset7::Reshape>(
                transposed_dilated_chunks,
                ngraph::opset7::Constant::create(
                    ngraph::element::i64,
                    ngraph::Shape{4},
                    ngraph::Shape{1, 1, output_width, h_1_filter_channel_count * conv_data.filter_width}),
                false);

            copy_runtime_info(
                graph_data.conv,
                ngraph::NodeVector{flattened_dilated_conv_input, transposed_dilated_chunks, dilated_chunks_concat});

            nhwc_conv_y_input = flattened_dilated_conv_input;
        } else {
            // If no horizontal split is done, only reshape is required before decomposed convolution
            nhwc_conv_y_input = std::make_shared<ngraph::opset7::Reshape>(
                nhwc_conv_y_input,
                ngraph::opset7::Constant::create(ngraph::element::i64,
                                                 ngraph::Shape{4},
                                                 ngraph::Shape{1, 1, conv_data.input_width, h_1_filter_channel_count}),
                false);
        }

        // Pointwise convolutions
        // Valid 1D convolution wrapped with transposes NHWC => NCHW => Conv => NCHW => NHWC
        // Activation function can be fused with convolution only if it isn't split
        auto nhwc_y_output =
            Create1DConv(graph_data, conv_data, nhwc_conv_y_input, h_1_filters[conv_index], conv_index, output_height);
        result_chunks.push_back(nhwc_y_output);
        last_op = nhwc_y_output;
    }

    // Horizontal dimemsion greater than 1
    if (result_chunks.size() > 1) {
        // Concat in horizontal dimension
        // In NHWC index of H is 1
        auto concatenated_sub_results = std::make_shared<ngraph::opset7::Concat>(result_chunks, 1);
        copy_runtime_info(graph_data.conv, concatenated_sub_results);
        last_op = concatenated_sub_results;
    }
    return last_op;
}

static void Decompose(const GraphData& graph_data, ConvData& conv_data) {
    std::vector<std::shared_ptr<ngraph::Node>> partial_conv_results;

    // Split input due to GNA filter element count limit
    auto split_planes = SplitInput(graph_data, conv_data);
    // Split filters due to GNA filter element count limit, 2D convolution shape, or dilations
    auto h_1_filters = SplitFilters(graph_data, conv_data);

    // Do transformations in each of the splits created above
    for (size_t conv_index = 0; conv_index < graph_data.conv_count; conv_index++) {
        ngraph::Output<ngraph::Node>& split_input_plane = split_planes[conv_index];

        // Input data needs to be prepared before 2D convolution decomposition
        if (conv_data.filter_height > 1 || conv_data.filter_stride_height > 1) {
            TransformInput(graph_data, conv_data, split_input_plane);
        }

        auto flat_conv = CreateDecomposedConv(graph_data, conv_data, split_input_plane, h_1_filters, conv_index);
        partial_conv_results.push_back(flat_conv);
    }

    std::shared_ptr<ngraph::Node> conv_result = partial_conv_results.front();
    for (size_t i = 1; i < partial_conv_results.size(); i++) {
        auto add_result = std::make_shared<ngraph::opset7::Add>(partial_conv_results[i], conv_result);
        copy_runtime_info(graph_data.conv, add_result);
        conv_result = add_result;
    }

    // TODO: Max Pool 2D case
    // if (graph_data.max_pool && (graph_data.pool_size_height > 1 || graph_data.pool_stride_height > 1)) {
    //}

    // Activation function after trailing Transpose NCHW->NHWC
    if (graph_data.af && graph_data.conv_count > 1) {
        auto af_result = graph_data.af->clone_with_new_inputs({conv_result});
        copy_runtime_info(graph_data.conv, af_result);
        conv_result = af_result;
    }
    // We need to put the same name as before for the Convolution layer, so its output can be used as network result
    std::string conv_result_name = graph_data.last_op_in_sequence_for_replacement->get_friendly_name();
    replace_node(graph_data.last_op_in_sequence_for_replacement, conv_result);
    conv_result->set_friendly_name(conv_result_name);
}

static bool Convert(const DeviceVersion& compile_target,
                    const InferenceEngine::Precision& gnaPrecision,
                    std::shared_ptr<ngraph::Node> leading_transpose,
                    std::shared_ptr<ngraph::Node> fq_filters,
                    std::shared_ptr<ngraph::Node> conv,
                    std::shared_ptr<ngraph::Node> trailing_transpose,
                    std::shared_ptr<ngraph::Node> fq_conv,
                    std::shared_ptr<ngraph::Node> bias,
                    std::shared_ptr<ngraph::Node> bias_const,
                    std::shared_ptr<ngraph::Node> fq_bias,
                    std::shared_ptr<ngraph::Node> max_pool,
                    std::shared_ptr<ngraph::Node> af,
                    std::shared_ptr<ngraph::Node> fq_af,
                    std::shared_ptr<ngraph::Node> last_op_for_replacement) {
    GraphData graph_data{std::dynamic_pointer_cast<ngraph::opset7::Transpose>(leading_transpose),
                         std::dynamic_pointer_cast<ngraph::opset7::FakeQuantize>(fq_filters),
                         std::dynamic_pointer_cast<ngraph::opset7::Convolution>(conv),
                         std::dynamic_pointer_cast<ngraph::opset7::Transpose>(trailing_transpose),
                         std::dynamic_pointer_cast<ngraph::opset7::FakeQuantize>(fq_conv),
                         std::dynamic_pointer_cast<ngraph::opset7::FakeQuantize>(fq_bias),
                         std::dynamic_pointer_cast<ngraph::opset7::MaxPool>(max_pool),
                         std::dynamic_pointer_cast<ngraph::op::util::UnaryElementwiseArithmetic>(af),
                         std::dynamic_pointer_cast<ngraph::opset7::FakeQuantize>(fq_af),
                         last_op_for_replacement,
                         bias_const,
                         1,
                         1,
                         1};
    ConvData conv_data;

    if (!VerifyAndGetConvData(std::dynamic_pointer_cast<ngraph::opset7::Convolution>(conv), conv_data))
        return false;

    if (max_pool && !VerifyMaxPool(graph_data, std::dynamic_pointer_cast<ngraph::opset7::MaxPool>(max_pool)))
        return false;

    // If compile target is GNA 3.0 and the convolution is supported on it, then skip decomposition
    if (GNA30SupportedConv(compile_target, gnaPrecision, graph_data, conv_data))
        return false;

    // We are looking for Transpose(NHWC->NCHW) => Conv => Transpose(NCHW->NHWC)
    // or similar cases, so required network must be in NHWC order like in TF
    if (!TransposeOrderMatches(std::dynamic_pointer_cast<ngraph::opset7::Transpose>(leading_transpose), {0, 3, 1, 2}))
        return false;

    if (!TransposeOrderMatches(std::dynamic_pointer_cast<ngraph::opset7::Transpose>(trailing_transpose), {0, 2, 3, 1}))
        return false;

    if (!ShouldDecompose(graph_data, conv_data))
        return false;

    // All checks applied - now we may start decomposition
    Decompose(graph_data, conv_data);

    return true;
}

Decompose2DConv::Decompose2DConv(const DeviceVersion& compile_target, const InferenceEngine::Precision& gnaPrecision) {
    MATCHER_SCOPE(Decompose2DConv);

    auto const_input = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto leading_transpose =
        ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({ngraph::pattern::any_input(), const_input},
                                                              consumers_and_rank(1, 4));
    auto filters_const = ngraph::pattern::wrap_type<ngraph::opset7::Constant>(ngraph::pattern::rank_equals(4));
    auto fq_filters = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>(
        {const_input, const_input, const_input, const_input, const_input},
        consumers_and_rank(1, 4));
    auto filters = std::make_shared<ngraph::pattern::op::Or>(ngraph::OutputVector{filters_const, fq_filters});
    auto conv =
        ngraph::pattern::wrap_type<ngraph::opset7::Convolution>({leading_transpose, filters}, consumers_and_rank(1, 4));
    auto bias =
        ngraph::pattern::wrap_type<ngraph::opset7::Add>({conv, const_input}, ngraph::pattern::consumers_count(1));
    auto fq_bias = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>(
        {bias, const_input, const_input, const_input, const_input},
        ngraph::pattern::consumers_count(1));
    auto max_pool1 = ngraph::pattern::wrap_type<ngraph::opset7::MaxPool>({bias}, ngraph::pattern::consumers_count(1));
    auto max_pool2 =
        ngraph::pattern::wrap_type<ngraph::opset7::MaxPool>({fq_bias}, ngraph::pattern::consumers_count(1));
    auto af1 = ngraph::pattern::wrap_type<ngraph::opset7::Relu,
                                          ngraph::opset7::Sigmoid,
                                          ngraph::opset7::Tanh,
                                          ngraph::opset7::Abs,
                                          ngraph::opset7::Log,
                                          ngraph::opset7::Exp,
                                          ngraph::opset7::Sign,
                                          ngraph::opset7::Clamp>({conv}, ngraph::pattern::consumers_count(1));
    auto af2 = ngraph::pattern::wrap_type<ngraph::opset7::Relu,
                                          ngraph::opset7::Sigmoid,
                                          ngraph::opset7::Tanh,
                                          ngraph::opset7::Abs,
                                          ngraph::opset7::Log,
                                          ngraph::opset7::Exp,
                                          ngraph::opset7::Sign,
                                          ngraph::opset7::Clamp>({bias}, ngraph::pattern::consumers_count(1));
    auto af3 = ngraph::pattern::wrap_type<ngraph::opset7::Relu,
                                          ngraph::opset7::Sigmoid,
                                          ngraph::opset7::Tanh,
                                          ngraph::opset7::Abs,
                                          ngraph::opset7::Log,
                                          ngraph::opset7::Exp,
                                          ngraph::opset7::Sign,
                                          ngraph::opset7::Clamp>({fq_bias}, ngraph::pattern::consumers_count(1));
    auto af4 = ngraph::pattern::wrap_type<ngraph::opset7::Relu,
                                          ngraph::opset7::Sigmoid,
                                          ngraph::opset7::Tanh,
                                          ngraph::opset7::Abs,
                                          ngraph::opset7::Log,
                                          ngraph::opset7::Exp,
                                          ngraph::opset7::Sign,
                                          ngraph::opset7::Clamp>({max_pool1}, ngraph::pattern::consumers_count(1));
    auto af5 = ngraph::pattern::wrap_type<ngraph::opset7::Relu,
                                          ngraph::opset7::Sigmoid,
                                          ngraph::opset7::Tanh,
                                          ngraph::opset7::Abs,
                                          ngraph::opset7::Log,
                                          ngraph::opset7::Exp,
                                          ngraph::opset7::Sign,
                                          ngraph::opset7::Clamp>({max_pool2}, ngraph::pattern::consumers_count(1));
    auto fq_af1 = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>(
        {af3, const_input, const_input, const_input, const_input},
        ngraph::pattern::consumers_count(1));
    auto fq_af2 = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>(
        {af5, const_input, const_input, const_input, const_input},
        ngraph::pattern::consumers_count(1));
    auto fq_conv = ngraph::pattern::wrap_type<ngraph::opset7::FakeQuantize>(
        {conv, const_input, const_input, const_input, const_input},
        ngraph::pattern::consumers_count(1));
    auto transpose_input = std::make_shared<ngraph::pattern::op::Or>(
        ngraph::OutputVector{conv, bias, max_pool1, max_pool2, fq_bias, af1, af2, af3, af4, fq_af1, fq_af2, fq_conv});
    auto trailing_transpose =
        ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({transpose_input, const_input}, consumers_and_rank(1, 4));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto fq_filters_it = pattern_map.find(fq_filters);
        auto fq_filters_node =
            (fq_filters_it == std::end(pattern_map) ? nullptr : fq_filters_it->second.get_node_shared_ptr());
        auto bias_it = pattern_map.find(bias);
        auto bias_node = (bias_it == std::end(pattern_map) ? nullptr : bias_it->second.get_node_shared_ptr());
        std::shared_ptr<ngraph::Node> bias_const_node = nullptr;

        if (bias_node && !(bias_const_node = VerifyBiasGetConst(pattern_map.at(conv).get_node_shared_ptr(), bias_node)))
            return false;

        auto fq_conv_it = pattern_map.find(fq_conv);
        auto fq_conv_node = (fq_conv_it == std::end(pattern_map) ? nullptr : fq_conv_it->second.get_node_shared_ptr());
        auto fq_bias_it = pattern_map.find(fq_bias);
        auto fq_bias_node = (fq_bias_it == std::end(pattern_map) ? nullptr : fq_bias_it->second.get_node_shared_ptr());
        auto fq_af1_it = pattern_map.find(fq_af1);
        auto fq_af2_it = pattern_map.find(fq_af2);
        auto fq_af_node =
            (fq_af1_it == std::end(pattern_map)
                 ? ((fq_af2_it == std::end(pattern_map) ? nullptr : fq_af2_it->second.get_node_shared_ptr()))
                 : fq_af1_it->second.get_node_shared_ptr());
        auto max_pool1_it = pattern_map.find(max_pool1);
        auto max_pool2_it = pattern_map.find(max_pool2);
        auto max_pool_node =
            (max_pool1_it == std::end(pattern_map)
                 ? ((max_pool2_it == std::end(pattern_map) ? nullptr : max_pool2_it->second.get_node_shared_ptr()))
                 : max_pool1_it->second.get_node_shared_ptr());
        std::shared_ptr<ngraph::Node> af_node = nullptr;
        std::vector<ngraph::pattern::PatternValueMap::const_iterator> af_it{pattern_map.find(af1),
                                                                            pattern_map.find(af2),
                                                                            pattern_map.find(af3),
                                                                            pattern_map.find(af4)};

        for (auto const& af : af_it) {
            if (af != std::end(pattern_map)) {
                af_node = af->second.get_node_shared_ptr();
                break;
            }
        }

        return Convert(compile_target,
                       gnaPrecision,
                       pattern_map.at(leading_transpose).get_node_shared_ptr(),
                       fq_filters_node,
                       pattern_map.at(conv).get_node_shared_ptr(),
                       pattern_map.at(trailing_transpose).get_node_shared_ptr(),
                       fq_conv_node,
                       bias_node,
                       bias_const_node,
                       fq_bias_node,
                       max_pool_node,
                       af_node,
                       fq_af_node,
                       pattern_map.at(trailing_transpose).get_node_shared_ptr());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(trailing_transpose, matcher_name);
    this->register_matcher(m, callback);
}

Decompose2DConvTransposedWithBias::Decompose2DConvTransposedWithBias(const DeviceVersion& compile_target,
                                                                     const InferenceEngine::Precision& gnaPrecision) {
    MATCHER_SCOPE(Decompose2DConvTransposedWithBias);

    auto const_input_i64 =
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(ngraph::pattern::type_matches(ngraph::element::i64));
    auto const_input = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto leading_transpose =
        ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({ngraph::pattern::any_input(), const_input_i64},
                                                              consumers_and_rank(1, 4));
    auto conv = ngraph::pattern::wrap_type<ngraph::opset7::Convolution>(
        {leading_transpose, ngraph::pattern::wrap_type<ngraph::opset7::Constant>(ngraph::pattern::rank_equals(4))},
        consumers_and_rank(1, 4));
    auto trailing_transpose =
        ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({conv, const_input_i64}, consumers_and_rank(1, 4));
    auto bias = ngraph::pattern::wrap_type<ngraph::opset7::Add>({trailing_transpose, const_input},
                                                                ngraph::pattern::consumers_count(1));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        std::shared_ptr<ngraph::Node> bias_const_node = nullptr;

        if (!(bias_const_node = VerifyBiasGetConst(pattern_map.at(conv).get_node_shared_ptr(),
                                                   pattern_map.at(bias).get_node_shared_ptr())))
            return false;

        return Convert(compile_target,
                       gnaPrecision,
                       pattern_map.at(leading_transpose).get_node_shared_ptr(),
                       nullptr,
                       pattern_map.at(conv).get_node_shared_ptr(),
                       pattern_map.at(trailing_transpose).get_node_shared_ptr(),
                       nullptr,
                       pattern_map.at(bias).get_node_shared_ptr(),
                       bias_const_node,
                       nullptr,
                       nullptr,
                       nullptr,
                       nullptr,
                       pattern_map.at(bias).get_node_shared_ptr());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(bias, matcher_name);
    this->register_matcher(m, callback);
}

Decompose2DConvTransposedWithBiasAF::Decompose2DConvTransposedWithBiasAF(
    const DeviceVersion& compile_target,
    const InferenceEngine::Precision& gnaPrecision) {
    MATCHER_SCOPE(Decompose2DConvTransposedWithBiasAF);

    auto const_input_i64 =
        ngraph::pattern::wrap_type<ngraph::opset7::Constant>(ngraph::pattern::type_matches(ngraph::element::i64));
    auto const_input = ngraph::pattern::wrap_type<ngraph::opset7::Constant>();
    auto leading_transpose =
        ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({ngraph::pattern::any_input(), const_input_i64},
                                                              consumers_and_rank(1, 4));
    auto conv = ngraph::pattern::wrap_type<ngraph::opset7::Convolution>(
        {leading_transpose, ngraph::pattern::wrap_type<ngraph::opset7::Constant>(ngraph::pattern::rank_equals(4))},
        consumers_and_rank(1, 4));
    auto trailing_transpose =
        ngraph::pattern::wrap_type<ngraph::opset7::Transpose>({conv, const_input_i64}, consumers_and_rank(1, 4));
    auto bias = ngraph::pattern::wrap_type<ngraph::opset7::Add>({trailing_transpose, const_input},
                                                                ngraph::pattern::consumers_count(1));
    auto af = ngraph::pattern::wrap_type<ngraph::opset7::Relu,
                                         ngraph::opset7::Sigmoid,
                                         ngraph::opset7::Tanh,
                                         ngraph::opset7::Abs,
                                         ngraph::opset7::Log,
                                         ngraph::opset7::Exp,
                                         ngraph::opset7::Sign,
                                         ngraph::opset7::Clamp>({bias}, ngraph::pattern::consumers_count(1));

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        std::shared_ptr<ngraph::Node> bias_const_node = nullptr;

        if (!(bias_const_node = VerifyBiasGetConst(pattern_map.at(conv).get_node_shared_ptr(),
                                                   pattern_map.at(bias).get_node_shared_ptr())))
            return false;

        return Convert(compile_target,
                       gnaPrecision,
                       pattern_map.at(leading_transpose).get_node_shared_ptr(),
                       nullptr,
                       pattern_map.at(conv).get_node_shared_ptr(),
                       pattern_map.at(trailing_transpose).get_node_shared_ptr(),
                       nullptr,
                       pattern_map.at(bias).get_node_shared_ptr(),
                       bias_const_node,
                       nullptr,
                       nullptr,
                       pattern_map.at(af).get_node_shared_ptr(),
                       nullptr,
                       pattern_map.at(af).get_node_shared_ptr());
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(af, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
