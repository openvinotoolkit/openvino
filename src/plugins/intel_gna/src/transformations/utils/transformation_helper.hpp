// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ngraph_ops/convolution_ie.hpp>
#include <ngraph/opsets/opset7.hpp>

#include "openvino/opsets/opset12.hpp"
#include "ops/gna_convolution.hpp"

namespace ov {
namespace intel_gna {
namespace pass {
namespace helper {

struct ConvData {
    size_t input_height;
    size_t input_width;
    size_t input_channel_count;
    size_t filter_height;
    size_t filter_width;
    size_t filter_count;
    size_t filter_channel_count;
    size_t filter_dilation_height;
    size_t filter_dilation_width;
    size_t filter_stride_height;
    size_t filter_stride_width;
    size_t output_height;
    size_t output_width;
    size_t output_channel_count;
    size_t pads_begin_width;
    size_t pads_begin_height;
    size_t pads_end_width;
    size_t pads_end_height;
    ngraph::op::PadType padding_type;
    ngraph::element::Type element_type;
};

/**
 * @brief gets all convolution related data into a struct for further processing
 * @param conv convolution node to get data of
 * @param conv_data convolution data structure to put data into
 * @return void
 */
void GetConvData(std::shared_ptr<ngraph::opset7::Convolution> conv, ConvData& conv_data);

/**
 * @brief gets all convolution related data into a struct for further processing
 * @param conv GNA custom convolution node to get data of
 * @param conv_data convolution data structure to put data into
 * @return void
 */
void GetConvData(std::shared_ptr<ov::intel_gna::op::GNAConvolution> conv, ConvData& conv_data);

/**
 * @brief ngraph matcher predicate fusing existing predicates for consumers count and rank of a layer
 * @param expected_count expected consumers count for of node
 * @param expected_rank expected node rank
 * @return predicate function wrapper
 */
std::function<bool(ngraph::Output<ngraph::Node>)> consumers_and_rank(const size_t expected_count,
                                                                     const ngraph::Dimension& expected_rank);

/**
 * @brief checks whether transpose matches a given order
 * @param transpose transpose layer
 * @param order order of transposition to be compared with
 * @return true if the order matches, false otherwise
 */
bool TransposeOrderMatches(std::shared_ptr<ngraph::opset7::Transpose> transpose, std::vector<size_t> order);

/**
 * @brief performs a crop of a flattened input tensor
 * @param input input layer
 * @param offset offset to start the crop at*
 * @param size size of the crop
 * @return pointer to the newly created slice
 */
std::shared_ptr<ngraph::opset7::StridedSlice> FlatCrop(ngraph::Output<ngraph::Node> input, size_t offset, size_t size);

/**
 * @brief checks whether an add present after convolution is a bias and gets its const input
 * @param conv convolution layer preceding potential bias
 * @param add potential bias layer passed from ngraph matcher
 * @return bias const if the add layer present after convolution is a bias, nullptr otherwise
 */
std::shared_ptr<ngraph::Node> VerifyBiasGetConst(std::shared_ptr<ngraph::Node> conv, std::shared_ptr<ngraph::Node> add);

/**
 * @brief inserts a new fake quantize layer copied from an existing one and connects it to the output of a given layer
 * @param fq_layer existing fake quantize layer to be copied
 * @param last_node the node to which output the new fake quantize layer will be connected
 * @return new fake quantize layer or the last node
 */
std::shared_ptr<ngraph::Node> InsertFQLayer(const std::shared_ptr<ngraph::opset7::FakeQuantize> fq_layer,
                                            std::shared_ptr<ngraph::Node> last_node);

/**
 * @brief removes single node and inserts Reshape if its input and output shapes differ
 * @param node the node to be removed
 * @return void
 */
void remove_single_input_node(std::shared_ptr<ov::Node> node);

/**
 * @brief Swaps @args output tensor names
 */
void swap_output_names(ov::Output<ov::Node>, ov::Output<ov::Node>);

/**
 * @brief Swaps @args friendly names
 */
void swap_friendly_names(std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>);

/**
 * @brief Swaps @args output tensor names and friendly names
 */
void swap_names(std::shared_ptr<ov::Node>, std::shared_ptr<ov::Node>);

/**
 * @brief Reverses axis order. Final result will be such an order, that together
 * with initial order will be {0, 1, 2, ...}
 */
ov::AxisVector reverse_transpose_order(const ov::AxisVector& axis_order);

/**
 * @brief Finds all input node transposes
 */
ov::NodeVector find_input_transposes(const std::shared_ptr<const ov::Node>& node);

/**
 * @brief Marks all input transposes with flag NoSinking
 */
void mark_input_transposes_as_nosinking(std::shared_ptr<const ov::Node> node);

struct TransposeInfo {
    std::shared_ptr<ov::opset12::Transpose> transpose;
    std::shared_ptr<ov::opset12::Constant> transpose_const;

    bool isEmpty() const {
        return !transpose || !transpose_const;
    }
};

/**
 * @brief Finds first input node transpose
 */
TransposeInfo get_first_input_transpose(const std::shared_ptr<const ov::Node>& node);

/**
 * @brief Finds first output node transpose
 */
TransposeInfo get_first_output_transpose(const std::shared_ptr<const ov::Node>& node);

}  // namespace helper
}  // namespace pass
}  // namespace intel_gna
}  // namespace ov
