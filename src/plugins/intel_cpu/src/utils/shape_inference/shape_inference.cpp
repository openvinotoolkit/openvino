// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shape_inference.hpp"

#include <ngraph/runtime/host_tensor.hpp>
#include <openvino/core/node.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset2.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/opsets/opset5.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset7.hpp>
#include <openvino/opsets/opset8.hpp>

#include "assign_shape_inference.hpp"
#include "convolution_shape_inference.hpp"
#include "ctc_greedy_decoder_seq_len_shape_inference.hpp"
#include "ctc_greedy_decoder_shape_inference.hpp"
#include "experimental_detectron_detection_output_shape_inference.hpp"
#include "experimental_detectron_prior_grid_generator_shape_inference.hpp"
#include "experimental_detectron_topkrois_shape_inference.hpp"
#include "extract_image_patches_shape_inference.hpp"
#include "fake_quantize.hpp"
#include "gather_elements_shape_inference.hpp"
#include "gather_shape_inference.hpp"
#include "gather_tree_shape_inference.hpp"
#include "interpolate_shape_inference.hpp"
#include "lstm_cell_shape_inference.hpp"
#include "one_hot_shape_inference.hpp"
#include "read_value_shape_inference.hpp"
#include "reduce_shape_inference.hpp"
#include "reverse_sequence_shape_inference.hpp"
#include "scatter_elements_update_shape_inference.hpp"
#include "scatter_nd_base_shape_inference.hpp"
#include "ctc_loss_shape_inference.hpp"
#include "fft_base_shape_inference.hpp"
#include "shape_inference.hpp"
#include "shape_nodes.hpp"
#include "fake_quantize.hpp"
#include "experimental_detectron_detection_output_shape_inference.hpp"
#include "bucketize_shape_inference.hpp"
#include "embedding_segments_sum_shape_inference.hpp"
#include "embeddingbag_offsets_shape_inference.hpp"
#include "experimental_detectron_roi_feature_shape_inference.hpp"
#include "pad_shape_inference.hpp"
#include "range_shape_inference.hpp"
#include "region_yolo_shape_inference.hpp"
#include "reorg_yolo_shape_inference.hpp"
#include "split_shape_inference.hpp"
#include "topk_shape_inference.hpp"
#include "variadic_split_shape_inference.hpp"
#include "einsum_shape_inference.hpp"
#include "strided_slice_shape_inference.hpp"
#include "experimental_detectron_generate_proposals_shape_inference.hpp"
#include "roi_align_shape_inference.hpp"
#include "roll_shape_inference.hpp"
#include "proposal_shape_inference.hpp"
#include "static_shape.hpp"
#include "tile_shape_inference.hpp"
#include "utils.hpp"

void shape_inference(ov::Node* op,
                     const std::vector<ov::StaticShape>& input_shapes,
                     std::vector<ov::StaticShape>& output_shapes,
                     const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    if (auto node = ov::as_type<ov::opset8::Convolution>(op)) {
        ov::CoordinateDiff pads_begin, pads_end;
        bool status = resolve_auto_pad_for_shape(node, pads_begin, pads_end, input_shapes, 2, 2);
        OPENVINO_ASSERT(status,
                        "Convolution shape inference doesn't have enough information to calculate static shapes");
        shape_infer(node, pads_begin, pads_end, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset8::GroupConvolution>(op)) {
        ov::CoordinateDiff pads_begin, pads_end;
        bool status = resolve_auto_pad_for_shape(node, pads_begin, pads_end, input_shapes, 2, 3);
        OPENVINO_ASSERT(status,
                        "GroupConvolution shape inference doesn't have enough information to calculate static shapes");
        shape_infer(node, pads_begin, pads_end, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset8::ConvolutionBackpropData>(op)) {
        ov::CoordinateDiff pads_begin, pads_end;
        ov::StaticShape output_shape_input;
        if (node->get_input_size() == 3)
            get_data_as_shape<ov::StaticShape>(2, op, output_shape_input, constant_data);
        bool status =
            resolve_auto_pad_for_shape_back_prop(node, pads_begin, pads_end, input_shapes, output_shape_input, 2, 2);
        OPENVINO_ASSERT(
            status,
            "ConvolutionBackpropData shape inference doesn't have enough information to calculate static shapes");
        shape_infer(node, pads_begin, pads_end, output_shape_input, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset8::GroupConvolutionBackpropData>(op)) {
        ov::CoordinateDiff pads_begin, pads_end;
        ov::StaticShape output_shape_input;
        if (node->get_input_size() == 3)
            get_data_as_shape<ov::StaticShape>(2, op, output_shape_input, constant_data);
        bool status =
            resolve_auto_pad_for_shape_back_prop(node, pads_begin, pads_end, input_shapes, output_shape_input, 2, 3);
        OPENVINO_ASSERT(
            status,
            "GroupConvolutionBackpropData shape inference doesn't have enough information to calculate static shapes");
        shape_infer(node, pads_begin, pads_end, output_shape_input, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::op::util::ArithmeticReductionKeepDims>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::op::util::LogicalReductionKeepDims>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (ov::is_type<ov::op::util::UnaryElementwiseArithmetic>(op) ||
            ov::is_type<ov::opset1::Convert>(op) || ov::is_type<ov::opset1::Clamp>(op) ||
            ov::is_type<ov::opset1::GRN>(op) || ov::is_type<ov::opset1::LRN>(op) ||
            ov::is_type<ov::opset1::LogicalNot>(op) || ov::is_type<ov::opset4::Mish>(op) ||
            ov::is_type<ov::opset2::MVN>(op) || ov::is_type<ov::opset6::MVN>(op) ||
            ov::is_type<ov::opset1::PRelu>(op) || ov::is_type<ov::opset1::Relu>(op) ||
            ov::is_type<ov::opset4::Swish>(op) || ov::is_type<ov::opset1::Elu>(op) ||
            ov::is_type<ov::opset1::Softmax>(op) || ov::is_type<ov::opset8::Softmax>(op) ||
            ov::is_type<ov::opset5::Round>(op)) {
        copy_shape_infer(node, input_shapes, output_shapes);
    } else if (ov::is_type<ov::op::util::BinaryElementwiseArithmetic>(op) ||
               ov::is_type<ov::op::util::BinaryElementwiseComparison>(op) ||
               ov::is_type<ov::op::util::BinaryElementwiseLogical>(op)) {
        eltwise_shape_infer(op, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::FakeQuantize>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::Reshape>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::Squeeze>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::Unsqueeze>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::ShapeOf>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset3::ShapeOf>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::ExperimentalDetectronDetectionOutput>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset3::TopK>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset3::Bucketize>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset3::EmbeddingSegmentsSum>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset3::EmbeddingBagOffsetsSum>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::ExperimentalDetectronROIFeatureExtractor>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::Pad>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset4::Range>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::Range>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::RegionYolo>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset2::ReorgYolo>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::Split>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::VariadicSplit>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset7::Einsum>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::StridedSlice>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset3::Assign>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::Assign>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::ExperimentalDetectronPriorGridGenerator>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::LSTMCell>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::LSTMCell>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset3::ReadValue>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::ReadValue>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::Tile>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset6::ExperimentalDetectronTopKROIs>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset4::Interpolate>(op)) {
        std::vector<size_t> pads_begin, pads_end;
        correct_pads_attr(node, pads_begin, pads_end, input_shapes);
        shape_infer(node, pads_begin, pads_end, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::Interpolate>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset3::ScatterElementsUpdate>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset4::ScatterNDUpdate>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::GatherElements>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::op::util::GatherBase>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset1::GatherTree>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::OneHot>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset4::CTCLoss>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset7::DFT>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset7::IDFT>(op)) {
        shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset6::CTCGreedyDecoderSeqLen>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset6::CTCGreedyDecoder>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset3::ExtractImagePatches>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::ReverseSequence>(op)) {
        shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset7::Roll>(op)) {
      shape_infer(node, input_shapes, output_shapes, constant_data);
    } else if (auto node = ov::as_type<ov::opset6::ExperimentalDetectronGenerateProposalsSingleImage>(op)) {
      shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset4::Proposal>(op)) {
      shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset1::Proposal>(op)) {
      shape_infer(node, input_shapes, output_shapes);
    } else if (auto node = ov::as_type<ov::opset3::ROIAlign>(op)) {
      shape_infer(node, input_shapes, output_shapes);
    } else {
        ngraph::OutputVector new_inputs;
        for (size_t i = 0; i < op->get_input_size(); ++i) {
            if (constant_data.count(i)) {
                new_inputs.push_back(std::make_shared<ov::opset1::Constant>(constant_data.at(i)));
            } else {
                new_inputs.push_back(std::make_shared<ov::opset1::Parameter>(op->get_input_element_type(i),
                                                                             input_shapes[i].to_partial_shape()));
            }
        }
        const auto local_op = op->clone_with_new_inputs(new_inputs);
        local_op->validate_and_infer_types();

        output_shapes.resize(op->get_output_size());
        for (size_t i = 0; i < output_shapes.size(); ++i) {
            const auto& partial_shape = local_op->get_output_partial_shape(i);
            OPENVINO_ASSERT(
                partial_shape.is_static(),
                "On device shape infer shouldn't support default shape infer for nodes with internal dynamism");
            output_shapes[i] = ov::StaticShape(partial_shape.to_shape());
        }
    }
}
