// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <ngraph/runtime/host_tensor.hpp>
#include <openvino/core/node.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset2.hpp>
#include <openvino/opsets/opset4.hpp>
#include <openvino/opsets/opset5.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset7.hpp>
#include <openvino/opsets/opset8.hpp>

#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"

#include "assign_shape_inference.hpp"
#include "augru_cell_shape_inference.hpp"
#include "augru_sequence_shape_inference.hpp"
#include "batch_to_space_shape_inference.hpp"
#include "broadcast_shape_inference.hpp"
#include "bucketize_shape_inference.hpp"
#include "concat_shape_inference.hpp"
#include "convolution_shape_inference.hpp"
#include "ctc_greedy_decoder_seq_len_shape_inference.hpp"
#include "ctc_greedy_decoder_shape_inference.hpp"
#include "ctc_loss_shape_inference.hpp"
#include "depth_to_space_shape_inference.hpp"
#include "detection_output_shape_inference.hpp"
#include "einsum_shape_inference.hpp"
#include "embedding_segments_sum_shape_inference.hpp"
#include "embeddingbag_offsets_shape_inference.hpp"
#include "experimental_detectron_detection_output_shape_inference.hpp"
#include "experimental_detectron_generate_proposals_shape_inference.hpp"
#include "experimental_detectron_prior_grid_generator_shape_inference.hpp"
#include "experimental_detectron_roi_feature_shape_inference.hpp"
#include "experimental_detectron_topkrois_shape_inference.hpp"
#include "extract_image_patches_shape_inference.hpp"
#include "eye_shape_inference.hpp"
#include "fake_quantize.hpp"
#include "fft_base_shape_inference.hpp"
#include "gather_elements_shape_inference.hpp"
#include "gather_shape_inference.hpp"
#include "gather_tree_shape_inference.hpp"
#include "grid_sample_shape_inference.hpp"
#include "gru_sequence_shape_inference.hpp"
#include "gru_cell_shape_inference.hpp"
#include "interpolate_shape_inference.hpp"
#include "lstm_cell_shape_inference.hpp"
#include "matmul_shape_inference.hpp"
#include "one_hot_shape_inference.hpp"
#include "pad_shape_inference.hpp"
#include "proposal_shape_inference.hpp"
#include "range_shape_inference.hpp"
#include "read_value_shape_inference.hpp"
#include "reduce_shape_inference.hpp"
#include "region_yolo_shape_inference.hpp"
#include "reorg_yolo_shape_inference.hpp"
#include "reverse_sequence_shape_inference.hpp"
#include "roi_align_shape_inference.hpp"
#include "roll_shape_inference.hpp"
#include "scatter_elements_update_shape_inference.hpp"
#include "scatter_nd_base_shape_inference.hpp"
#include "select_shape_inference.hpp"
#include "shape_inference.hpp"
#include "shape_nodes.hpp"
#include "shuffle_channels_shape_inference.hpp"
#include "space_to_batch_shape_inference.hpp"
#include "space_to_depth_shape_inference.hpp"
#include "split_shape_inference.hpp"
#include "squeeze_shape_inference.hpp"
#include "static_shape.hpp"
#include "strided_slice_shape_inference.hpp"
#include "tile_shape_inference.hpp"
#include "topk_shape_inference.hpp"
#include "transpose_shape_inference.hpp"
#include "unsqueeze_shape_inference.hpp"
#include "utils.hpp"
#include "variadic_split_shape_inference.hpp"

namespace ov {
namespace intel_cpu {

void shape_inference(ov::Node* op,
                     const std::vector<StaticShape>& input_shapes,
                     std::vector<StaticShape>& output_shapes,
                     const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) {
    auto shapeInfer = make_shape_inference(op->shared_from_this());
    output_shapes = shapeInfer->infer(input_shapes, constant_data);
}

class entryBase : public IShapeInfer {
public:
    entryBase(std::shared_ptr<ov::Node> node) : node(node) {
        for (size_t i = 0; i < node->get_input_size(); i++) {
            const auto& shape = node->get_input_partial_shape(i);
            if (shape.rank().is_static()) {
                input_ranks.push_back(shape.rank().get_length());
            } else {
                input_ranks.push_back(-1);
            }
        }
    }

    const ov::CoordinateDiff& get_pads_begin() override {
        OPENVINO_ASSERT(false, "entryBase do not support get_pads_begin() by default.");
    }

    const ov::CoordinateDiff& get_pads_end() override {
        OPENVINO_ASSERT(false, "entryBase do not support get_pads_end() by default.");
    }

    const std::vector<int64_t>& get_input_ranks() override {
        return input_ranks;
    }

protected:
    std::vector<int64_t> input_ranks;
    std::shared_ptr<ov::Node> node;
};

template <typename OP>
class entryIO : public entryBase {
public:
    using entryBase::entryBase;

    std::vector<StaticShape> infer(
        const std::vector<StaticShape>& input_shapes,
        const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        auto op = static_cast<OP*>(node.get());
        std::vector<StaticShape> output_shapes(op->get_output_size());
        shape_infer(op, input_shapes, output_shapes);
        return output_shapes;
    }
};

template <typename OP>
class entryIOC : public entryBase {
public:
    using entryBase::entryBase;

    std::vector<StaticShape> infer(
        const std::vector<StaticShape>& input_shapes,
        const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        auto op = static_cast<OP*>(node.get());
        std::vector<StaticShape> output_shapes(op->get_output_size());
        shape_infer(op, input_shapes, output_shapes, constant_data);
        return output_shapes;
    }
};

class entryCopy : public entryBase {
public:
    using entryBase::entryBase;

    std::vector<StaticShape> infer(
        const std::vector<StaticShape>& input_shapes,
        const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        auto op = node.get();
        std::vector<StaticShape> output_shapes(op->get_output_size());
        copy_shape_infer(op, input_shapes, output_shapes);
        return output_shapes;
    }
};

class entryFirstPassthrough : public entryBase {
public:
    using entryBase::entryBase;

    std::vector<StaticShape> infer(
        const std::vector<StaticShape>& input_shapes,
        const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        auto op = node.get();
        std::vector<StaticShape> output_shapes(op->get_output_size());
        first_input_passthrough_infer(op, input_shapes, output_shapes);
        return output_shapes;
    }
};

class entryEltwise : public entryBase {
public:
    using entryBase::entryBase;

    std::vector<StaticShape> infer(
        const std::vector<StaticShape>& input_shapes,
        const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        auto op = node.get();
        std::vector<StaticShape> output_shapes(op->get_output_size());
        eltwise_shape_infer(op, input_shapes, output_shapes);
        return output_shapes;
    }
};

class entryFallback : public entryBase {
public:
    std::shared_ptr<ov::Node> local_op_default;

    entryFallback(std::shared_ptr<ov::Node> node) : entryBase(node) {
        ngraph::OutputVector new_inputs;
        auto op = node.get();
        for (size_t i = 0; i < op->get_input_size(); ++i) {
            if (dynamic_cast<ov::opset1::Constant*>(op->get_input_node_ptr(i))) {
                new_inputs.push_back(op->get_input_node_ptr(i)->clone_with_new_inputs(ov::OutputVector{}));
            } else {
                new_inputs.push_back(std::make_shared<ov::opset1::Parameter>(op->get_input_element_type(i),
                                                                             op->get_input_partial_shape(i)));
            }
        }

        local_op_default = op->clone_with_new_inputs(new_inputs);
    }

    virtual void post_validate_and_infer_types(const std::shared_ptr<ov::Node>& local_op) {}

    std::vector<StaticShape> infer(
        const std::vector<StaticShape>& input_shapes,
        const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        auto op = node.get();
        std::vector<StaticShape> output_shapes;

        std::shared_ptr<ov::Node> local_op;
        if (!constant_data.empty()) {
            ngraph::OutputVector new_inputs;
            for (size_t i = 0; i < op->get_input_size(); ++i) {
                if (constant_data.count(i)) {
                    new_inputs.push_back(std::make_shared<ov::opset1::Constant>(constant_data.at(i)));
                } else if (dynamic_cast<ov::opset1::Constant*>(op->get_input_node_ptr(i))) {
                    new_inputs.push_back(op->get_input_node_ptr(i)->clone_with_new_inputs(ov::OutputVector{}));
                } else {
                    new_inputs.push_back(std::make_shared<ov::opset1::Parameter>(op->get_input_element_type(i),
                                                                                 input_shapes[i].to_partial_shape()));
                }
            }
            local_op = op->clone_with_new_inputs(new_inputs);
        } else {
            local_op = local_op_default;
            OPENVINO_SUPPRESS_DEPRECATED_START
            for (size_t i = 0; i < local_op->get_input_size(); i++) {
                if (dynamic_cast<ov::opset1::Parameter*>(local_op->get_input_node_ptr(i))) {
                    local_op->get_input_tensor(i).set_partial_shape(input_shapes[i].to_partial_shape());
                }
            }
            OPENVINO_SUPPRESS_DEPRECATED_END
        }

        local_op->validate_and_infer_types();

        output_shapes.resize(local_op->get_output_size());
        for (size_t i = 0; i < output_shapes.size(); ++i) {
            const auto& partial_shape = local_op->get_output_partial_shape(i);

            if (partial_shape.is_dynamic()) {
                std::ostringstream errorMessage;
                errorMessage << "Can't compute static output shape on " << i
                             << " port for " << op->get_type_name() << " node with name: " << op->get_name();
                errorMessage << ". Input shapes = ( ";
                for (size_t in = 0; in < op->get_input_size(); in++) {
                    errorMessage << in << " port = " << op->get_input_partial_shape(in) << ", ";
                }
                errorMessage << "). Output shapes = ( ";
                for (size_t out = 0; out < op->get_output_size(); out++) {
                    errorMessage << out << " port = " << op->get_output_partial_shape(out) << ", ";
                }
                errorMessage << ")";
                OPENVINO_ASSERT(false, errorMessage.str());
            }

            output_shapes[i] = StaticShape(partial_shape.to_shape());
        }

        post_validate_and_infer_types(local_op);

        return output_shapes;
    }
};

static inline ov::CoordinateDiff convertPadding(const ov::CoordinateDiff& newPads) {
    return newPads;
}

static inline ov::CoordinateDiff convertPadding(const ov::Shape& newPads) {
    std::vector<ptrdiff_t> pads(newPads.size());
    for (int i = 0; i < newPads.size(); i++) {
        pads[i] = static_cast<ptrdiff_t>(newPads[i]);
    }
    return pads;
}

template <typename OP>
class entryFallbackWithPadding : public entryFallback {
public:
    using entryFallback::entryFallback;

    ov::CoordinateDiff pads_begin, pads_end;

    const ov::CoordinateDiff& get_pads_begin() override {
        return pads_begin;
    }
    const ov::CoordinateDiff& get_pads_end() override {
        return pads_end;
    }

    void post_validate_and_infer_types(const std::shared_ptr<ov::Node>& local_op) override {
        auto node = dynamic_cast<OP*>(local_op.get());
        OPENVINO_ASSERT(node);
        pads_begin = convertPadding(node->get_pads_begin());
        pads_end = convertPadding(node->get_pads_end());
    }
};

template <typename OP>
class entryInterpolate : public entryBase {
public:
    using entryBase::entryBase;

    std::vector<StaticShape> infer(
        const std::vector<StaticShape>& input_shapes,
        const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        std::vector<size_t> pads_begin, pads_end;
        auto op = static_cast<OP*>(node.get());
        std::vector<StaticShape> output_shapes(op->get_output_size());
        correct_pads_attr(op, pads_begin, pads_end, input_shapes);
        shape_infer(op, pads_begin, pads_end, input_shapes, output_shapes, constant_data);
        return output_shapes;
    }
};

template <typename OP>
class entryConv : public entryBase {
public:
    entryConv(std::shared_ptr<OP> node, bool is_grouped) : entryBase(node), is_grouped(is_grouped) {}
    const ov::CoordinateDiff& get_pads_begin() override {
        return pads_begin;
    }
    const ov::CoordinateDiff& get_pads_end() override {
        return pads_end;
    }
    std::vector<StaticShape> infer(
        const std::vector<StaticShape>& input_shapes,
        const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        auto op = static_cast<OP*>(node.get());
        std::vector<StaticShape> output_shapes(op->get_output_size());
        bool status = resolve_auto_pad_for_shape(op, pads_begin, pads_end, input_shapes, 2, is_grouped ? 3 : 2);
        OPENVINO_ASSERT(status,
                        "Convolution shape inference doesn't have enough information to calculate static shapes");
        shape_infer(op, pads_begin, pads_end, input_shapes, output_shapes);
        return output_shapes;
    }

protected:
    ov::CoordinateDiff pads_begin, pads_end;
    bool is_grouped;
};

template <typename OP>
class entryConvBackprop : public entryBase {
public:
    entryConvBackprop(std::shared_ptr<OP> node, bool is_grouped) : entryBase(node), is_grouped(is_grouped) {}
    const ov::CoordinateDiff& get_pads_begin() override {
        return pads_begin;
    }
    const ov::CoordinateDiff& get_pads_end() override {
        return pads_end;
    }
    std::vector<StaticShape> infer(
        const std::vector<StaticShape>& input_shapes,
        const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data) override {
        StaticShape output_shape_input;
        auto op = static_cast<OP*>(node.get());
        std::vector<StaticShape> output_shapes(op->get_output_size());
        if (op->get_input_size() == 3)
            get_data_as_shape<StaticShape>(2, op, output_shape_input, constant_data);
        bool status = resolve_auto_pad_for_shape_back_prop(op,
                                                           pads_begin,
                                                           pads_end,
                                                           input_shapes,
                                                           output_shape_input,
                                                           2,
                                                           is_grouped ? 3 : 2);
        OPENVINO_ASSERT(
            status,
            "ConvolutionBackpropData shape inference doesn't have enough information to calculate static shapes");
        shape_infer(op, pads_begin, pads_end, output_shape_input, input_shapes, output_shapes);
        return output_shapes;
    }

protected:
    ov::CoordinateDiff pads_begin, pads_end;
    bool is_grouped;
};

template <typename OP>
std::shared_ptr<entryIOC<OP>> make_shared_entryIOC(std::shared_ptr<OP> node) {
    return std::make_shared<entryIOC<OP>>(node);
}

template <typename OP>
std::shared_ptr<entryIO<OP>> make_shared_entryIO(std::shared_ptr<OP> node) {
    return std::make_shared<entryIO<OP>>(node);
}

std::shared_ptr<IShapeInfer> make_shape_inference(const std::shared_ptr<ngraph::Node>& op) {
    if (auto node = ov::as_type_ptr<ov::opset8::Convolution>(op)) {
        return std::make_shared<entryConv<ov::opset8::Convolution>>(node, false);
    } else if (auto node = ov::as_type_ptr<ov::opset8::GroupConvolution>(op)) {
        return std::make_shared<entryConv<ov::opset8::GroupConvolution>>(node, true);
    } else if (auto node = ov::as_type_ptr<ov::opset8::ConvolutionBackpropData>(op)) {
        return std::make_shared<entryConvBackprop<ov::opset8::ConvolutionBackpropData>>(node, false);
    } else if (auto node = ov::as_type_ptr<ov::opset8::GroupConvolutionBackpropData>(op)) {
        return std::make_shared<entryConvBackprop<ov::opset8::GroupConvolutionBackpropData>>(node, true);
    } else if (auto node = ov::as_type_ptr<ov::op::util::ArithmeticReductionKeepDims>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::op::util::LogicalReductionKeepDims>(op)) {
        return make_shared_entryIOC(node);
    } else if (ov::is_type<ov::op::util::UnaryElementwiseArithmetic>(op) || ov::is_type<ov::opset1::Convert>(op) ||
            ov::is_type<ov::opset1::LogicalNot>(op) || ov::is_type<ov::opset2::MVN>(op) ||
            ov::is_type<ov::opset1::Softmax>(op) || ov::is_type<ov::opset8::Softmax>(op)) {
        return std::make_shared<entryCopy>(op);
    } else if (ov::is_type<ov::opset6::MVN>(op) || ov::is_type<ov::opset1::LRN>(op) ||
            ov::is_type<ov::opset1::HardSigmoid>(op) || ov::is_type<ov::opset1::Selu>(op) ||
            ov::is_type<ov::opset1::PRelu>(op) || ov::is_type<ov::opset3::CumSum>(op) ||
            ov::is_type<ov::opset1::BatchNormInference>(op) || ov::is_type<ov::opset5::BatchNormInference>(op) ||
            ov::is_type<ov::opset4::Swish>(op) || ov::is_type<ov::opset1::NormalizeL2>(op)) {
        return std::make_shared<entryFirstPassthrough>(op);
    } else if (ov::is_type<ov::op::util::BinaryElementwiseArithmetic>(op) ||
               ov::is_type<ov::op::util::BinaryElementwiseComparison>(op) ||
               ov::is_type<ov::op::util::BinaryElementwiseLogical>(op)) {
        return std::make_shared<entryEltwise>(op);
    } else if (auto node = ov::as_type_ptr<ov::opset1::FakeQuantize>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Reshape>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Squeeze>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Unsqueeze>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::ShapeOf>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::ShapeOf>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::ExperimentalDetectronDetectionOutput>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::TopK>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::Bucketize>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::EmbeddingSegmentsSum>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::EmbeddingBagOffsetsSum>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::ExperimentalDetectronROIFeatureExtractor>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Pad>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset4::Range>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Range>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::RegionYolo>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset2::ReorgYolo>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Split>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::VariadicSplit>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset7::Einsum>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::StridedSlice>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::Assign>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::Assign>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::ExperimentalDetectronPriorGridGenerator>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::LSTMCell>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::LSTMCell>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::ReadValue>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::ReadValue>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::Tile>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::ExperimentalDetectronTopKROIs>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset4::Interpolate>(op)) {
        return std::make_shared<entryInterpolate<ov::opset4::Interpolate>>(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Interpolate>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::ScatterElementsUpdate>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset4::ScatterNDUpdate>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::GatherElements>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::op::util::GatherBase>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::GatherTree>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset9::GridSample>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset5::GRUSequence>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::op::internal::AUGRUSequence>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::GRUCell>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::op::internal::AUGRUCell>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::OneHot>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset4::CTCLoss>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset7::DFT>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset7::IDFT>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::CTCGreedyDecoderSeqLen>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::CTCGreedyDecoder>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::ExtractImagePatches>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::ReverseSequence>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset7::Roll>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset6::ExperimentalDetectronGenerateProposalsSingleImage>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset4::Proposal>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Proposal>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset3::ROIAlign>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::DetectionOutput>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset8::DetectionOutput>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Select>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::ShuffleChannels>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::MatMul>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset2::BatchToSpace>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset2::SpaceToBatch>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::DepthToSpace>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::SpaceToDepth>(op)) {
        return make_shared_entryIO(node);
    } else if (auto node = ov::as_type_ptr<ov::opset4::Broadcast>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Broadcast>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset9::Eye>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::op::v8::MaxPool>(op)) {
        return std::make_shared<entryFallbackWithPadding<ov::op::v8::MaxPool>>(node);
    } else if (auto node = ov::as_type_ptr<ov::op::v1::MaxPool>(op)) {
        return std::make_shared<entryFallbackWithPadding<ov::op::v1::MaxPool>>(node);
    } else if (auto node = ov::as_type_ptr<ov::op::v1::AvgPool>(op)) {
        return std::make_shared<entryFallbackWithPadding<ov::op::v1::AvgPool>>(node);
    } else if (auto node = ov::as_type_ptr<ov::op::v1::DeformableConvolution>(op)) {
        return std::make_shared<entryFallbackWithPadding<ov::op::v1::DeformableConvolution>>(node);
    } else if (auto node = ov::as_type_ptr<ov::op::v8::DeformableConvolution>(op)) {
        return std::make_shared<entryFallbackWithPadding<ov::op::v8::DeformableConvolution>>(node);
    } else if (auto node = ov::as_type_ptr<ov::opset8::Transpose>(op)) {
        return make_shared_entryIOC(node);
    } else if (auto node = ov::as_type_ptr<ov::opset1::Concat>(op)) {
        return make_shared_entryIO(node);
    } else {
        return std::make_shared<entryFallback>(op);
    }
}

}   // namespace intel_cpu
}   // namespace ov
