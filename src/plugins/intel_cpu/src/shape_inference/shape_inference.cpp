// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shape_inference.hpp"

#include <openvino/core/node.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/opsets/opset12.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/opsets/opset14.hpp>
#include <openvino/opsets/opset2.hpp>
#include <openvino/opsets/opset5.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset7.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/opsets/opset9.hpp>

#include "adaptive_avg_pool_shape_inference.hpp"
#include "adaptive_max_pool_shape_inference.hpp"
#include "assign_shape_inference.hpp"
#include "augru_cell_shape_inference.hpp"
#include "augru_sequence_shape_inference.hpp"
#include "avg_pool_shape_inference.hpp"
#include "batch_to_space_shape_inference.hpp"
#include "binary_convolution_shape_inference.hpp"
#include "broadcast_shape_inference.hpp"
#include "bucketize_shape_inference.hpp"
#include "concat_shape_inference.hpp"
#include "convolution_backprop_shape_inference.hpp"
#include "convolution_shape_inference.hpp"
#include "copy_shape_inference.hpp"
#include "ctc_greedy_decoder_seq_len_shape_inference.hpp"
#include "ctc_greedy_decoder_shape_inference.hpp"
#include "ctc_loss_shape_inference.hpp"
#include "deformable_convolution_shape_inference.hpp"
#include "deformable_psroi_pooling_shape_inference.hpp"
#include "depth_to_space_shape_inference.hpp"
#include "detection_output_shape_inference.hpp"
#include "einsum_shape_inference.hpp"
#include "eltwise_shape_inference.hpp"
#include "embedding_segments_sum_shape_inference.hpp"
#include "embeddingbag_offsets_shape_inference.hpp"
#include "embeddingbag_packed_shape_inference.hpp"
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
#include "gather_nd_shape_inference.hpp"
#include "gather_shape_inference.hpp"
#include "gather_tree_shape_inference.hpp"
#include "grid_sample_shape_inference.hpp"
#include "group_convolution_backprop_shape_inference.hpp"
#include "group_convolution_shape_inference.hpp"
#include "gru_cell_shape_inference.hpp"
#include "gru_sequence_shape_inference.hpp"
#include "i420_shape_inference.hpp"
#include "interpolate_shape_inference.hpp"
#include "inverse_shape_inference.hpp"
#include "irdft_shape_inference.hpp"
#include "lstm_cell_shape_inference.hpp"
#include "lstm_sequence_shape_inference.hpp"
#include "matmul_shape_inference.hpp"
#include "matrix_nms_shape_inference.hpp"
#include "max_pool_shape_inference.hpp"
#include "multinomial_shape_inference.hpp"
#include "nms_shape_inference.hpp"
#include "nv12_shape_inference.hpp"
#include "one_hot_shape_inference.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset11.hpp"
#include "openvino/opsets/opset3.hpp"
#include "openvino/opsets/opset4.hpp"
#include "pad_shape_inference.hpp"
#include "prior_box_clustered_shape_inference.hpp"
#include "prior_box_shape_inference.hpp"
#include "proposal_shape_inference.hpp"
#include "psroi_pooling_shape_inference.hpp"
#include "random_uniform_shape_inference.hpp"
#include "range_shape_inference.hpp"
#include "rdft_shape_inference.hpp"
#include "reduce_shape_inference.hpp"
#include "region_yolo_shape_inference.hpp"
#include "reorg_yolo_shape_inference.hpp"
#include "reshape_shape_inference.hpp"
#include "reverse_sequence_shape_inference.hpp"
#include "reverse_shape_inference.hpp"
#include "rms_norm_shape_inference.hpp"
#include "rnn_cell_shape_inference.hpp"
#include "rnn_sequence_shape_inference.hpp"
#include "roi_align_shape_inference.hpp"
#include "roi_pooling_shape_inference.hpp"
#include "roll_shape_inference.hpp"
#include "scaled_dot_product_attention_shape_inference.hpp"
#include "scatter_elements_update_shape_inference.hpp"
#include "scatter_nd_base_shape_inference.hpp"
#include "select_shape_inference.hpp"
#include "shape_nodes.hpp"
#include "shuffle_channels_shape_inference.hpp"
#include "slice_shape_inference.hpp"
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
#include "utils/bit_util.hpp"
#include "variadic_split_shape_inference.hpp"

namespace ov {
namespace intel_cpu {
/**
 * @brief Base shape inference object implementing the IStaticShapeInfer without padding support.
 *
 * Default shape inference is first input pass as output shape.
 */
class ShapeInferBase : public IStaticShapeInfer {
public:
    using iface_type = IStaticShapeInfer;
    virtual ~ShapeInferBase() = default;

    ShapeInferBase(std::shared_ptr<Node> node) : m_input_ranks{}, m_node{node} {
        static_assert(std::is_same<int64_t, Dimension::value_type>::value, "Rank type not match to input_ranks type.");
        for (size_t i = 0; i < node->get_input_size(); ++i) {
            const auto& shape = node->get_input_partial_shape(i);
            const auto& rank_length = shape.rank().is_static() ? shape.rank().get_length() : -1;
            m_input_ranks.push_back(rank_length);
        }
    }

    ov::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                 const ov::ITensorAccessor&) override {
        NODE_VALIDATION_CHECK(m_node.get(), input_shapes.size() > 0, "Incorrect number of input shapes");
        return {std::vector<StaticShape>{input_shapes[0]}};
    }

    const ov::CoordinateDiff& get_pads_begin() override {
        OPENVINO_ASSERT(false, "ShapeInferBase do not support get_pads_begin() by default.");
    }

    const ov::CoordinateDiff& get_pads_end() override {
        OPENVINO_ASSERT(false, "ShapeInferBase do not support get_pads_end() by default.");
    }

    const std::vector<int64_t>& get_input_ranks() override {
        return m_input_ranks;
    }

    port_mask_t get_port_mask() const override {
        return 0;
    }

protected:
    std::vector<int64_t> m_input_ranks;
    std::shared_ptr<ov::Node> m_node;
};

/**
 * @brief Shape inference which copy single input shape to output shape.
 */
class ShapeInferCopy : public ShapeInferBase {
public:
    using ShapeInferBase::ShapeInferBase;

    ov::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                 const ov::ITensorAccessor&) override {
        return {op::copy_shape_infer(m_node.get(), input_shapes)};
    }
};

/**
 * @brief Shape inference applied for element wise operators.
 */
class ShapeInferEltwise : public ShapeInferBase {
public:
    using ShapeInferBase::ShapeInferBase;

    ov::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                 const ov::ITensorAccessor&) override {
        return {op::eltwise_shape_infer(m_node.get(), input_shapes)};
    }
};

/**
 * @brief Shape inference used as fallback if specific inference not implemented.
 */
class ShapeInferFallback : public ShapeInferBase {
public:
    using ShapeInferBase::ShapeInferBase;

    ov::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                 const ov::ITensorAccessor& tensor_accessor) override {
        auto op = m_node.get();
        std::vector<StaticShape> output_shapes;

        std::shared_ptr<ov::Node> local_op;
        ov::OutputVector new_inputs;
        for (size_t i = 0; i < op->get_input_size(); ++i) {
            if (auto t = tensor_accessor(i)) {
                new_inputs.push_back(
                    std::make_shared<ov::opset1::Constant>(t));
            } else if (dynamic_cast<ov::opset1::Constant*>(op->get_input_node_ptr(i))) {
                new_inputs.push_back(op->get_input_node_ptr(i)->clone_with_new_inputs(ov::OutputVector{}));
            } else {
                new_inputs.push_back(std::make_shared<ov::opset1::Parameter>(op->get_input_element_type(i),
                                                                             input_shapes[i].to_partial_shape()));
            }
        }
        local_op = op->clone_with_new_inputs(new_inputs);
        local_op->validate_and_infer_types();

        output_shapes.resize(local_op->get_output_size());
        for (size_t i = 0; i < output_shapes.size(); ++i) {
            const auto& partial_shape = local_op->get_output_partial_shape(i);

            if (partial_shape.is_dynamic()) {
                return {};
            }

            output_shapes[i] = StaticShape(partial_shape.to_shape());
        }

        return {std::move(output_shapes)};
    }
};

template <class TOp, uint32_t MASK>
class ShapeInferTA : public ShapeInferBase {
public:
    using ShapeInferBase::ShapeInferBase;

    ov::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                 const ov::ITensorAccessor& tensor_accessor) override {
        return {shape_infer(static_cast<TOp*>(m_node.get()), input_shapes, tensor_accessor)};
    }

    port_mask_t get_port_mask() const override {
        return MASK;
    }
};

/**
 * @brief Shape inference not using tensor accessor
 *
 * The MASK is 0 there is no dependant inputs with data for shape inference.
 *
 * @tparam TOp  Type of operator.
 */
template <class TOp>
class ShapeInferTA<TOp, 0> : public ShapeInferBase {
public:
    using ShapeInferBase::ShapeInferBase;

    ov::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                 const ov::ITensorAccessor&) override {
        return {shape_infer(static_cast<TOp*>(m_node.get()), input_shapes)};
    }
};

/** @brief Base shape inference object implementing the IStaticShapeInfer with padding support. */
class ShapeInferPaddingBase : public ShapeInferBase {
public:
    ShapeInferPaddingBase(std::shared_ptr<Node> node) : ShapeInferBase(std::move(node)), m_pads_begin{}, m_pads_end{} {}

    const ov::CoordinateDiff& get_pads_begin() override {
        return m_pads_begin;
    }

    const ov::CoordinateDiff& get_pads_end() override {
        return m_pads_end;
    }

protected:
    ov::CoordinateDiff m_pads_begin, m_pads_end;
};

/**
 * @brief Shape inference using tensor accessor to get constant data and padding
 *
 * @tparam TOp   Type of operator.
 * @tparam MASK  The bit mask where each bit corresponds to an input port number.
 */
template <class TOp, uint32_t MASK>
class ShapeInferPaddingTA : public ShapeInferPaddingBase {
public:
    using ShapeInferPaddingBase::ShapeInferPaddingBase;

    ov::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                 const ov::ITensorAccessor& tensor_accessor) override {
        return {shape_infer(static_cast<TOp*>(m_node.get()), input_shapes, m_pads_begin, m_pads_end, tensor_accessor)};
    }

    port_mask_t get_port_mask() const override {
        return MASK;
    }
};

/**
 * @brief Shape inference using tensor accessor to get constant data and padding
 *
 * @tparam TOp   Type of operator.
 * @tparam MASK  The bit mask where each bit corresponds to an input port number.
 */
template <class TOp>
class ShapeInferPaddingTA<TOp, 0> : public ShapeInferPaddingBase {
public:
    using ShapeInferPaddingBase::ShapeInferPaddingBase;

    ov::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                 const ov::ITensorAccessor&) override {
        return {shape_infer(static_cast<TOp*>(m_node.get()), input_shapes, m_pads_begin, m_pads_end)};
    }
};

/**
 * \brief Shape infer factory
 *
 * \tparam R     Result type of created interface object.
 * \tparam TKey  Type of Maker map key.
 * \tparam Args  TypesInference object ctor args.
 */
template <class TKey, class R, class... Args>
class ShapeInferFactory {
public:
    // Helper type to define specific Makers map values.
    using TValue = std::function<R(Args...)>;

    // Helper type to define specific Makers map type.
    using TRegistry = std::unordered_map<TKey, TValue>;

    /**
     * \brief  Creates the shape inference object.
     *
     * \param key   Key value to get specified shape inference object maker.
     * \param args  Inference object args.
     *
     * \return The shape inference object or R{} if not found in the map.
     */
    static R make(const TKey& key, Args... args) {
        const auto& maker_iter = registry.find(key);
        if (maker_iter != registry.end()) {
            return maker_iter->second(std::forward<Args>(args)...);
        } else {
            return {};
        }
    }

private:
    /** \brief Factory makers registry which can be specialized for key and value. */
    static const TRegistry registry;
};

template <template <class, IStaticShapeInfer::port_mask_t> class TShapeInfer,
          class TOp,
          IStaticShapeInfer::port_mask_t mask>
std::shared_ptr<typename TShapeInfer<TOp, mask>::iface_type> make_shape_infer(std::shared_ptr<ov::Node> node) {
    return std::make_shared<TShapeInfer<TOp, mask>>(std::move(node));
}

template <template <class> class TShapeInfer, class TOp>
std::shared_ptr<typename TShapeInfer<TOp>::iface_type> make_shape_infer(std::shared_ptr<ov::Node> node) {
    return std::make_shared<TShapeInfer<TOp>>(std::move(node));
}

template <class TShapeInfer>
std::shared_ptr<typename TShapeInfer::iface_type> make_shape_infer(std::shared_ptr<ov::Node> node) {
    return std::make_shared<TShapeInfer>(std::move(node));
}

// Type of key in shape inference Makers maps.
using ShapeInferKey = ov::NodeTypeInfo;

// Default opset used for 'default' in inference map.
using namespace ov::opset10;

// Helper macros to make map entries
#define _OV_OP_SHAPE_INFER_VA_REG(OP, ...) \
    { OP::get_type_info_static(), make_shape_infer<__VA_ARGS__> }
#define _OV_OP_SHAPE_INFER_MASK_REG(OP, SHAPE_INFER, MASK)   _OV_OP_SHAPE_INFER_VA_REG(OP, SHAPE_INFER, OP, MASK)
#define _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(OP, SHAPE_INFER) _OV_OP_SHAPE_INFER_VA_REG(OP, SHAPE_INFER)

// Helper types for IStaticShapeInfer makers.
using IStaticShapeInferFactory =
    ShapeInferFactory<ShapeInferKey, std::shared_ptr<IStaticShapeInfer>, std::shared_ptr<ov::Node>>;

// clang-format off
// Initialization map for operators supporting IStaticShapeInfer objects.
// First group in map is 'default' opset defined by alias above.
// To use other version of operators, explicitly specify operator with opset version namespace.
template <>
const IStaticShapeInferFactory::TRegistry IStaticShapeInferFactory::registry{
    // opset14
    _OV_OP_SHAPE_INFER_MASK_REG(op::v14::RMSNorm, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset14::Inverse, ShapeInferTA, util::bit::mask()),
    // opset13
    _OV_OP_SHAPE_INFER_MASK_REG(opset13::Multinomial, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset13::ScaledDotProductAttention, ShapeInferTA, util::bit::mask(3, 5)),
    // opset12
    _OV_OP_SHAPE_INFER_MASK_REG(opset12::Pad, ShapeInferTA, util::bit::mask(1, 2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset12::ScatterElementsUpdate, ShapeInferTA, util::bit::mask(3)),
    // opset11
    _OV_OP_SHAPE_INFER_MASK_REG(opset11::Interpolate, ShapeInferPaddingTA, util::bit::mask(1, 2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset11::TopK, ShapeInferTA, util::bit::mask(1)),
    // opset9
    _OV_OP_SHAPE_INFER_MASK_REG(opset9::Eye, ShapeInferTA, util::bit::mask(0, 1, 3)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset9::GridSample, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset9::IRDFT, ShapeInferTA, util::bit::mask(1, 2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset9::RDFT, ShapeInferTA, util::bit::mask(1, 2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset9::ROIAlign, ShapeInferTA, util::bit::mask()),
    // opset8
    _OV_OP_SHAPE_INFER_MASK_REG(opset8::AdaptiveAvgPool, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset8::AdaptiveMaxPool, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset8::DeformableConvolution, ShapeInferPaddingTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset8::DetectionOutput, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset8::Gather, ShapeInferTA, util::bit::mask(2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset8::GatherND, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset8::I420toBGR, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset8::I420toRGB, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset8::MatrixNms, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset8::MaxPool, ShapeInferPaddingTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset8::NV12toBGR, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset8::NV12toRGB, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset8::PriorBox, ShapeInferTA, util::bit::mask(0)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset8::RandomUniform, ShapeInferTA, util::bit::mask(0, 1, 2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset8::Slice, ShapeInferTA, util::bit::mask(1, 2, 3, 4)),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset8::Softmax, ShapeInferCopy),
    // opset7
    _OV_OP_SHAPE_INFER_MASK_REG(opset7::DFT, ShapeInferTA, util::bit::mask(1, 2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset7::Einsum, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset7::IDFT, ShapeInferTA, util::bit::mask(1, 2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset7::Roll, ShapeInferTA, util::bit::mask(2)),
    _OV_OP_SHAPE_INFER_VA_REG(opset7::Gather, ShapeInferTA, op::util::GatherBase, util::bit::mask(2)),
    // opset6
    _OV_OP_SHAPE_INFER_MASK_REG(opset6::CTCGreedyDecoderSeqLen, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset6::CTCGreedyDecoderSeqLen, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset6::ExperimentalDetectronDetectionOutput, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset6::ExperimentalDetectronGenerateProposalsSingleImage, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset6::ExperimentalDetectronPriorGridGenerator, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset6::ExperimentalDetectronROIFeatureExtractor, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset6::ExperimentalDetectronTopKROIs, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset6::GatherElements, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset6::Assign, ShapeInferCopy),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset6::MVN, ShapeInferBase),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset6::ReadValue, ShapeInferCopy),
    // opset5
    _OV_OP_SHAPE_INFER_MASK_REG(opset5::GatherND, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset5::GRUSequence, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset5::LSTMSequence, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset5::RNNSequence, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset5::BatchNormInference, ShapeInferBase),
    // opset4
    _OV_OP_SHAPE_INFER_MASK_REG(opset4::CTCLoss, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset4::Interpolate, ShapeInferPaddingTA, util::bit::mask(1, 2, 3)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset4::LSTMCell, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset4::NonMaxSuppression, ShapeInferTA, util::bit::mask(2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset4::Proposal, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset4::Range, ShapeInferTA, util::bit::mask(0, 1, 2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset4::ReduceL1, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset4::ReduceL2, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset4::ScatterNDUpdate, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset4::Swish, ShapeInferBase),
    // opset3
    _OV_OP_SHAPE_INFER_MASK_REG(opset3::Assign, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset3::Broadcast, ShapeInferTA, util::bit::mask(1, 2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset3::Bucketize, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset3::EmbeddingBagOffsetsSum, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset3::EmbeddingBagPackedSum, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset3::EmbeddingSegmentsSum, ShapeInferTA, util::bit::mask(3)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset3::ExtractImagePatches, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset3::GRUCell, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset3::NonMaxSuppression, ShapeInferTA, util::bit::mask(2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset3::ROIAlign, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset3::ScatterElementsUpdate, ShapeInferTA, util::bit::mask(3)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset3::ShapeOf, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset3::TopK, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset3::CumSum, ShapeInferBase),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset3::ReadValue, ShapeInferCopy),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset3::ScatterUpdate, ShapeInferBase),
    // opset2
    _OV_OP_SHAPE_INFER_MASK_REG(opset2::BatchToSpace, ShapeInferTA, util::bit::mask(1, 2, 3)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset2::PSROIPooling, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset2::RegionYolo, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset2::ReorgYolo, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset2::ROIPooling, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset2::SpaceToBatch, ShapeInferTA, util::bit::mask(1, 2, 3)),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset2::MVN, ShapeInferBase),
    // opset1
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::AvgPool, ShapeInferPaddingTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::BinaryConvolution, ShapeInferPaddingTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Broadcast, ShapeInferTA, util::bit::mask(1, 2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Concat, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Convolution, ShapeInferPaddingTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::ConvolutionBackpropData, ShapeInferPaddingTA, util::bit::mask(2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::CTCGreedyDecoder, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::DeformableConvolution, ShapeInferPaddingTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::DeformablePSROIPooling, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::DepthToSpace, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::DetectionOutput, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::FakeQuantize, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Gather, ShapeInferTA, util::bit::mask(2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::GatherTree, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::GroupConvolution, ShapeInferPaddingTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::GroupConvolutionBackpropData, ShapeInferPaddingTA, util::bit::mask(2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Interpolate, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::LSTMCell, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::LSTMSequence, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::MatMul, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::MaxPool, ShapeInferPaddingTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::NonMaxSuppression, ShapeInferTA, util::bit::mask(2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::OneHot, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Pad, ShapeInferTA, util::bit::mask(1, 2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::PriorBox, ShapeInferTA, util::bit::mask(0)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::PriorBoxClustered, ShapeInferTA, util::bit::mask(0)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Proposal, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Range, ShapeInferTA, util::bit::mask(0, 1, 2)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::ReduceLogicalAnd, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::ReduceLogicalOr, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::ReduceMax, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::ReduceMean, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::ReduceMin, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::ReduceProd, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::ReduceSum, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Reshape, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Reverse, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::ReverseSequence, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::RNNCell, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Select, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::ShapeOf, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::ShuffleChannels, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::SpaceToDepth, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Split, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Squeeze, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::StridedSlice, ShapeInferTA, util::bit::mask(1, 2, 3)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Tile, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::TopK, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Transpose, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::Unsqueeze, ShapeInferTA, util::bit::mask(1)),
    _OV_OP_SHAPE_INFER_MASK_REG(opset1::VariadicSplit, ShapeInferTA, util::bit::mask(1, 2)),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset1::BatchNormInference, ShapeInferBase),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset1::BatchNormInference, ShapeInferBase),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset1::Convert, ShapeInferCopy),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset1::Convert, ShapeInferCopy),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset1::Convert, ShapeInferCopy),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset1::HardSigmoid, ShapeInferBase),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset1::LogicalNot, ShapeInferCopy),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset1::LRN, ShapeInferBase),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset1::NormalizeL2, ShapeInferBase),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset1::PRelu, ShapeInferBase),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset1::Selu, ShapeInferBase),
    _OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(opset1::Softmax, ShapeInferCopy),
    //
    _OV_OP_SHAPE_INFER_MASK_REG(ov::op::internal::AUGRUCell, ShapeInferTA, util::bit::mask()),
    _OV_OP_SHAPE_INFER_MASK_REG(ov::op::internal::AUGRUSequence, ShapeInferTA, util::bit::mask()),
};
// clang-format on

#undef _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG
#undef _OV_OP_SHAPE_INFER_MASK_REG
#undef _OV_OP_SHAPE_INFER_VA_REG

std::shared_ptr<IStaticShapeInfer> make_shape_inference(std::shared_ptr<ov::Node> op) {
    if (auto shape_infer = IStaticShapeInferFactory::make(op->get_type_info(), op)) {
        return shape_infer;
    } else if (ov::is_type<op::util::UnaryElementwiseArithmetic>(op)) {
        return std::make_shared<ShapeInferCopy>(op);
    } else if (ov::is_type<op::util::BinaryElementwiseArithmetic>(op) ||
               ov::is_type<op::util::BinaryElementwiseComparison>(op) ||
               ov::is_type<op::util::BinaryElementwiseLogical>(op)) {
        return std::make_shared<ShapeInferEltwise>(op);
    } else {
        return std::make_shared<ShapeInferFallback>(op);
    }
}
}  // namespace intel_cpu
}  // namespace ov
