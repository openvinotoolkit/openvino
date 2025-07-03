// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "shape_inference.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <openvino/core/node.hpp>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// @todo try to get rid of supression
// NOLINTBEGIN(misc-include-cleaner)
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
#include "col2im_shape_inference.hpp"
#include "concat_shape_inference.hpp"
#include "convolution_backprop_shape_inference.hpp"
#include "convolution_shape_inference.hpp"
#include "copy_shape_inference.hpp"
#include "cpu_memory.h"
#include "cpu_types.h"
#include "ctc_greedy_decoder_seq_len_shape_inference.hpp"
#include "ctc_greedy_decoder_shape_inference.hpp"
#include "ctc_loss_shape_inference.hpp"
#include "custom/convolution.hpp"
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
#include "glu_shape_inference.hpp"
#include "grid_sample_shape_inference.hpp"
#include "group_convolution_backprop_shape_inference.hpp"
#include "group_convolution_shape_inference.hpp"
#include "gru_cell_shape_inference.hpp"
#include "gru_sequence_shape_inference.hpp"
#include "i420_shape_inference.hpp"
#include "interpolate_shape_inference.hpp"
#include "inverse_shape_inference.hpp"
#include "irdft_shape_inference.hpp"
#include "istft_shape_inference.hpp"
#include "lstm_cell_shape_inference.hpp"
#include "lstm_sequence_shape_inference.hpp"
#include "matmul_shape_inference.hpp"
#include "matrix_nms_shape_inference.hpp"
#include "max_pool_shape_inference.hpp"
#include "memory_accessor.hpp"
#include "multinomial_shape_inference.hpp"
#include "nms_shape_inference.hpp"
#include "nv12_shape_inference.hpp"
#include "one_hot_shape_inference.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/adaptive_avg_pool.hpp"
#include "openvino/op/adaptive_max_pool.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/avg_pool.hpp"
#include "openvino/op/batch_norm.hpp"
#include "openvino/op/batch_to_space.hpp"
#include "openvino/op/binary_convolution.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/bucketize.hpp"
#include "openvino/op/col2im.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/ctc_greedy_decoder.hpp"
#include "openvino/op/ctc_greedy_decoder_seq_len.hpp"
#include "openvino/op/ctc_loss.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/deformable_convolution.hpp"
#include "openvino/op/deformable_psroi_pooling.hpp"
#include "openvino/op/depth_to_space.hpp"
#include "openvino/op/detection_output.hpp"
#include "openvino/op/dft.hpp"
#include "openvino/op/einsum.hpp"
#include "openvino/op/embedding_segments_sum.hpp"
#include "openvino/op/embeddingbag_offsets_sum.hpp"
#include "openvino/op/embeddingbag_packed.hpp"
#include "openvino/op/embeddingbag_packedsum.hpp"
#include "openvino/op/experimental_detectron_detection_output.hpp"
#include "openvino/op/experimental_detectron_generate_proposals.hpp"
#include "openvino/op/experimental_detectron_prior_grid_generator.hpp"
#include "openvino/op/experimental_detectron_roi_feature.hpp"
#include "openvino/op/experimental_detectron_topkrois.hpp"
#include "openvino/op/extractimagepatches.hpp"
#include "openvino/op/eye.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/gather_nd.hpp"
#include "openvino/op/gather_tree.hpp"
#include "openvino/op/grid_sample.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/gru_cell.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/hard_sigmoid.hpp"
#include "openvino/op/i420_to_bgr.hpp"
#include "openvino/op/i420_to_rgb.hpp"
#include "openvino/op/idft.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/inverse.hpp"
#include "openvino/op/irdft.hpp"
#include "openvino/op/istft.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/lrn.hpp"
#include "openvino/op/lstm_cell.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/matrix_nms.hpp"
#include "openvino/op/max_pool.hpp"
#include "openvino/op/multinomial.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/non_max_suppression.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/nv12_to_bgr.hpp"
#include "openvino/op/nv12_to_rgb.hpp"
#include "openvino/op/one_hot.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/prelu.hpp"
#include "openvino/op/prior_box.hpp"
#include "openvino/op/prior_box_clustered.hpp"
#include "openvino/op/proposal.hpp"
#include "openvino/op/psroi_pooling.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/rdft.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "openvino/op/reduce_logical_or.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/region_yolo.hpp"
#include "openvino/op/reorg_yolo.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/reverse.hpp"
#include "openvino/op/reverse_sequence.hpp"
#include "openvino/op/rms_norm.hpp"
#include "openvino/op/rnn_cell.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "openvino/op/roi_align.hpp"
#include "openvino/op/roi_pooling.hpp"
#include "openvino/op/roll.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/search_sorted.hpp"
#include "openvino/op/segment_max.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/selu.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/shuffle_channels.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/slice_scatter.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/space_to_batch.hpp"
#include "openvino/op/space_to_depth.hpp"
#include "openvino/op/sparse_fill_empty_rows.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/stft.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/string_tensor_pack.hpp"
#include "openvino/op/string_tensor_unpack.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "openvino/op/util/gather_base.hpp"
#include "openvino/op/util/unary_elementwise_arithmetic.hpp"
#include "openvino/op/variadic_split.hpp"
#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"
#include "ov_ops/glu.hpp"
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
#include "search_sorted_shape_inference.hpp"
#include "segment_max_shape_inference.hpp"
#include "select_shape_inference.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_status.hpp"
#include "shape_nodes.hpp"
#include "shuffle_channels_shape_inference.hpp"
#include "slice_scatter_shape_inference.hpp"
#include "slice_shape_inference.hpp"
#include "space_to_batch_shape_inference.hpp"
#include "space_to_depth_shape_inference.hpp"
#include "sparse_fill_empty_rows_shape_inference.hpp"
#include "split_shape_inference.hpp"
#include "squeeze_shape_inference.hpp"
#include "static_shape.hpp"
#include "stft_shape_inference.hpp"
#include "strided_slice_shape_inference.hpp"
#include "string_tensor_pack_shape_inference.hpp"
#include "string_tensor_unpack_shape_inference.hpp"
#include "tensor_data_accessor.hpp"
#include "tile_shape_inference.hpp"
#include "topk_shape_inference.hpp"
#include "transpose_shape_inference.hpp"
#include "unsqueeze_shape_inference.hpp"
#include "utils.hpp"
#include "utils/bit_util.hpp"
#include "variadic_split_shape_inference.hpp"
// NOLINTEND(misc-include-cleaner)

namespace ov::intel_cpu {
/**
 * @brief Base shape inference object implementing the IStaticShapeInfer without padding support.
 *
 * Default shape inference is first input pass as output shape.
 */
class ShapeInferBase : public IStaticShapeInfer {
public:
    using iface_type = IStaticShapeInfer;

    explicit ShapeInferBase(std::shared_ptr<ov::Node> node) : m_node{std::move(node)} {
        static_assert(std::is_same_v<int64_t, Dimension::value_type>, "Rank type not match to input_ranks type.");
        for (size_t i = 0; i < m_node->get_input_size(); ++i) {
            const auto& shape = m_node->get_input_partial_shape(i);
            const auto& rank_length = shape.rank().is_static() ? shape.rank().get_length() : -1;
            m_input_ranks.push_back(rank_length);
        }
    }

    std::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                  const ov::ITensorAccessor& /*tensor_accessor*/) override {
        NODE_VALIDATION_CHECK(m_node.get(), !input_shapes.empty(), "Incorrect number of input shapes");
        return {std::vector<StaticShape>{input_shapes[0]}};
    }

    IShapeInfer::Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                              const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        const auto& input_ranks = get_input_ranks();
        const auto inputs_count = input_shapes.size();
        OPENVINO_ASSERT(input_ranks.size() <= inputs_count, "Too few input shapes passed to Shape infer.");
        std::vector<StaticShapeRef> input_static_shapes;

        input_static_shapes.reserve(inputs_count);
        for (size_t port = 0; port < input_ranks.size(); ++port) {
            input_static_shapes.push_back(input_ranks[port] == 0 ? StaticShapeRef() : input_shapes[port].get());
        }

        // call shape inference API
        auto shape_infer_result = infer(input_static_shapes, MemoryAccessor(data_dependency, input_ranks));
        return shape_infer_result ? move_shapes_to_result(*shape_infer_result) : Result{{}, ShapeInferStatus::skip};
    }

    const ov::CoordinateDiff& get_pads_begin() override {
        OPENVINO_THROW("ShapeInferBase do not support get_pads_begin() by default.");
    }

    const ov::CoordinateDiff& get_pads_end() override {
        OPENVINO_THROW("ShapeInferBase do not support get_pads_end() by default.");
    }

    const std::vector<int64_t>& get_input_ranks() override {
        return m_input_ranks;
    }

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return EMPTY_PORT_MASK;
    }

protected:
    std::vector<int64_t> m_input_ranks;
    std::shared_ptr<ov::Node> m_node;

private:
    static Result move_shapes_to_result(std::vector<StaticShape>& output_shapes) {
        Result result{decltype(Result::dims){output_shapes.size()}, ShapeInferStatus::success};
        std::transform(output_shapes.begin(), output_shapes.end(), result.dims.begin(), [](StaticShape& s) {
            return std::move(*s);
        });
        return result;
    }
};

/**
 * @brief Shape inference which copy single input shape to output shape.
 */
class ShapeInferCopy : public ShapeInferBase {
public:
    using ShapeInferBase::ShapeInferBase;

    std::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                  [[maybe_unused]] const ov::ITensorAccessor& acc) override {
        return {op::copy_shape_infer(m_node.get(), input_shapes)};
    }
};

/**
 * @brief Shape inference applied for element wise operators.
 */
class ShapeInferEltwise : public ShapeInferBase {
public:
    using ShapeInferBase::ShapeInferBase;

    std::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                  [[maybe_unused]] const ov::ITensorAccessor& acc) override {
        return {op::eltwise_shape_infer(m_node.get(), input_shapes)};
    }
};

/**
 * @brief Shape inference used as fallback if specific inference not implemented.
 */
class ShapeInferFallback : public ShapeInferBase {
public:
    using ShapeInferBase::ShapeInferBase;

    std::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                  const ov::ITensorAccessor& tensor_accessor) override {
        auto* const op = m_node.get();

        std::shared_ptr<ov::Node> local_op;
        ov::OutputVector new_inputs;
        for (size_t i = 0; i < op->get_input_size(); ++i) {
            if (auto t = tensor_accessor(i)) {
                new_inputs.emplace_back(std::make_shared<ov::op::v0::Constant>(t));
            } else if (const auto* c = ov::as_type<const op::v0::Constant>(op->get_input_node_ptr(i))) {
                new_inputs.emplace_back(c->clone_with_new_inputs(ov::OutputVector{}));
            } else {
                new_inputs.emplace_back(std::make_shared<op::v0::Parameter>(op->get_input_element_type(i),
                                                                            input_shapes[i].to_partial_shape()));
            }
        }
        local_op = op->clone_with_new_inputs(new_inputs);
        local_op->validate_and_infer_types();

        std::vector<StaticShape> output_shapes(local_op->get_output_size());
        for (size_t i = 0; i < output_shapes.size(); ++i) {
            const auto& partial_shape = local_op->get_output_partial_shape(i);

            if (partial_shape.is_dynamic()) {
                return {};
            }

            output_shapes[i] = StaticShape(partial_shape.to_shape());
        }

        return {std::move(output_shapes)};
    }

    [[nodiscard]] port_mask_t get_port_mask() const override {
        // For fallback return full port mask to try get data for all node's inputs
        return FULL_PORT_MASK;
    }
};

template <class TOp, IShapeInfer::port_mask_t MASK>
class ShapeInferTA : public ShapeInferBase {
public:
    using ShapeInferBase::ShapeInferBase;

    std::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                  const ov::ITensorAccessor& tensor_accessor) override {
        return {shape_infer(static_cast<TOp*>(m_node.get()), input_shapes, tensor_accessor)};
    }

    [[nodiscard]] port_mask_t get_port_mask() const override {
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
class ShapeInferTA<TOp, EMPTY_PORT_MASK> : public ShapeInferBase {
public:
    using ShapeInferBase::ShapeInferBase;

    std::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                  [[maybe_unused]] const ov::ITensorAccessor& acc) override {
        return {shape_infer(static_cast<TOp*>(m_node.get()), input_shapes)};
    }
};

/** @brief Base shape inference object implementing the IStaticShapeInfer with padding support. */
class ShapeInferPaddingBase : public ShapeInferBase {
public:
    explicit ShapeInferPaddingBase(std::shared_ptr<ov::Node> node) : ShapeInferBase(std::move(node)) {}

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
template <class TOp, IShapeInfer::port_mask_t MASK>
class ShapeInferPaddingTA : public ShapeInferPaddingBase {
public:
    using ShapeInferPaddingBase::ShapeInferPaddingBase;

    std::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                  const ov::ITensorAccessor& tensor_accessor) override {
        return {shape_infer(static_cast<TOp*>(m_node.get()), input_shapes, m_pads_begin, m_pads_end, tensor_accessor)};
    }

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return MASK;
    }
};

/**
 * @brief Shape inference without using tensor accessor to get constant data and padding
 *
 * @tparam TOp   Type of operator.
 * @tparam MASK  The bit mask where each bit corresponds to an input port number.
 */
template <class TOp>
class ShapeInferPaddingTA<TOp, EMPTY_PORT_MASK> : public ShapeInferPaddingBase {
public:
    using ShapeInferPaddingBase::ShapeInferPaddingBase;

    std::optional<std::vector<StaticShape>> infer(const std::vector<StaticShapeRef>& input_shapes,
                                                  [[maybe_unused]] const ov::ITensorAccessor& acc) override {
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
class ShapeInferenceFactory {
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
        }
        return {};
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

// Helper macros to make map entries
#define _OV_OP_SHAPE_INFER_VA_REG(OP, ...)                  {OP::get_type_info_static(), make_shape_infer<__VA_ARGS__>}
#define OV_OP_SHAPE_INFER_MASK_REG(OP, SHAPE_INFER, MASK)   _OV_OP_SHAPE_INFER_VA_REG(OP, SHAPE_INFER, OP, MASK)
#define OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(OP, SHAPE_INFER) _OV_OP_SHAPE_INFER_VA_REG(OP, SHAPE_INFER)

// Helper types for IStaticShapeInfer makers.
using IStaticShapeInferFactory =
    ShapeInferenceFactory<ShapeInferKey, std::shared_ptr<IStaticShapeInfer>, std::shared_ptr<ov::Node>>;

// clang-format off
// Initialization map for operators supporting IStaticShapeInfer objects.
// First group in map is 'default' opset defined by alias above.
// To use other version of operators, explicitly specify operator with opset version namespace.
template <>
const IStaticShapeInferFactory::TRegistry IStaticShapeInferFactory::registry{
    // opset16
    OV_OP_SHAPE_INFER_MASK_REG(op::v16::ISTFT, ShapeInferTA, util::bit::mask(2, 3, 4)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v16::SegmentMax, ShapeInferTA, util::bit::mask(1, 2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v16::SparseFillEmptyRows, ShapeInferTA, util::bit::mask(1, 2)),
    // opset15
    OV_OP_SHAPE_INFER_MASK_REG(op::v15::Squeeze, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v15::SearchSorted, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v15::StringTensorUnpack, ShapeInferTA, util::bit::mask(0)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v15::StringTensorPack, ShapeInferTA, util::bit::mask(0, 1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v3::EmbeddingBagOffsetsSum, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v15::EmbeddingBagPacked, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v15::Col2Im, ShapeInferTA, util::bit::mask(1, 2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v15::ScatterNDUpdate, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v15::SliceScatter, ShapeInferTA, util::bit::mask(2, 3, 4, 5)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v15::STFT, ShapeInferTA, util::bit::mask(2, 3)),
    // opset14
    OV_OP_SHAPE_INFER_MASK_REG(op::v14::Inverse, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v14::MaxPool, ShapeInferPaddingTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v14::AvgPool, ShapeInferPaddingTA, util::bit::mask()),
    // opset13
    OV_OP_SHAPE_INFER_MASK_REG(op::v13::Multinomial, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v13::ScaledDotProductAttention, ShapeInferTA, util::bit::mask(3, 5)),
    // opset12
    OV_OP_SHAPE_INFER_MASK_REG(op::v12::Pad, ShapeInferTA, util::bit::mask(1, 2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v12::ScatterElementsUpdate, ShapeInferTA, util::bit::mask(3)),
    // opset11
    OV_OP_SHAPE_INFER_MASK_REG(op::v11::Interpolate, ShapeInferPaddingTA, util::bit::mask(1, 2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v11::TopK, ShapeInferTA, util::bit::mask(1)),
    // opset9
    OV_OP_SHAPE_INFER_MASK_REG(op::v9::Eye, ShapeInferTA, util::bit::mask(0, 1, 3)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v9::GridSample, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v9::IRDFT, ShapeInferTA, util::bit::mask(1, 2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v9::RDFT, ShapeInferTA, util::bit::mask(1, 2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v9::ROIAlign, ShapeInferTA, util::bit::mask()),
    // opset8
    OV_OP_SHAPE_INFER_MASK_REG(op::v8::AdaptiveAvgPool, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v8::AdaptiveMaxPool, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v8::DeformableConvolution, ShapeInferPaddingTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v8::DetectionOutput, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v8::Gather, ShapeInferTA, util::bit::mask(2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v8::GatherND, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v8::I420toBGR, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v8::I420toRGB, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v8::MatrixNms, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v8::MaxPool, ShapeInferPaddingTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v8::NV12toBGR, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v8::NV12toRGB, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v8::PriorBox, ShapeInferTA, util::bit::mask(0)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v8::RandomUniform, ShapeInferTA, util::bit::mask(0, 1, 2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v8::Slice, ShapeInferTA, util::bit::mask(1, 2, 3, 4)),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v8::Softmax, ShapeInferCopy),
    // opset7
    OV_OP_SHAPE_INFER_MASK_REG(op::v7::DFT, ShapeInferTA, util::bit::mask(1, 2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v7::Einsum, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v7::IDFT, ShapeInferTA, util::bit::mask(1, 2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v7::Roll, ShapeInferTA, util::bit::mask(2)),
    _OV_OP_SHAPE_INFER_VA_REG(op::v7::Gather, ShapeInferTA, op::util::GatherBase, util::bit::mask(2)),
    // opset6
    OV_OP_SHAPE_INFER_MASK_REG(op::v6::CTCGreedyDecoderSeqLen, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v6::ExperimentalDetectronDetectionOutput, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v6::ExperimentalDetectronGenerateProposalsSingleImage, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v6::ExperimentalDetectronPriorGridGenerator, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v6::ExperimentalDetectronROIFeatureExtractor, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v6::ExperimentalDetectronTopKROIs, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v6::GatherElements, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v6::Assign, ShapeInferCopy),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v6::MVN, ShapeInferBase),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v6::ReadValue, ShapeInferCopy),
    // opset5
    OV_OP_SHAPE_INFER_MASK_REG(op::v5::GatherND, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v5::GRUSequence, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v5::LSTMSequence, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v5::RNNSequence, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v5::BatchNormInference, ShapeInferBase),
    // opset4
    OV_OP_SHAPE_INFER_MASK_REG(op::v4::CTCLoss, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v4::Interpolate, ShapeInferPaddingTA, util::bit::mask(1, 2, 3)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v4::LSTMCell, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v4::NonMaxSuppression, ShapeInferTA, util::bit::mask(2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v4::Proposal, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v4::Range, ShapeInferTA, util::bit::mask(0, 1, 2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v4::ReduceL1, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v4::ReduceL2, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v3::ScatterNDUpdate, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v4::Swish, ShapeInferBase),
    // opset3
    OV_OP_SHAPE_INFER_MASK_REG(op::v3::Assign, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v3::Broadcast, ShapeInferTA, util::bit::mask(1, 2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v3::Bucketize, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v3::EmbeddingBagOffsetsSum, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v3::EmbeddingBagPackedSum, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v3::EmbeddingSegmentsSum, ShapeInferTA, util::bit::mask(3)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v3::ExtractImagePatches, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v3::GRUCell, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v3::NonMaxSuppression, ShapeInferTA, util::bit::mask(2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v3::ROIAlign, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v3::ScatterElementsUpdate, ShapeInferTA, util::bit::mask(3)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v3::ShapeOf, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v3::TopK, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v0::CumSum, ShapeInferBase),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v3::ReadValue, ShapeInferCopy),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v3::ScatterUpdate, ShapeInferBase),
    // opset2
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::BatchToSpace, ShapeInferTA, util::bit::mask(1, 2, 3)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::PSROIPooling, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::RegionYolo, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::ReorgYolo, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::ROIPooling, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::SpaceToBatch, ShapeInferTA, util::bit::mask(1, 2, 3)),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v0::MVN, ShapeInferBase),
    // opset1
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::AvgPool, ShapeInferPaddingTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::BinaryConvolution, ShapeInferPaddingTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::Broadcast, ShapeInferTA, util::bit::mask(1, 2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::Concat, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::ConvolutionBackpropData, ShapeInferPaddingTA, util::bit::mask(2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::CTCGreedyDecoder, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::DeformableConvolution, ShapeInferPaddingTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::DeformablePSROIPooling, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::DepthToSpace, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::DetectionOutput, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::FakeQuantize, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::Gather, ShapeInferTA, util::bit::mask(2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::GatherTree, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::GroupConvolution, ShapeInferPaddingTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::GroupConvolutionBackpropData, ShapeInferPaddingTA, util::bit::mask(2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::Interpolate, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::LSTMCell, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::MatMul, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::MaxPool, ShapeInferPaddingTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::NonMaxSuppression, ShapeInferTA, util::bit::mask(2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::OneHot, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::Pad, ShapeInferTA, util::bit::mask(1, 2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::PriorBox, ShapeInferTA, util::bit::mask(0)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::PriorBoxClustered, ShapeInferTA, util::bit::mask(0)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::Proposal, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::Range, ShapeInferTA, util::bit::mask(0, 1, 2)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::ReduceLogicalAnd, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::ReduceLogicalOr, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::ReduceMax, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::ReduceMean, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::ReduceMin, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::ReduceProd, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::ReduceSum, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::Reshape, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::Reverse, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::ReverseSequence, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::RNNCell, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::Select, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::ShapeOf, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::ShuffleChannels, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::SpaceToDepth, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::Split, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::Squeeze, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::StridedSlice, ShapeInferTA, util::bit::mask(1, 2, 3)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::Tile, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::TopK, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::Transpose, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v0::Unsqueeze, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(op::v1::VariadicSplit, ShapeInferTA, util::bit::mask(1, 2)),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v0::BatchNormInference, ShapeInferBase),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v0::Convert, ShapeInferCopy),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v0::HardSigmoid, ShapeInferBase),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v1::LogicalNot, ShapeInferCopy),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v0::LRN, ShapeInferBase),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v0::NormalizeL2, ShapeInferBase),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v0::PRelu, ShapeInferBase),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v0::Selu, ShapeInferBase),
    OV_OP_SHAPE_INFER_NON_TEMPLATE_REG(op::v1::Softmax, ShapeInferCopy),
    //
    OV_OP_SHAPE_INFER_MASK_REG(ov::op::internal::AUGRUCell, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(ov::op::internal::AUGRUSequence, ShapeInferTA, util::bit::mask()),
    OV_OP_SHAPE_INFER_MASK_REG(ov::op::internal::RMSNorm, ShapeInferTA, util::bit::mask(1)),
    OV_OP_SHAPE_INFER_MASK_REG(ov::op::internal::GLU, ShapeInferTA, util::bit::mask()),
};
// clang-format on

#undef _OV_OP_NON_TEMPLATE_SHAPE_INFER_REG
#undef _OV_OP_SHAPE_INFER_MASK_REG
#undef _OV_OP_SHAPE_INFER_VA_REG

std::shared_ptr<IStaticShapeInfer> make_shape_inference(std::shared_ptr<ov::Node> op) {
    if (auto shape_infer = IStaticShapeInferFactory::make(op->get_type_info(), op)) {
        return shape_infer;
    }
    if (ov::is_type<op::util::UnaryElementwiseArithmetic>(op)) {
        return std::make_shared<ShapeInferCopy>(std::move(op));
    }
    if (ov::is_type_any_of<op::util::BinaryElementwiseArithmetic,
                           op::util::BinaryElementwiseComparison,
                           op::util::BinaryElementwiseLogical>(op)) {
        return std::make_shared<ShapeInferEltwise>(std::move(op));
    }
    return std::make_shared<ShapeInferFallback>(std::move(op));
}

}  // namespace ov::intel_cpu
