// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn.h"

#include <utility>

#include "common/primitive_hashing_utils.hpp"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/common/cpu_convert.h"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/input.h"
#include "nodes/reorder.h"
#include "openvino/core/parallel.hpp"
#include "openvino/op/gru_cell.hpp"
#include "openvino/op/gru_sequence.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/rnn_cell.hpp"
#include "openvino/op/rnn_sequence.hpp"
#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"
#include "shape_inference/shape_inference.hpp"
#include "transformations/utils/utils.hpp"

using namespace dnnl;

namespace ov::intel_cpu::node {

static rnn_direction ieDirection2dnnl(const std::shared_ptr<const ov::Node>& op) {
    ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::FORWARD;
    if (ov::is_type<ov::op::v5::GRUSequence>(op)) {
        direction = ov::as_type_ptr<const ov::op::v5::GRUSequence>(op)->get_direction();
    } else if (ov::is_type<ov::op::v5::LSTMSequence>(op)) {
        direction = ov::as_type_ptr<const ov::op::v5::LSTMSequence>(op)->get_direction();
    } else if (ov::is_type<ov::op::v5::RNNSequence>(op)) {
        direction = ov::as_type_ptr<const ov::op::v5::RNNSequence>(op)->get_direction();
    }
    return direction == ov::op::RecurrentSequenceDirection::FORWARD         ? rnn_direction::unidirectional_left2right
           : direction == ov::op::RecurrentSequenceDirection::REVERSE       ? rnn_direction::unidirectional_right2left
           : direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? rnn_direction::bidirectional_concat
                                                                            : rnn_direction::unidirectional_left2right;
}

static dnnl::algorithm ie2dnnl(const std::string& act_type) {
    return act_type == "sigmoid" ? dnnl::algorithm::eltwise_logistic
           : act_type == "tanh"  ? dnnl::algorithm::eltwise_tanh
           : act_type == "relu"  ? dnnl::algorithm::eltwise_relu
                                 : dnnl::algorithm::undef;
}

static dnnl::algorithm ie2dnnl(const std::shared_ptr<const ov::Node>& op) {
    if (one_of(op->get_type_info(),
               ov::op::v3::GRUCell::get_type_info_static(),
               ov::op::v5::GRUSequence::get_type_info_static())) {
        auto gruCellOp = ov::as_type_ptr<const ov::op::v3::GRUCell>(op);
        auto gruSeqOp = ov::as_type_ptr<const ov::op::v5::GRUSequence>(op);
        if ((gruCellOp && gruCellOp->get_linear_before_reset()) || (gruSeqOp && gruSeqOp->get_linear_before_reset())) {
            return dnnl::algorithm::lbr_gru;
        }
        return dnnl::algorithm::vanilla_gru;
    }
    if (one_of(op->get_type_info(),
               ov::op::internal::AUGRUCell::get_type_info_static(),
               ov::op::internal::AUGRUSequence::get_type_info_static())) {
        auto gruCellOp = ov::as_type_ptr<const ov::op::internal::AUGRUCell>(op);
        auto gruSeqOp = ov::as_type_ptr<const ov::op::internal::AUGRUSequence>(op);
        if ((gruCellOp && gruCellOp->get_linear_before_reset()) || (gruSeqOp && gruSeqOp->get_linear_before_reset())) {
            return dnnl::algorithm::lbr_augru;
        }
        return dnnl::algorithm::vanilla_augru;
    }
    if (one_of(op->get_type_info(),
               ov::op::v0::LSTMCell::get_type_info_static(),
               ov::op::v4::LSTMCell::get_type_info_static(),
               ov::op::v5::LSTMSequence::get_type_info_static())) {
        return dnnl::algorithm::vanilla_lstm;
    }
    if (one_of(op->get_type_info(),
               ov::op::v0::RNNCell::get_type_info_static(),
               ov::op::v5::RNNSequence::get_type_info_static())) {
        return dnnl::algorithm::vanilla_rnn;
    }
    OPENVINO_THROW("Operation ",
                   op->get_type_name(),
                   " with name '",
                   op->get_friendly_name(),
                   "' has unsupported cell type.");
}

inline size_t gatesCount(const algorithm& alg) {
    switch (alg) {
    case algorithm::vanilla_rnn:
        return 1;
    case algorithm::vanilla_gru:
    case algorithm::vanilla_augru:
    case algorithm::lbr_augru:
    case algorithm::lbr_gru:
        return 3;
    case algorithm::vanilla_lstm:
        return 4;
    default:
        OPENVINO_THROW("Unsupported cell type");
        return 0;
    }
}

inline size_t statesCount(const dnnl::algorithm& alg) {
    switch (alg) {
    case dnnl::algorithm::vanilla_rnn:
    case dnnl::algorithm::vanilla_gru:
    case dnnl::algorithm::vanilla_augru:
    case dnnl::algorithm::lbr_augru:
    case dnnl::algorithm::lbr_gru:
        return 1;
    case dnnl::algorithm::vanilla_lstm:
        return 2;
    default:
        OPENVINO_THROW("Unsupported cell type");
        return 0;
    }
}

inline bool haveCellState(const dnnl::algorithm& alg) {
    return alg == dnnl::algorithm::vanilla_lstm;
}
inline bool haveAttention(const dnnl::algorithm& alg) {
    return alg == dnnl::algorithm::vanilla_augru || alg == dnnl::algorithm::lbr_augru;
}

// what weight data type should be used for particular input data type
const std::map<memory::data_type, memory::data_type> RNN::weightsByinputDataType{
    // layer data type        weights data type
    {memory::data_type::f32, memory::data_type::f32},
    {memory::data_type::f16, memory::data_type::f16},
    {memory::data_type::bf16, memory::data_type::bf16},
    {memory::data_type::u8, memory::data_type::s8},
    {memory::data_type::s8, memory::data_type::s8},
};

struct RNNKey {
    const std::vector<DnnlBlockedMemoryDescPtr> inDataDescs;
    const std::vector<DnnlBlockedMemoryDescPtr> outDataDescs;
    const std::vector<dnnl::memory::desc> wDescs;
    dnnl::algorithm cellType;
    dnnl::algorithm cellAct;
    dnnl::rnn_direction direction;
    dnnl::primitive_attr attr;
    [[nodiscard]] size_t hash() const;
    bool operator==(const RNNKey& rhs) const;
};

size_t RNNKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0lu;

    for (auto& desc : inDataDescs) {
        if (desc != nullptr) {
            seed = hash_combine(seed, get_md_hash(*desc->getDnnlDesc().get()));
        }
    }
    for (auto& desc : outDataDescs) {
        if (desc != nullptr) {
            seed = hash_combine(seed, get_md_hash(*desc->getDnnlDesc().get()));
        }
    }
    for (auto& desc : wDescs) {
        seed = hash_combine(seed, get_md_hash(*desc.get()));
    }
    seed = hash_combine(seed, cellType);
    seed = hash_combine(seed, cellAct);
    seed = hash_combine(seed, direction);
    return seed;
}

bool RNNKey::operator==(const RNNKey& rhs) const {
    if (inDataDescs.size() != rhs.inDataDescs.size() || outDataDescs.size() != rhs.outDataDescs.size() ||
        wDescs.size() != rhs.wDescs.size() || cellType != rhs.cellType || cellAct != rhs.cellAct ||
        direction != rhs.direction) {
        return false;
    }

    for (size_t i = 0lu; i < inDataDescs.size(); i++) {
        if (inDataDescs[i] != rhs.inDataDescs[i] &&
            (inDataDescs[i] == nullptr || rhs.inDataDescs[i] == nullptr ||
             inDataDescs[i]->getDnnlDesc() != rhs.inDataDescs[i]->getDnnlDesc())) {
            return false;
        }
    }
    for (size_t i = 0lu; i < outDataDescs.size(); i++) {
        if (outDataDescs[i] != rhs.outDataDescs[i] &&
            (outDataDescs[i] == nullptr || rhs.outDataDescs[i] == nullptr ||
             outDataDescs[i]->getDnnlDesc() != rhs.outDataDescs[i]->getDnnlDesc())) {
            return false;
        }
    }
    for (size_t i = 0lu; i < wDescs.size(); i++) {
        if (wDescs[i] != rhs.wDescs[i]) {
            return false;
        }
    }

    return true;
}

bool RNN::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                    ov::op::v3::GRUCell::get_type_info_static(),
                    ov::op::internal::AUGRUCell::get_type_info_static(),
                    ov::op::internal::AUGRUSequence::get_type_info_static(),
                    ov::op::v0::LSTMCell::get_type_info_static(),
                    ov::op::v4::LSTMCell::get_type_info_static(),
                    ov::op::v0::RNNCell::get_type_info_static(),
                    ov::op::v5::GRUSequence::get_type_info_static(),
                    ov::op::v5::LSTMSequence::get_type_info_static(),
                    ov::op::v5::RNNSequence::get_type_info_static())) {
            errorMessage = "Unsupported sequence operation.";
            return false;
        }

        if (one_of(op->get_type_info(),
                   ov::op::v0::RNNCell::get_type_info_static(),
                   ov::op::v3::GRUCell::get_type_info_static())) {
            // Plug-in does not support dynamism on weights.
            if (!ov::op::util::is_on_constant_path(op->input_value(2)) ||
                !ov::op::util::is_on_constant_path(op->input_value(3)) ||
                (op->get_input_size() > 4 && !ov::op::util::is_on_constant_path(op->input_value(4)))) {
                errorMessage = "Node expects constants as W, R, B inputs.";
                return false;
            }
        } else if (one_of(op->get_type_info(),
                          ov::op::v0::LSTMCell::get_type_info_static(),
                          ov::op::v4::LSTMCell::get_type_info_static(),
                          ov::op::v5::GRUSequence::get_type_info_static(),
                          ov::op::v5::RNNSequence::get_type_info_static())) {
            // Plug-in does not support dynamism on weights.
            if (!ov::op::util::is_on_constant_path(op->input_value(3)) ||
                !ov::op::util::is_on_constant_path(op->input_value(4)) ||
                (op->get_input_size() > 5 && !ov::op::util::is_on_constant_path(op->input_value(5)))) {
                errorMessage = "Node expects constants as W, R, B inputs.";
                return false;
            }
            if (ov::is_type<ov::op::v0::LSTMCell>(op) && op->get_input_size() != 6) {
                errorMessage = "Node expects 6 inputs. Actual: " + std::to_string(op->get_input_size());
                return false;
            }
        } else if (one_of(op->get_type_info(), ov::op::v5::LSTMSequence::get_type_info_static())) {
            if (op->get_input_size() != 7) {
                errorMessage = "Node expects 7 inputs. Actual: " + std::to_string(op->get_input_size());
                return false;
            }
            // Plug-in does not support dynamism on weights.
            if (!ov::op::util::is_on_constant_path(op->input_value(4)) ||
                !ov::op::util::is_on_constant_path(op->input_value(5)) ||
                !ov::op::util::is_on_constant_path(op->input_value(6))) {
                errorMessage = "Node expects static shaped W, R, B inputs.";
                return false;
            }
        }

        auto rnnCellBase = ov::as_type_ptr<const ov::op::util::RNNCellBase>(op);
        if (rnnCellBase) {
            if (rnnCellBase->get_clip() != 0.f) {
                errorMessage = "Clipping is not supported for RNN primitive.";
                return false;
            }
            if (one_of(rnnCellBase->get_type_info(),
                       ov::op::v0::LSTMCell::get_type_info_static(),
                       ov::op::v4::LSTMCell::get_type_info_static(),
                       ov::op::v5::LSTMSequence::get_type_info_static())) {
                if (rnnCellBase->get_activations() != std::vector<std::string>{"sigmoid", "tanh", "tanh"}) {
                    errorMessage = "Not supported activation functions";
                    return false;
                }
            } else if (one_of(rnnCellBase->get_type_info(),
                              ov::op::v3::GRUCell::get_type_info_static(),
                              ov::op::v5::GRUSequence::get_type_info_static(),
                              ov::op::internal::AUGRUCell::get_type_info_static(),
                              ov::op::internal::AUGRUSequence::get_type_info_static())) {
                if (rnnCellBase->get_activations() != std::vector<std::string>{"sigmoid", "tanh"}) {
                    errorMessage = "Not supported activation functions";
                    return false;
                }
            } else if (one_of(rnnCellBase->get_type_info(),
                              ov::op::v5::RNNSequence::get_type_info_static(),
                              ov::op::v0::RNNCell::get_type_info_static())) {
                if (rnnCellBase->get_activations().empty() ||
                    !one_of(rnnCellBase->get_activations().front(), "sigmoid", "tanh", "relu")) {
                    errorMessage = "Not supported activation functions";
                    return false;
                }
            }
        }

        ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::FORWARD;
        int64_t seqLenIdx = -1;
        if (auto gru_seq = ov::as_type_ptr<const ov::op::v5::GRUSequence>(op)) {
            direction = gru_seq->get_direction();
            seqLenIdx = 2;
        } else if (auto lstm_seq = ov::as_type_ptr<const ov::op::v5::LSTMSequence>(op)) {
            direction = lstm_seq->get_direction();
            seqLenIdx = 3;
        } else if (auto augru_seq = ov::as_type_ptr<const ov::op::internal::AUGRUSequence>(op)) {
            direction = augru_seq->get_direction();
            seqLenIdx = 2;
        } else if (auto rnn_seq = ov::as_type_ptr<const ov::op::v5::RNNSequence>(op)) {
            direction = rnn_seq->get_direction();
            seqLenIdx = 2;
        }

        if (!one_of(direction,
                    ov::op::RecurrentSequenceDirection::FORWARD,
                    ov::op::RecurrentSequenceDirection::REVERSE)) {
            errorMessage = "Unsupported sequence direction.";
            return false;
        }

        if (seqLenIdx > 0) {
            const auto& data_pshape = op->get_input_partial_shape(0);

            // WA: dynamic shapes make impossible to check seq_len due to shapeOf subgraphs
            // but the sequence is still supported in CPU and doesn't need to be decomposed
            if (data_pshape.is_dynamic()) {
                return true;
            }

            const int64_t maxSeqLenDimIdx = 1;

            if (data_pshape.rank().is_static() && data_pshape.rank().get_length() > maxSeqLenDimIdx &&
                !data_pshape[maxSeqLenDimIdx].is_static()) {
                errorMessage = "Max sequence length dimension is dynamic";
                return false;
            }

            if (ov::op::util::is_seq_len_provided(op->get_input_node_shared_ptr(0),
                                                  op->get_input_node_shared_ptr(seqLenIdx))) {
                errorMessage = "Unsupported sequence length.";
                return false;
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

bool RNN::isCell(const std::shared_ptr<const ov::Node>& op) {
    return one_of(op->get_type_info(),
                  ov::op::v0::RNNCell::get_type_info_static(),
                  ov::op::v3::GRUCell::get_type_info_static(),
                  ov::op::internal::AUGRUCell::get_type_info_static(),
                  ov::op::v0::LSTMCell::get_type_info_static(),
                  ov::op::v4::LSTMCell::get_type_info_static());
}

bool RNN::testNativeOrder(const std::shared_ptr<const ov::Node>& op) {
    if (isCell(op)) {
        return true;
    }
    const auto& rtInfo = op->get_rt_info();
    if (rtInfo.count("seqAxis")) {
        return rtInfo.at("seqAxis").as<int64_t>() == 0;
    }
    return false;
}

namespace {
/**
 * Extends Rnn ngraph shape inference implementation. The main purpose of this class is to do the trick with
 * dimentions permutation, necessary due to the mismatch between the ngrpah and the oneDNN RNN node descriptions.
 *
 */
class RnnShapeInfer : public IShapeInfer {
public:
    RnnShapeInfer(std::shared_ptr<ov::Node> op)
        : is_sequence(!(RNN::isCell(op))),
          native_order(RNN::testNativeOrder(op)),
          m_shape_infer(make_shape_inference(std::move(op))) {}

    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        auto result = m_shape_infer->infer(input_shapes, data_dependency);
        if (ShapeInferStatus::success != result.status) {
            OPENVINO_THROW("Unexpected: Unexpected shape inference result status");
        }

        auto& originOutputShapes = result.dims;

        // Graph optimizer makes the same optimization. So this is required to make shapes compatible.
        if (is_sequence && !native_order && originOutputShapes[0].size() == 4lu && originOutputShapes[0][1] == 1lu) {
            originOutputShapes[0].erase(originOutputShapes[0].begin() + 1);
        }
        return {std::move(originOutputShapes), result.status};
    }

    const ov::CoordinateDiff& get_pads_begin() override {
        return m_shape_infer->get_pads_begin();
    }

    const ov::CoordinateDiff& get_pads_end() override {
        return m_shape_infer->get_pads_end();
    }

    [[nodiscard]] port_mask_t get_port_mask() const override {
        return m_shape_infer->get_port_mask();
    }

private:
    bool is_sequence;
    bool native_order;
    ShapeInferPtr m_shape_infer;
};

class RnnShapeInferFactory final : public ShapeInferFactory {
public:
    RnnShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<RnnShapeInfer>(m_op);
    }

private:
    std::shared_ptr<ov::Node> m_op;
};

}  // namespace

RNN::RNN(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context)
    : Node(op, context, RnnShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        OPENVINO_THROW_NOT_IMPLEMENTED(errorMessage);
    }

    is_augru = one_of(op->get_type_info(),
                      ov::op::internal::AUGRUCell::get_type_info_static(),
                      ov::op::internal::AUGRUSequence::get_type_info_static());

    is_cell = isCell(op);

    if (one_of(op->get_type_info(),
               ov::op::v0::RNNCell::get_type_info_static(),
               ov::op::v3::GRUCell::get_type_info_static())) {
        wIdx = 2;
        rIdx = 3;
        bIdx = 4;
        hoIdx = 0;
    } else if (op->get_type_info() == ov::op::internal::AUGRUCell::get_type_info_static()) {
        wIdx = 2;
        rIdx = 3;
        bIdx = 4;
        aIdx = 5;
    } else if (one_of(op->get_type_info(),
                      ov::op::v0::LSTMCell::get_type_info_static(),
                      ov::op::v4::LSTMCell::get_type_info_static())) {
        wIdx = 3;
        rIdx = 4;
        bIdx = 5;
        yIdx = hoIdx = 0;
        coIdx = 1;
    } else if (one_of(op->get_type_info(),
                      ov::op::v5::RNNSequence::get_type_info_static(),
                      ov::op::v5::GRUSequence::get_type_info_static())) {
        sIdx = 2;
        wIdx = 3;
        rIdx = 4;
        bIdx = 5;
        yIdx = 0;
        hoIdx = 1;
    } else if (op->get_type_info() == ov::op::internal::AUGRUSequence::get_type_info_static()) {
        sIdx = 2;
        wIdx = 3;
        rIdx = 4;
        bIdx = 5;
        aIdx = 6;
        yIdx = 0;
        hoIdx = 1;
    } else if (one_of(op->get_type_info(), ov::op::v5::LSTMSequence::get_type_info_static())) {
        sIdx = 3;
        wIdx = 4;
        rIdx = 5;
        bIdx = 6;
        yIdx = 0;
        hoIdx = 1;
        coIdx = 2;
    }

    auto rnnCellBase = ov::as_type_ptr<ov::op::util::RNNCellBase>(op);
    if (!rnnCellBase) {
        THROW_CPU_NODE_ERR("does not have original layer for RNNCell.");
    }

    cell_type = ie2dnnl(op);
    if (!rnnCellBase->get_activations().empty()) {
        cell_act = ie2dnnl(rnnCellBase->get_activations()[0]);  // Works only for RNN with one gate
    }

    G = gatesCount(cell_type);
    Gb = (cell_type != dnnl::algorithm::lbr_gru) ? G : G + 1;
    S = statesCount(cell_type);
    SC = rnnCellBase->get_hidden_size();
    N = {getInputShapeAtPort(0).getMinDims()[0], getInputShapeAtPort(0).getMaxDims()[0]};
    if (!is_cell) {
        N_SEQ = {getInputShapeAtPort(sIdx).getMinDims()[0], getInputShapeAtPort(sIdx).getMaxDims()[0]};
    }

    const auto& rtInfo = op->get_rt_info();

    if (rtInfo.count("inputScale")) {
        inputScale = rtInfo.at("inputScale").as<float>();
    }

    if (rtInfo.count("inputShift")) {
        inputShift = rtInfo.at("inputShift").as<float>();
    }

    if (rtInfo.count("weightsScales")) {
        weightsScales = rtInfo.at("weightsScales").as<std::vector<float>>();
    }

    if (is_cell) {
        initCell();
    } else {
        direction = ieDirection2dnnl(op);

        nativeOrder = testNativeOrder(op);

        initSequence();
    }

    inDataTypes.resize(getOriginalInputsNumber());
    outDataTypes.resize(getOriginalOutputsNumber());
}

bool RNN::created() const {
    return getType() == (is_cell ? Type::RNNCell : Type::RNNSeq);
}

void RNN::configurePortDataTypes() {
    inDataTypes[xIdx] = DnnlExtensionUtils::ElementTypeToDataType(getOriginalInputPrecisionAtPort(0));
    inDataTypes[hIdx] = DnnlExtensionUtils::ElementTypeToDataType(getOriginalInputPrecisionAtPort(1));
    if (haveCellState(cell_type)) {
        inDataTypes[cIdx] = memory::data_type::f32;  // @todo bf16 is also allowed, should be tried out
    }
    if (!is_cell) {
        inDataTypes[sIdx] = memory::data_type::s32;
    }
    inDataTypes[wIdx] = DnnlExtensionUtils::ElementTypeToDataType(getOriginalInputPrecisionAtPort(wIdx));
    inDataTypes[rIdx] = DnnlExtensionUtils::ElementTypeToDataType(getOriginalInputPrecisionAtPort(rIdx));

    inDataTypes[bIdx] = memory::data_type::f32;  // @todo bf16 is also allowed, should be tried out
    if (haveAttention(cell_type)) {
        inDataTypes[aIdx] = DnnlExtensionUtils::ElementTypeToDataType(getOriginalInputPrecisionAtPort(aIdx));
    }

    if (!is_cell) {
        outDataTypes[yIdx] = DnnlExtensionUtils::ElementTypeToDataType(getOriginalOutputPrecisionAtPort(0));
    }

    outDataTypes[hoIdx] =
        inDataTypes[hIdx];  // required by oneDNN. Output hidden state is a input hidden state for the next iteration

    if (haveCellState(cell_type)) {
        outDataTypes[coIdx] = inDataTypes[cIdx];  // required by oneDNN.
    }

    if (one_of(memory::data_type::bf16, inDataTypes[xIdx], inDataTypes[hIdx])) {
        inDataTypes[xIdx] = outDataTypes[yIdx] = outDataTypes[hoIdx] = inDataTypes[hIdx] =
            memory::data_type::bf16;  // required by oneDNN.
    }

    if (one_of(memory::data_type::f16, inDataTypes[xIdx], inDataTypes[hIdx])) {
        // onednn doesn't have fp16 instance
        inDataTypes[xIdx] = outDataTypes[yIdx] = outDataTypes[hoIdx] = inDataTypes[hIdx] =
            memory::data_type::f32;  // required by oneDNN.
    }

    // OneDNN unsupported fp16 precision for this layer
    if (cell_type == dnnl::algorithm::vanilla_augru && inDataTypes[aIdx] == memory::data_type::f16) {
        inDataTypes[aIdx] = memory::data_type::f32;
    }

    if (outDataTypes[yIdx] == memory::data_type::bf16 &&
        one_of(inDataTypes[xIdx], memory::data_type::s8, memory::data_type::u8)) {
        outDataTypes[yIdx] =
            memory::data_type::f32;  // oneDNN does not support bf16 output precision for quantized rnn primitive yet
    }
}

void RNN::getSupportedDescriptors() {
    configurePortDataTypes();

    if (is_cell) {
        fillCellDesc();
    } else {
        fillSequenceDesc();
    }
}

void RNN::initCell() {
    if (getInputShapeAtPort(0).getRank() != 2lu || getInputShapeAtPort(1).getRank() != 2lu) {
        THROW_CPU_NODE_ERR("has incorrect input ranks. Data rank: ",
                           getInputShapeAtPort(0).getRank(),
                           "; Hidden state rank: ",
                           getInputShapeAtPort(1).getRank());
    }
    if (is_augru && getInputShapeAtPort(5).getRank() != 2lu) {
        THROW_CPU_NODE_ERR("has incorrect input ranks. Attention rank: ", getInputShapeAtPort(2).getRank());
    }

    T = {1, 1};
    if (cell_type == algorithm::vanilla_lstm) {
        DC = getInputShapeAtPort(3).getDims()[1];
    } else {
        DC = getInputShapeAtPort(2).getDims()[1];
    }

    if (N.isStatic()) {
        // Expected shapes.
        const auto B = N.minVal;
        const Shape shapeD{B, DC}, shapeS{B, SC};

        if ((getInputShapeAtPort(0).isStatic() && getInputShapeAtPort(0) != shapeD) ||
            (getInputShapeAtPort(1).isStatic() && getInputShapeAtPort(1) != shapeS) ||
            (getOutputShapeAtPort(0).isStatic() && getOutputShapeAtPort(0) != shapeS)) {
            THROW_CPU_NODE_ERR("has incorrect input/output shapes. Data shape: ",
                               getInputShapeAtPort(0).toString(),
                               "; Hidden state input: ",
                               getInputShapeAtPort(1).toString(),
                               "; Hidden state output: ",
                               getOutputShapeAtPort(0).toString());
        }

        if (S == 2) {
            if ((getInputShapeAtPort(2).isStatic() && getInputShapeAtPort(2) != shapeS) ||
                (getOutputShapeAtPort(1).isStatic() && getOutputShapeAtPort(1) != shapeS)) {
                THROW_CPU_NODE_ERR("has incorrect input/output shapes. Cell state input: ",
                                   getInputShapeAtPort(2).toString(),
                                   "; Cell state output: ",
                                   getOutputShapeAtPort(1).toString());
            }
        }

        if (is_augru) {
            const Shape shapeA{B, 1};
            if (getInputShapeAtPort(5).isStatic() && getInputShapeAtPort(5) != shapeA) {
                THROW_CPU_NODE_ERR("has incorrect input shapes. Attention shape: ", getInputShapeAtPort(5).toString());
            }
        }
    }
}

void RNN::fillCellDesc() {
    const Shape shapeS_4D = MemoryDescUtils::makeDummyShape({{L, D, N.minVal, SC}, {L, D, N.maxVal, SC}});
    const Shape inShape = MemoryDescUtils::makeDummyShape({{T.minVal, N.minVal, DC}, {T.maxVal, N.maxVal, DC}});
    const Shape outShape =
        MemoryDescUtils::makeDummyShape({{T.minVal, N.minVal, D * SC}, {T.maxVal, N.maxVal, D * SC}});

    // layer input plus states
    if (haveAttention(cell_type)) {
        inDataDescs.reserve(S + 2);
    } else {
        inDataDescs.reserve(S + 1);
    }
    outDataDescs.reserve(S + 1);

    // @todo use indexies instead of emplacing back, since order matters
    inDataDescs.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(inShape, inDataTypes[xIdx], memory::format_tag::tnc));
    outDataDescs.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(outShape, outDataTypes[yIdx], memory::format_tag::tnc));

    inDataDescs.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, inDataTypes[hIdx], memory::format_tag::ldnc));
    outDataDescs.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, outDataTypes[hoIdx], memory::format_tag::ldnc));

    if (haveCellState(cell_type)) {
        inDataDescs.emplace_back(
            std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, inDataTypes[cIdx], memory::format_tag::ldnc));
        outDataDescs.emplace_back(
            std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, outDataTypes[coIdx], memory::format_tag::ldnc));
    } else if (haveAttention(cell_type)) {
        const Shape attnShape = MemoryDescUtils::makeDummyShape({{T.minVal, N.minVal, 1}, {T.maxVal, N.maxVal, 1}});
        inDataDescs.emplace_back(
            std::make_shared<DnnlBlockedMemoryDesc>(attnShape, inDataTypes[aIdx], memory::format_tag::tnc));
    }

    copyWeightsData();

    // Expected shapes.
    const Shape shapeD{{N.minVal, DC}, {N.maxVal, DC}};
    const Shape shapeS{{N.minVal, SC}, {N.maxVal, SC}};
    const Shape WShape{SC * G, DC};
    const Shape RShape{SC * G, SC};
    const Shape BShape{SC * Gb};

    std::vector<MemoryDescPtr> inCandidate, outCandidate;

    inCandidate.reserve(getOriginalInputsNumber());
    outCandidate.reserve(getOriginalOutputsNumber());

    inCandidate.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(shapeD, inDataTypes[xIdx], memory::format_tag::nc));

    inCandidate.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(shapeS, inDataTypes[hIdx], memory::format_tag::nc));
    outCandidate.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(shapeS, outDataTypes[hoIdx], memory::format_tag::nc));

    if (haveCellState(cell_type)) {
        inCandidate.emplace_back(
            std::make_shared<DnnlBlockedMemoryDesc>(shapeS, inDataTypes[cIdx], memory::format_tag::nc));
        outCandidate.emplace_back(
            std::make_shared<DnnlBlockedMemoryDesc>(shapeS, outDataTypes[coIdx], memory::format_tag::nc));
    }
    // The weight and weights_iter would expose nc layout to avoid unnecessary reorder.
    // The onednn would determine the final layout when prepareParams.
    inCandidate.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(WShape, inDataTypes[wIdx], memory::format_tag::nc));
    inCandidate.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(RShape, inDataTypes[rIdx], memory::format_tag::nc));

    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(BShape, inDataTypes[bIdx], memory::format_tag::x));

    if (haveAttention(cell_type)) {
        Shape shapeAttn{{N.minVal, 1}, {N.maxVal, 1}};
        inCandidate.emplace_back(
            std::make_shared<DnnlBlockedMemoryDesc>(shapeAttn, inDataTypes[aIdx], memory::format_tag::nc));
    }

    createDescriptor(inCandidate, outCandidate);
}

void RNN::initSequence() {
    const auto& inDataShape = getInputShapeAtPort(0);
    const auto& outDataShape = getOutputShapeAtPort(0);

    if (inDataShape.getRank() != 3lu || outDataShape.getRank() != 4lu) {
        THROW_CPU_NODE_ERR("has incorrect input/output shapes. Input data shape: ",
                           inDataShape.toString(),
                           " Output shape: ",
                           outDataShape.toString());
    }

    if (!one_of(getOriginalInputsNumber(), 6u, 7u)) {
        THROW_CPU_NODE_ERR("has incorrect number of input ports: ", getOriginalInputsNumber());
    }
    if (!one_of(getOriginalOutputsNumber(), 2u, 3u)) {
        THROW_CPU_NODE_ERR("has incorrect number of output ports: ", getOriginalOutputsNumber());
    }

    T = {inDataShape.getMinDims()[1], inDataShape.getMaxDims()[1]};
    if (cell_type == algorithm::vanilla_lstm) {
        DC = getInputShapeAtPort(4).getDims()[2];
    } else {
        DC = getInputShapeAtPort(3).getDims()[2];
    }

    // layer input plus states
    if (haveAttention(cell_type)) {
        inDataDescs.reserve(S + 2);
    } else {
        inDataDescs.reserve(S + 1);
    }

    outDataDescs.reserve(S + 1);
}

void RNN::fillSequenceDesc() {
    const Shape shapeS_4D = MemoryDescUtils::makeDummyShape({{L, D, N.minVal, SC}, {L, D, N.maxVal, SC}});
    const Shape inShape = MemoryDescUtils::makeDummyShape({{T.minVal, N.minVal, DC}, {T.maxVal, N.maxVal, DC}});
    const Shape outShape =
        MemoryDescUtils::makeDummyShape({{T.minVal, N.minVal, D * SC}, {T.maxVal, N.maxVal, D * SC}});

    // Try to create descriptor and corresponding configuration
    inDataDescs.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(inShape, inDataTypes[xIdx], memory::format_tag::tnc));
    outDataDescs.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(outShape, outDataTypes[yIdx], memory::format_tag::tnc));

    inDataDescs.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, inDataTypes[hIdx], memory::format_tag::ldnc));
    outDataDescs.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, outDataTypes[hoIdx], memory::format_tag::ldnc));

    if (haveCellState(cell_type)) {
        inDataDescs.emplace_back(
            std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, inDataTypes[cIdx], memory::format_tag::ldnc));
        outDataDescs.emplace_back(
            std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, outDataTypes[coIdx], memory::format_tag::ldnc));
    } else if (haveAttention(cell_type)) {
        const Shape attnShape = MemoryDescUtils::makeDummyShape({{T.minVal, N.minVal, 1}, {T.maxVal, N.maxVal, 1}});
        inDataDescs.emplace_back(
            std::make_shared<DnnlBlockedMemoryDesc>(attnShape, inDataTypes[aIdx], memory::format_tag::tnc));
    }

    copyWeightsData();

    const Shape shapeNDSC{{N.minVal, D, SC}, {N.maxVal, D, SC}};
    Shape shapeNTSC{{N.minVal, T.minVal, SC}, {N.maxVal, T.maxVal, SC}};
    const Shape shapeNTDC{{N.minVal, T.minVal, DC}, {N.maxVal, T.maxVal, DC}};
    const Shape TShape{VectorDims{N_SEQ.minVal}, VectorDims{N_SEQ.maxVal}};
    const Shape WShape{D, G * SC, DC};
    const Shape RShape{D, G * SC, SC};
    const Shape BShape{D, Gb * SC};

    std::vector<MemoryDescPtr> inCandidate, outCandidate;

    inCandidate.reserve(getOriginalInputsNumber());
    outCandidate.reserve(getOriginalOutputsNumber());

    auto srcLayerMemoryFormat = memory::format_tag::undef;
    auto dstLayerMemoryFormat = memory::format_tag::undef;

    if (nativeOrder) {
        srcLayerMemoryFormat = memory::format_tag::tnc;
        dstLayerMemoryFormat = memory::format_tag::abcd;
        shapeNTSC = {{N.minVal, D, T.minVal, SC}, {N.maxVal, D, T.maxVal, SC}};
    } else if (N.isStatic() && N.maxVal == 1) {
        srcLayerMemoryFormat = memory::format_tag::tnc;
        dstLayerMemoryFormat = memory::format_tag::tnc;
    } else {
        srcLayerMemoryFormat = memory::format_tag::ntc;
        dstLayerMemoryFormat = memory::format_tag::ntc;
    }

    inCandidate.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(shapeNTDC, inDataTypes[xIdx], srcLayerMemoryFormat));
    outCandidate.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(shapeNTSC, outDataTypes[yIdx], dstLayerMemoryFormat));

    inCandidate.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(shapeNDSC, inDataTypes[hIdx], memory::format_tag::tnc));
    outCandidate.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(shapeNDSC, outDataTypes[hoIdx], memory::format_tag::tnc));

    // initial cell state
    if (haveCellState(cell_type)) {
        inCandidate.emplace_back(
            std::make_shared<DnnlBlockedMemoryDesc>(shapeNDSC, inDataTypes[cIdx], memory::format_tag::tnc));
        outCandidate.emplace_back(
            std::make_shared<DnnlBlockedMemoryDesc>(shapeNDSC, outDataTypes[coIdx], memory::format_tag::tnc));
    }

    inCandidate.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(TShape, inDataTypes[sIdx], memory::format_tag::x));  // sequence lengths
    // The weight and weights_iter would expose tnc layout to avoid unnecessary reorder.
    // The onednn would determine the final layout when prepareParams.
    inCandidate.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(WShape, inDataTypes[wIdx], memory::format_tag::tnc));  // W
    inCandidate.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(RShape, inDataTypes[rIdx], memory::format_tag::tnc));  // R

    inCandidate.emplace_back(
        std::make_shared<DnnlBlockedMemoryDesc>(BShape, inDataTypes[bIdx], memory::format_tag::nc));  // B

    if (haveAttention(cell_type)) {
        Shape shapeAttn{{N.minVal, T.minVal, 1}, {N.maxVal, T.maxVal, 1}};
        inCandidate.emplace_back(
            std::make_shared<DnnlBlockedMemoryDesc>(shapeAttn, inDataTypes[aIdx], memory::format_tag::ntc));
    }

    createDescriptor(inCandidate, outCandidate);
}

template <element::Type_t ET>
void RNN::fillWeights() {
    using DataType = typename element_type_traits<ET>::value_type;
    if (getParentEdgeAt(wIdx)->getParent()->getType() != Type::Input) {
        THROW_CPU_NODE_ERR("expects Constant for port ", wIdx);
    }
    auto w_const_blob = static_cast<Input*>(getParentEdgeAt(wIdx)->getParent().get())->getMemoryPtr();
    if (getParentEdgeAt(rIdx)->getParent()->getType() != Type::Input) {
        THROW_CPU_NODE_ERR("expects Constant for port ", rIdx);
    }
    auto r_const_blob = static_cast<Input*>(getParentEdgeAt(rIdx)->getParent().get())->getMemoryPtr();

    const auto& weightPrec = DnnlExtensionUtils::DataTypeToElementType(inDataTypes[wIdx]);
    const auto& targetWeightDataType = weightsByinputDataType.at(inDataTypes[xIdx]);
    const auto& targetWeightPrec = DnnlExtensionUtils::DataTypeToElementType(targetWeightDataType);

    const VectorDims dims_w = {L, D, DC, G, SC};
    auto w_data_desc =
        std::make_shared<DnnlBlockedMemoryDesc>(Shape(dims_w), targetWeightDataType, getWeightsFormatTagByDims(dims_w));

    auto create_w = [&]() {
        MemoryPtr w_data_mem = std::make_shared<Memory>(getEngine(), w_data_desc);
        auto w_ptr = reinterpret_cast<DataType*>(w_data_mem->getData());
        if (w_ptr == nullptr) {
            THROW_CPU_NODE_ERR("has unallocated internal blob.");
        }
        std::vector<DataType> ie_w_vec;
        DataType* ie_w_ptr = nullptr;

        if (weightPrec != targetWeightPrec) {
            const size_t ie_w_vec_size = getInputShapeAtPort(wIdx).getElementsCount();
            ie_w_vec.resize(ie_w_vec_size);
            ie_w_ptr = ie_w_vec.data();

            cpu_convert(w_const_blob->getData(), ie_w_ptr, weightPrec, targetWeightPrec, ie_w_vec_size);
        } else {
            ie_w_ptr = reinterpret_cast<DataType*>(w_const_blob->getData());
        }

        const uint64_t step = SC * G;
        const uint64_t SC_DC = SC * DC;
        parallel_for2d(G, SC, [&](size_t g, size_t out_i) {
            DataType* l_w_ptr = w_ptr + m_gate_map[g] * SC + out_i;
            DataType* s_w_ptr = ie_w_ptr + out_i * DC + g * SC_DC;
            for (size_t in_i = 0; in_i < DC; in_i++) {
                *l_w_ptr = *s_w_ptr;
                s_w_ptr++;
                l_w_ptr += step;
            }
        });

        return w_data_mem;
    };

    const VectorDims dims_s = {L, D, SC, G, SC};
    auto w_state_desc =
        std::make_shared<DnnlBlockedMemoryDesc>(Shape(dims_s), targetWeightDataType, getWeightsFormatTagByDims(dims_s));

    auto create_r = [&]() {
        MemoryPtr w_state_mem = std::make_shared<Memory>(getEngine(), w_state_desc);
        auto r_ptr = reinterpret_cast<DataType*>(w_state_mem->getData());
        if (r_ptr == nullptr) {
            THROW_CPU_NODE_ERR("has unallocated internal blob.");
        }
        std::vector<DataType> ie_r_vec;
        DataType* ie_r_ptr = nullptr;

        if (weightPrec != targetWeightPrec) {
            const size_t ie_r_vec_size = getInputShapeAtPort(rIdx).getElementsCount();
            ie_r_vec.resize(ie_r_vec_size);
            ie_r_ptr = ie_r_vec.data();

            cpu_convert(r_const_blob->getData(), ie_r_ptr, weightPrec, targetWeightPrec, ie_r_vec_size);
        } else {
            ie_r_ptr = reinterpret_cast<DataType*>(r_const_blob->getData());
        }

        const uint64_t step = SC * G;
        const uint64_t SC_2 = SC * SC;
        parallel_for2d(G, SC, [&](size_t g, size_t out_i) {
            DataType* l_r_ptr = r_ptr + m_gate_map[g] * SC + out_i;
            DataType* s_r_ptr = ie_r_ptr + out_i * SC + g * SC_2;
            for (size_t in_i = 0; in_i < SC; in_i++) {
                *l_r_ptr = *s_r_ptr;
                s_r_ptr++;
                l_r_ptr += step;
            }
        });

        return w_state_mem;
    };

    if (auto weight_cache = context->getWeightsCache()) {
        const std::string hash_w =
            getName() + "_0_" +
            std::to_string(dnnl::impl::primitive_hashing::get_md_hash(*w_data_desc->getDnnlDesc().get()));
        m_initial_weights[0] = *weight_cache->findOrCreate(hash_w, create_w);

        const std::string hash_r =
            getName() + "_1_" +
            std::to_string(dnnl::impl::primitive_hashing::get_md_hash(*w_state_desc->getDnnlDesc().get()));
        m_initial_weights[1] = *weight_cache->findOrCreate(hash_r, create_r);
    } else {
        m_initial_weights[0] = create_w();
        m_initial_weights[1] = create_r();
    }
}

template <element::Type_t ET>
void RNN::fillBiases() {
    using DataType = typename element_type_traits<ET>::value_type;

    if (getParentEdgeAt(bIdx)->getParent()->getType() != Type::Input) {
        THROW_CPU_NODE_ERR("expects Constant for port ", bIdx);
    }
    auto b_const_blob = static_cast<Input*>(getParentEdgeAt(bIdx)->getParent().get())->getMemoryPtr();

    if (inDataTypes[bIdx] != memory::data_type::f32) {
        THROW_CPU_NODE_ERR("doesn't support bias data type: ",
                           DnnlExtensionUtils::DataTypeToElementType(inDataTypes[bIdx]));
    }

    VectorDims dims_b = {L, D, Gb, SC};

    auto dnnl_type = DnnlExtensionUtils::ElementTypeToDataType(ET);
    auto w_bias_data_desc =
        std::make_shared<DnnlBlockedMemoryDesc>(Shape(dims_b), dnnl_type, getWeightsFormatTagByDims(dims_b));

    auto create = [&]() {
        MemoryPtr w_bias_data_mem = std::make_shared<Memory>(getEngine(), w_bias_data_desc);
        auto b_ptr = reinterpret_cast<DataType*>(w_bias_data_mem->getData());
        if (b_ptr == nullptr) {
            THROW_CPU_NODE_ERR("has unallocated internal blob.");
        }

        std::vector<DataType> ie_b_vec;
        DataType* ie_b_ptr = nullptr;

        if (dnnl_type != b_const_blob->getDataType()) {
            const size_t ie_b_vec_size = getInputShapeAtPort(bIdx).getElementsCount();
            ie_b_vec.resize(ie_b_vec_size);
            ie_b_ptr = ie_b_vec.data();

            cpu_convert(b_const_blob->getData(),
                        ie_b_ptr,
                        DnnlExtensionUtils::DataTypeToElementType(b_const_blob->getDataType()),
                        ET,
                        ie_b_vec_size);
        } else {
            ie_b_ptr = reinterpret_cast<DataType*>(b_const_blob->getData());
        }

        const uint64_t step = SC * sizeof(DataType);
        parallel_for(Gb, [&](size_t g) {
            DataType* l_b_ptr = b_ptr + m_gate_map[g] * SC;
            const DataType* l_ie_b_ptr = ie_b_ptr + g * SC;
            cpu_memcpy(l_b_ptr, l_ie_b_ptr, step);
        });

        return w_bias_data_mem;
    };

    if (auto weight_cache = context->getWeightsCache()) {
        const std::string hash_str =
            getName() + "_2_" +
            std::to_string(dnnl::impl::primitive_hashing::get_md_hash(*w_bias_data_desc->getDnnlDesc().get()));
        m_initial_weights[2] = *weight_cache->findOrCreate(hash_str, create);
    } else {
        m_initial_weights[2] = create();
    }
}

void RNN::prepareMemory(const DnnlMemoryDescPtr& new_desc, size_t idx) {
    if (idx >= 3lu) {
        THROW_CPU_NODE_ERR("got invalid weights index: ", idx);
    }

    auto create = [&]() {
        Memory memory{getEngine(), m_initial_weights[idx]->getDescPtr(), m_initial_weights[idx]->getData()};
        MemoryPtr res_ptr = std::make_shared<Memory>(getEngine(), new_desc);
        node::Reorder::reorderData(memory, *res_ptr, context->getParamsCache());
        return res_ptr;
    };

    MemoryPtr res_ptr;
    if (auto weight_cache = context->getWeightsCache()) {
        const std::string hash_str =
            getName() + "_" + std::to_string(idx) + "_" +
            std::to_string(dnnl::impl::primitive_hashing::get_md_hash(*new_desc->getDnnlDesc().get()));
        res_ptr = *weight_cache->findOrCreate(hash_str, create);
        m_weights_pull.insert(res_ptr);
    } else {
        res_ptr = create();
    }

    internalBlobMemory[idx] = std::move(res_ptr);
}

void RNN::copyWeightsData() {
    /* Copy Weight data
     * OV format:
     *   W - [gates, out_state_size, in_data_size]
     *   R - [gates, out_state_size, in_state_size]
     *   B - [gates, out_state_size]
     *
     * DNNL format:
     *   W - [1, 1, in_date_size,  gates, out_state_size]
     *   R - [1, 1, in_state_size, gates, out_state_size]
     *   B - [gates, out_state_size]
     *
     *   Gate order
     *   ====== LSTM ======
     *   Caffe - IFOC, ONNX   - IOFC
     *   OV    - FICO, onednn - IFCO
     *
     *   ====== GRU ======
     *   OV - URO, onednn - URO
     */
    static const uint64_t gate_map_lstm[] = {1, 0, 2, 3};  // FICO -> IFCO
    static const uint64_t gate_map_gru[] = {0, 1, 2, 3};
    static const uint64_t gate_map_rnn[] = {0};
    const uint64_t gate_map_lstm_size = sizeof(gate_map_lstm) / sizeof(uint64_t);
    const uint64_t gate_map_gru_size = sizeof(gate_map_gru) / sizeof(uint64_t);
    const uint64_t gate_map_rnn_size = sizeof(gate_map_rnn) / sizeof(uint64_t);
    if (cell_type == dnnl::algorithm::vanilla_lstm) {
        m_gate_map = gate_map_lstm;
        if (G > gate_map_lstm_size) {
            THROW_CPU_NODE_ERR(". G isn't equal to the size of gate_map.");
        }
    } else if (cell_type == dnnl::algorithm::vanilla_gru || cell_type == dnnl::algorithm::vanilla_augru) {
        m_gate_map = gate_map_gru;
        if (G > gate_map_gru_size) {
            THROW_CPU_NODE_ERR(". G isn't equal to the size of gate_map");
        }
    } else if (cell_type == dnnl::algorithm::lbr_gru || cell_type == dnnl::algorithm::lbr_augru) {
        m_gate_map = gate_map_gru;
        if (G > gate_map_gru_size) {
            THROW_CPU_NODE_ERR(". G isn't equal to the size of gate_map.");
        }
    } else if (cell_type == dnnl::algorithm::vanilla_rnn) {
        m_gate_map = gate_map_rnn;
        if (G > gate_map_rnn_size) {
            THROW_CPU_NODE_ERR(". G isn't equal to the size of gate_map.");
        }
    } else {
        m_gate_map = gate_map_gru;
        if (G > gate_map_gru_size) {
            THROW_CPU_NODE_ERR(". G isn't equal to the size of gate_map.");
        }
    }

    switch (inDataTypes[xIdx]) {
    case memory::data_type::bf16:
    case memory::data_type::f16:
        fillWeights<element::u16>();
        break;
    case memory::data_type::f32:
        fillWeights<element::f32>();
        break;
    case memory::data_type::u8:
    case memory::data_type::s8:
        fillWeights<element::i8>();
        break;
    default:
        THROW_CPU_NODE_ERR("has unsupported data type: ", DnnlExtensionUtils::DataTypeToElementType(inDataTypes[xIdx]));
    }

    fillBiases<element::f32>();

    internalBlobMemory.resize(3);
}

namespace {
dnnl::primitive_desc createPrimitiveDescriptor(const dnnl::engine& engine,
                                               const dnnl::algorithm cellType,
                                               const dnnl::algorithm cellAct,
                                               const dnnl::rnn_direction direction,
                                               const std::vector<DnnlBlockedMemoryDescPtr>& inDataDescs,
                                               const std::vector<DnnlBlockedMemoryDescPtr>& outDataDescs,
                                               const std::vector<dnnl::memory::desc>& wDescs,
                                               const dnnl::primitive_attr& attr) {
    const dnnl::prop_kind propKind = dnnl::prop_kind::forward_inference;

    switch (cellType) {
    case dnnl::algorithm::vanilla_rnn:
        return dnnl::vanilla_rnn_forward::primitive_desc(
            engine,
            propKind,
            cellAct,
            direction,
            inDataDescs[RNN::InOutKind::Layer]->getDnnlDesc(),         // In Data
            inDataDescs[RNN::InOutKind::HiddenState]->getDnnlDesc(),   // In State
            wDescs[0],                                                 // Weights data
            wDescs[1],                                                 // Weights state
            wDescs[2],                                                 // Bias
            outDataDescs[RNN::InOutKind::Layer]->getDnnlDesc(),        // Out Data
            outDataDescs[RNN::InOutKind::HiddenState]->getDnnlDesc(),  // Out State
            attr);
    case dnnl::algorithm::vanilla_gru:
        return dnnl::gru_forward::primitive_desc(engine,
                                                 propKind,
                                                 direction,
                                                 inDataDescs[RNN::InOutKind::Layer]->getDnnlDesc(),        // In Data
                                                 inDataDescs[RNN::InOutKind::HiddenState]->getDnnlDesc(),  // In State
                                                 wDescs[0],                                           // Weights data
                                                 wDescs[1],                                           // Weights state
                                                 wDescs[2],                                           // Bias
                                                 outDataDescs[RNN::InOutKind::Layer]->getDnnlDesc(),  // Out Data
                                                 outDataDescs[RNN::InOutKind::HiddenState]->getDnnlDesc(),  // Out State
                                                 attr);
    case dnnl::algorithm::lbr_gru:
        return dnnl::lbr_gru_forward::primitive_desc(
            engine,
            propKind,
            direction,
            inDataDescs[RNN::InOutKind::Layer]->getDnnlDesc(),         // In Data
            inDataDescs[RNN::InOutKind::HiddenState]->getDnnlDesc(),   // In State
            wDescs[0],                                                 // Weights data
            wDescs[1],                                                 // Weights state
            wDescs[2],                                                 // Bias
            outDataDescs[RNN::InOutKind::Layer]->getDnnlDesc(),        // Out Data
            outDataDescs[RNN::InOutKind::HiddenState]->getDnnlDesc(),  // Out State
            attr);
    case dnnl::algorithm::vanilla_lstm:
        return dnnl::lstm_forward::primitive_desc(
            engine,
            propKind,
            direction,
            inDataDescs[RNN::InOutKind::Layer]->getDnnlDesc(),         // In Data
            inDataDescs[RNN::InOutKind::HiddenState]->getDnnlDesc(),   // In State
            inDataDescs[RNN::InOutKind::CellState]->getDnnlDesc(),     // In State C
            wDescs[0],                                                 // Weights data
            wDescs[1],                                                 // Weights state
            wDescs[2],                                                 // Bias
            outDataDescs[RNN::InOutKind::Layer]->getDnnlDesc(),        // Out Data
            outDataDescs[RNN::InOutKind::HiddenState]->getDnnlDesc(),  // Out State
            outDataDescs[RNN::InOutKind::CellState]->getDnnlDesc(),    // Out State C
            attr);
    case dnnl::algorithm::vanilla_augru:
        return dnnl::augru_forward::primitive_desc(
            engine,
            propKind,
            direction,
            inDataDescs[RNN::InOutKind::Layer]->getDnnlDesc(),         // In Data
            inDataDescs[RNN::InOutKind::HiddenState]->getDnnlDesc(),   // In State
            inDataDescs[RNN::InOutKind::Attention]->getDnnlDesc(),     // In Attention
            wDescs[0],                                                 // Weights data
            wDescs[1],                                                 // Weights state
            wDescs[2],                                                 // Bias
            outDataDescs[RNN::InOutKind::Layer]->getDnnlDesc(),        // Out Data
            outDataDescs[RNN::InOutKind::HiddenState]->getDnnlDesc(),  // Out State
            attr);
    case dnnl::algorithm::lbr_augru:
        return dnnl::lbr_augru_forward::primitive_desc(
            engine,
            propKind,
            direction,
            inDataDescs[RNN::InOutKind::Layer]->getDnnlDesc(),         // In Data
            inDataDescs[RNN::InOutKind::HiddenState]->getDnnlDesc(),   // In State
            inDataDescs[RNN::InOutKind::Attention]->getDnnlDesc(),     // In Attention
            wDescs[0],                                                 // Weights data
            wDescs[1],                                                 // Weights state
            wDescs[2],                                                 // Bias
            outDataDescs[RNN::InOutKind::Layer]->getDnnlDesc(),        // Out Data
            outDataDescs[RNN::InOutKind::HiddenState]->getDnnlDesc(),  // Out State
            attr);
    default:
        OPENVINO_THROW("RNN. Unknown cell type");
    }
}
}  // namespace

void RNN::fillDescs() {
    descs.clear();

    const auto attr = initPrimitiveAttr();

    auto desc = createPrimitiveDescriptor(getEngine(),
                                          cell_type,
                                          cell_act,
                                          direction,
                                          inDataDescs,
                                          outDataDescs,
                                          wDescs,
                                          *attr);

    descs.emplace_back(desc);
}

void RNN::createDescriptor(const std::vector<MemoryDescPtr>& inputDesc, const std::vector<MemoryDescPtr>& outputDesc) {
    if (descs.empty()) {
        wDescs.resize(3);

        /* for descriptor configuration use the same type which is used for internalBlobs
           since internalBlobs are used for the execution, not the initial weights */
        const auto& targetWeightDataType = weightsByinputDataType.at(inDataTypes[xIdx]);
        auto weightsDims = DnnlExtensionUtils::convertToDnnlDims(VectorDims{L, D, DC, G, SC});
        // onednn determines the preferred weight layout.
        wDescs[0] = dnnl::memory::desc(weightsDims, targetWeightDataType, memory::format_tag::any);
        auto statesDims = DnnlExtensionUtils::convertToDnnlDims(VectorDims{L, D, SC, G, SC});
        // onednn determines the preferred weights_iter layout.
        wDescs[1] = dnnl::memory::desc(statesDims, targetWeightDataType, memory::format_tag::any);
        auto biasDims = DnnlExtensionUtils::convertToDnnlDims(VectorDims{L, D, Gb, SC});
        wDescs[2] = dnnl::memory::desc(biasDims, inDataTypes[bIdx], memory::format_tag::ldgo);

        fillDescs();
    }

    // Fill supported config
    NodeConfig config;
    for (const auto& desc : inputDesc) {
        PortConfig dataConfig;
        dataConfig.inPlace(-1);
        dataConfig.constant(false);
        dataConfig.setMemDesc(desc);
        config.inConfs.push_back(dataConfig);
    }

    for (const auto& desc : outputDesc) {
        PortConfig dataConfig;
        dataConfig.inPlace(-1);
        dataConfig.constant(false);
        dataConfig.setMemDesc(desc);
        config.outConfs.push_back(dataConfig);
    }

    supportedPrimitiveDescriptors.emplace_back(config, ref_any);
}

Node::AttrPtr RNN::initPrimitiveAttr() {
    auto attr = std::make_shared<dnnl::primitive_attr>(dnnl::primitive_attr());
    attr->set_scratchpad_mode(dnnl::scratchpad_mode::user);

    if (one_of(inDataTypes[xIdx], memory::data_type::u8, memory::data_type::s8)) {
        const int weightsScaleMask = 0 + (1 << 3)  // bit, indicating the unique scales for `g` dim in `ldigo`
                                     + (1 << 4);   // bit, indicating the unique scales for `o` dim in `ldigo`

        DEBUG_LOG(getName(),
                  ": inputScale: ",
                  inputScale,
                  ", inputShift: ",
                  inputShift,
                  ", weightsScaleMask: ",
                  weightsScaleMask,
                  ", weightsScales[0]: ",
                  weightsScales[0]);

        attr->set_rnn_weights_qparams(weightsScaleMask, weightsScales);
        attr->set_rnn_data_qparams(inputScale, inputShift);
    }

    return attr;
}

void RNN::prepareParams() {
    for (size_t i = 0; i < wIdx; i++) {
        auto memPtr = getSrcMemoryAtPort(i);
        if (!memPtr || !memPtr->isDefined()) {
            THROW_CPU_NODE_ERR("has uninitialized memory at port ", i);
        }
    }
    if ((is_cell && DC != getParentEdgeAt(0)->getMemory().getDesc().getShape().getStaticDims()[1]) ||
        (!is_cell && DC != getParentEdgeAt(0)->getMemory().getDesc().getShape().getStaticDims()[2])) {
        THROW_CPU_NODE_ERR("has incorrect input size value in the first input.");
    }

    auto dataMemPtr = getSrcMemoryAtPort(0);
    const size_t B = dataMemPtr->getShape().getStaticDims()[0];
    const size_t SL = is_cell ? 1lu : dataMemPtr->getShape().getStaticDims()[1];
    const Shape shapeS_4D{L, D, B, SC};

    inDataDescs[0] =
        std::make_shared<DnnlBlockedMemoryDesc>(Shape{SL, B, DC}, inDataTypes[xIdx], memory::format_tag::tnc);
    outDataDescs[0] =
        std::make_shared<DnnlBlockedMemoryDesc>(Shape{SL, B, D * SC}, outDataTypes[yIdx], memory::format_tag::tnc);

    inDataDescs[1] = std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, inDataTypes[hIdx], memory::format_tag::ldnc);
    outDataDescs[1] = std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, outDataTypes[hoIdx], memory::format_tag::ldnc);

    if (haveCellState(cell_type)) {
        inDataDescs[2] =
            std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, inDataTypes[cIdx], memory::format_tag::ldnc);
        outDataDescs[2] =
            std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, outDataTypes[coIdx], memory::format_tag::ldnc);
    } else if (haveAttention(cell_type)) {
        inDataDescs[2] =
            std::make_shared<DnnlBlockedMemoryDesc>(Shape{SL, B, 1}, inDataTypes[aIdx], memory::format_tag::tnc);
    }

    const auto attr = initPrimitiveAttr();
    RNNKey key = {inDataDescs, outDataDescs, wDescs, cell_type, cell_act, direction, *attr};

    auto engine = getEngine();
    auto builder = [&engine](const RNNKey& key) -> executorPtr {
        const auto descPtr = createPrimitiveDescriptor(engine,
                                                       key.cellType,
                                                       key.cellAct,
                                                       key.direction,
                                                       key.inDataDescs,
                                                       key.outDataDescs,
                                                       key.wDescs,
                                                       key.attr);

        return descPtr ? std::make_shared<RnnDnnlExecutor>(descPtr) : nullptr;
    };

    auto cache = context->getParamsCache();
    auto result = cache->getOrCreate(key, builder);
    auto prevExecPtr = execPtr;
    execPtr = result.first;

    if (!execPtr) {
        THROW_CPU_NODE_ERR("does not have primitive descriptor.");
    }

#ifdef CPU_DEBUG_CAPS
    auto pd = execPtr->getPrimitiveDesc();
    DEBUG_LOG("verbose##", getName(), "##", DnnlExtensionUtils::query_pd_info(pd), "\n");
#endif

    if (!primArgs.count(DNNL_ARG_WEIGHTS_LAYER) || !prevExecPtr ||
        !execPtr->getWeightDesc()->isCompatible(*(prevExecPtr->getWeightDesc()))) {
        prepareMemory(execPtr->getWeightDesc(), 0);
        primArgs[DNNL_ARG_WEIGHTS_LAYER] = internalBlobMemory[0]->getPrimitive();
    }

    if (!primArgs.count(DNNL_ARG_WEIGHTS_ITER) || !prevExecPtr ||
        !execPtr->getWeightIterDesc()->isCompatible(*(prevExecPtr->getWeightIterDesc()))) {
        prepareMemory(execPtr->getWeightIterDesc(), 1);
        primArgs[DNNL_ARG_WEIGHTS_ITER] = internalBlobMemory[1]->getPrimitive();
    }

    if (!primArgs.count(DNNL_ARG_BIAS) || !prevExecPtr ||
        !execPtr->getBiasDesc()->isCompatible(*(prevExecPtr->getBiasDesc()))) {
        prepareMemory(execPtr->getBiasDesc(), 2);
        primArgs[DNNL_ARG_BIAS] = internalBlobMemory[2]->getPrimitive();
    }

    auto scratchpadMem = getScratchPadMem(execPtr->getScratchPadDesc());
    primArgs[DNNL_ARG_SCRATCHPAD] = scratchpadMem->getPrimitive();
}

std::shared_ptr<MemoryDesc> RNN::getSrcMemDesc(const dnnl::primitive_desc& prim_desc, size_t idx) const {
    (void)prim_desc;
    return supportedPrimitiveDescriptors[0].getConfig().inConfs[idx].getMemDesc();
}

std::shared_ptr<MemoryDesc> RNN::getDstMemDesc(const dnnl::primitive_desc& prim_desc, size_t idx) const {
    (void)prim_desc;
    return supportedPrimitiveDescriptors[0].getConfig().outConfs[idx].getMemDesc();
}

void RNN::execute(const dnnl::stream& strm) {
    if (!execPtr) {
        THROW_CPU_NODE_ERR("does not have initialized primitive to execute.");
    }

    const auto src_data_mem = getSrcMemoryAtPort(0);
    const auto dst_data_mem = getDstMemoryAtPort(0);

    auto args = primArgs;

    args[DNNL_ARG_SRC_LAYER] = src_data_mem->getPrimitive();
    args[DNNL_ARG_DST_LAYER] = dst_data_mem->getPrimitive();

    int state_i_tags[]{DNNL_ARG_SRC_ITER, DNNL_ARG_SRC_ITER_C};
    int state_o_tags[]{DNNL_ARG_DST_ITER, DNNL_ARG_DST_ITER_C};
    for (size_t s = 0; s < S; s++) {
        args[state_i_tags[s]] = getSrcMemoryAtPort(s + 1)->getPrimitive();
    }
    if (is_augru) {
        const auto atten_port = is_cell ? 5 : 6;
        args[DNNL_ARG_AUGRU_ATTENTION] = getSrcMemoryAtPort(atten_port)->getPrimitive();
    }

    if (is_cell) {
        for (size_t s = 0; s < S; s++) {
            args[state_o_tags[s]] = getDstMemoryAtPort(s)->getPrimitive();
        }
    } else {
        size_t n_ports_with_init_states = outputShapes.size() - 1;  // first is a sequence data
        for (size_t s = 0; s < std::min(S, n_ports_with_init_states); s++) {
            if (s < outputShapes.size()) {
                args[state_o_tags[s]] = getDstMemoryAtPort(s + 1)->getPrimitive();
            }
        }
    }

    execPtr->exec(args, strm);
}

void RNN::executeDynamicImpl(const dnnl::stream& strm) {
    execute(strm);
}

void RNN::cleanup() {
    if (!isDynamicNode()) {
        m_initial_weights[0].reset();
        m_initial_weights[1].reset();
        m_initial_weights[2].reset();
    }

    for (const auto& it : fusedWith) {
        it->cleanup();
    }

    for (const auto& it : mergedWith) {
        it->cleanup();
    }
}

RNN::RnnDnnlExecutor::RnnDnnlExecutor(const dnnl::primitive_desc& pd) : DnnlExecutor(pd) {
    wghts_iter_md = DnnlExtensionUtils::makeDescriptor(pd.weights_desc(1));
    bias_md = DnnlExtensionUtils::makeDescriptor(pd.weights_desc(2));
}

}  // namespace ov::intel_cpu::node
