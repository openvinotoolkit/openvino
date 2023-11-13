// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn.h"
#include <utils/general_utils.h>
#include "ie_precision.hpp"
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"
#include "utils/bfloat16.hpp"
#include "input.h"
#include <dnnl_extension_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include <common/primitive_hashing_utils.hpp>
#include <memory>
#include <shape_inference/shape_inference_ngraph.hpp>
#include "transformations/utils/utils.hpp"

#include "ov_ops/augru_cell.hpp"
#include "ov_ops/augru_sequence.hpp"

#include <ngraph/node.hpp>

#include <oneapi/dnnl/dnnl.hpp>
#include <string>
#include <utility>

#define THROW_ERROR IE_THROW() << getTypeStr() << " node with name '" << getName() << "' "

using namespace dnnl;
using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {
namespace node {

static rnn_direction ieDirection2dnnl(const std::shared_ptr<const ov::Node>& op) {
    ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::FORWARD;
    if (ov::is_type<ov::op::v5::GRUSequence>(op)) {
        direction = ov::as_type_ptr<const ov::op::v5::GRUSequence>(op)->get_direction();
    } else if (ov::is_type<ov::op::v0::LSTMSequence>(op)) {
        direction = ov::as_type_ptr<const ov::op::v0::LSTMSequence>(op)->get_direction();
    } else if (ov::is_type<ov::op::v5::LSTMSequence>(op)) {
        direction = ov::as_type_ptr<const ov::op::v5::LSTMSequence>(op)->get_direction();
    } else if (ov::is_type<ov::op::v5::RNNSequence>(op)) {
        direction = ov::as_type_ptr<const ov::op::v5::RNNSequence>(op)->get_direction();
    }
    return direction == ov::op::RecurrentSequenceDirection::FORWARD ? rnn_direction::unidirectional_left2right
         : direction == ov::op::RecurrentSequenceDirection::REVERSE ? rnn_direction::unidirectional_right2left
         : direction == ov::op::RecurrentSequenceDirection::BIDIRECTIONAL ? rnn_direction::bidirectional_concat
         : rnn_direction::unidirectional_left2right;
}

static dnnl::algorithm ie2dnnl(const std::string& act_type) {
    return act_type == "sigmoid" ? dnnl::algorithm::eltwise_logistic
         : act_type == "tanh"    ? dnnl::algorithm::eltwise_tanh
         : act_type == "relu"    ? dnnl::algorithm::eltwise_relu
         : dnnl::algorithm::undef;
}

static dnnl::algorithm ie2dnnl(const std::shared_ptr<const ov::Node>& op) {
    if (one_of(op->get_type_info(),
            ov::op::v3::GRUCell::get_type_info_static(),
            ov::op::v5::GRUSequence::get_type_info_static())) {
        auto gruCellOp = ov::as_type_ptr<const ov::op::v3::GRUCell>(op);
        auto gruSeqOp = ov::as_type_ptr<const ov::op::v5::GRUSequence>(op);
        if ((gruCellOp && gruCellOp->get_linear_before_reset()) ||
                (gruSeqOp && gruSeqOp->get_linear_before_reset()))
            return dnnl::algorithm::lbr_gru;
        else
            return dnnl::algorithm::vanilla_gru;
    } else if (one_of(op->get_type_info(),
            ov::op::internal::AUGRUCell::get_type_info_static(),
            ov::op::internal::AUGRUSequence::get_type_info_static())) {
        auto gruCellOp = ov::as_type_ptr<const ov::op::internal::AUGRUCell>(op);
        auto gruSeqOp = ov::as_type_ptr<const ov::op::internal::AUGRUSequence>(op);
        if ((gruCellOp && gruCellOp->get_linear_before_reset()) ||
                (gruSeqOp && gruSeqOp->get_linear_before_reset()))
            return dnnl::algorithm::lbr_augru;
        else
            return dnnl::algorithm::vanilla_augru;
    } else if (one_of(op->get_type_info(),
            ov::op::v0::LSTMCell::get_type_info_static(),
            ov::op::v4::LSTMCell::get_type_info_static(),
            ov::op::v0::LSTMSequence::get_type_info_static(),
            ov::op::v5::LSTMSequence::get_type_info_static())) {
        return dnnl::algorithm::vanilla_lstm;
    } else if (one_of(op->get_type_info(),
            ov::op::v0::RNNCell::get_type_info_static(),
            ov::op::v5::RNNSequence::get_type_info_static())) {
        return dnnl::algorithm::vanilla_rnn;
    } else {
        IE_THROW() << "Operation " << op->get_type_name() << " with name '" << op->get_friendly_name() << "' has unsupported cell type.";
    }
}

inline size_t gatesCount(const algorithm& alg) {
    switch (alg) {
        case algorithm::vanilla_rnn:     return 1;
        case algorithm::vanilla_gru:
        case algorithm::vanilla_augru:
        case algorithm::lbr_augru:
        case algorithm::lbr_gru:         return 3;
        case algorithm::vanilla_lstm:    return 4;
        default:
            IE_THROW() << "Unsupported cell type";
            return 0;
    }
}

inline size_t statesCount(const dnnl::algorithm& alg) {
    switch (alg) {
        case dnnl::algorithm::vanilla_rnn:
        case dnnl::algorithm::vanilla_gru:
        case dnnl::algorithm::vanilla_augru:
        case dnnl::algorithm::lbr_augru:
        case dnnl::algorithm::lbr_gru:         return 1;
        case dnnl::algorithm::vanilla_lstm:    return 2;
        default:
            IE_THROW() << "Unsupported cell type";
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
const std::map<memory::data_type, memory::data_type> RNN::weightsByinputDataType {
    // layer data type        weights data type
    {memory::data_type::f32,  memory::data_type::f32},
    {memory::data_type::f16,  memory::data_type::f16},
    {memory::data_type::bf16, memory::data_type::bf16},
    {memory::data_type::u8,   memory::data_type::s8},
    {memory::data_type::s8,   memory::data_type::s8},
};


struct RNNKey {
    const std::vector<DnnlBlockedMemoryDescPtr> inDataDescs;
    const std::vector<DnnlBlockedMemoryDescPtr> outDataDescs;
    const std::vector<dnnl::memory::desc> wDescs;
    dnnl::algorithm cellType;
    dnnl::algorithm cellAct;
    dnnl::rnn_direction direction;
    dnnl::primitive_attr attr;
    size_t hash() const;
    bool operator==(const RNNKey& rhs) const;
};

size_t RNNKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0lu;

    for (auto& desc : inDataDescs) {
        if (desc != nullptr)
            seed = hash_combine(seed, get_md_hash(*desc->getDnnlDesc().get()));
    }
    for (auto& desc : outDataDescs) {
        if (desc != nullptr)
            seed = hash_combine(seed, get_md_hash(*desc->getDnnlDesc().get()));
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
    if (inDataDescs.size() != rhs.inDataDescs.size() || outDataDescs.size() != rhs.outDataDescs.size() || wDescs.size() != rhs.wDescs.size() ||
            cellType != rhs.cellType || cellAct != rhs.cellAct || direction != rhs.direction) {
        return false;
    }

    for (size_t i = 0lu; i < inDataDescs.size(); i++) {
        if (inDataDescs[i] != rhs.inDataDescs[i] && (inDataDescs[i] == nullptr || rhs.inDataDescs[i] == nullptr ||
                inDataDescs[i]->getDnnlDesc() != rhs.inDataDescs[i]->getDnnlDesc()))
            return false;
    }
    for (size_t i = 0lu; i < outDataDescs.size(); i++) {
        if (outDataDescs[i] != rhs.outDataDescs[i] && (outDataDescs[i] == nullptr || rhs.outDataDescs[i] == nullptr ||
                outDataDescs[i]->getDnnlDesc() != rhs.outDataDescs[i]->getDnnlDesc()))
            return false;
    }
    for (size_t i = 0lu; i < wDescs.size(); i++) {
        if (wDescs[i] != rhs.wDescs[i])
            return false;
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
                ov::op::v0::LSTMSequence::get_type_info_static(),
                ov::op::v5::LSTMSequence::get_type_info_static(),
                ov::op::v5::RNNSequence::get_type_info_static())) {
            errorMessage = "Unsupported sequence operation.";
            return false;
        }

        if (one_of(op->get_type_info(), ov::op::v0::RNNCell::get_type_info_static(), ov::op::v3::GRUCell::get_type_info_static())) {
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
        } else if (one_of(op->get_type_info(),
                ov::op::v0::LSTMSequence::get_type_info_static(),
                ov::op::v5::LSTMSequence::get_type_info_static())) {
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
            } else if (!ov::is_type<ov::op::v5::RNNSequence>(op) && rnnCellBase->get_activations() != std::vector<std::string>{"sigmoid", "tanh"}) {
                errorMessage = "Not supported activation functions";
                return false;
            }
        }

        ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::FORWARD;
        int64_t seqLenIdx = -1;
        if (auto gru_seq = ov::as_type_ptr<const ov::op::v5::GRUSequence>(op)) {
            direction = gru_seq->get_direction();
            seqLenIdx = 2;
        } else if (auto lstm_seq = ov::as_type_ptr<const ov::op::v0::LSTMSequence>(op)) {
            if (lstm_seq->get_activations() != std::vector<std::string>{"sigmoid", "tanh", "tanh"}) {
                errorMessage = "Not supported activation functions";
                return false;
            }
            direction = lstm_seq->get_direction();
            seqLenIdx = 3;
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

        if (!one_of(direction, ov::op::RecurrentSequenceDirection::FORWARD, ov::op::RecurrentSequenceDirection::REVERSE)) {
            errorMessage = "Unsupported sequence direction.";
            return false;
        }

        if (seqLenIdx > 0) {
            const auto& data_pshape = op->get_input_partial_shape(0);

            // WA: dynamic shapes make impossible to check seq_len due to shapeOf subgraphs
            // but the sequence is still supported in CPU and doesn't need to be decomposed
            if (data_pshape.is_dynamic())
                return true;

            const int64_t maxSeqLenDimIdx = 1;

            if (data_pshape.rank().is_static() && data_pshape.rank().get_length() > maxSeqLenDimIdx && !data_pshape[maxSeqLenDimIdx].is_static()) {
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
class RnnShapeInfer : public NgraphShapeInfer {
public:
    RnnShapeInfer(std::shared_ptr<ov::Node> op) :
        NgraphShapeInfer(make_shape_inference(op), EMPTY_PORT_MASK) {
            is_sequence = !(RNN::isCell(op));

            native_order = RNN::testNativeOrder(op);
        }

    Result infer(
        const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
        const std::unordered_map<size_t, MemoryPtr>& data_dependency) override {
        auto result = NgraphShapeInfer::infer(input_shapes, data_dependency);
        if (ShapeInferStatus::success != result.status) {
            IE_THROW(Unexpected) << "Unexpected shape inference result status";
        }

        auto& originOutputShapes = result.dims;

        // Graph optimizer makes the same optimization. So this is required to make shapes compatible.
        if (is_sequence && !native_order && originOutputShapes[0].size() == 4lu && originOutputShapes[0][1] == 1lu) {
            originOutputShapes[0].erase(originOutputShapes[0].begin() + 1);
        }
        return {std::move(originOutputShapes), result.status};
    }

private:
    bool is_sequence = false;
    bool native_order = true;
};
class RnnShapeInferFactory final : public ShapeInferFactory {
public:
    RnnShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(op) {}
    ShapeInferPtr makeShapeInfer() const override {
        return std::make_shared<RnnShapeInfer>(m_op);
    }
private:
    std::shared_ptr<ov::Node> m_op;
};

} // namespace

RNN::RNN(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context) :
        Node(op, context, RnnShapeInferFactory(op)) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    is_augru = one_of(op->get_type_info(),
            ov::op::internal::AUGRUCell::get_type_info_static(),
            ov::op::internal::AUGRUSequence::get_type_info_static());

    is_cell = isCell(op);

    if (one_of(op->get_type_info(),
               ov::op::v0::RNNCell::get_type_info_static(),
               ov::op::v3::GRUCell::get_type_info_static())) {
        wIdx = 2; rIdx = 3; bIdx = 4;
        hoIdx = 0;
    } else if (op->get_type_info() == ov::op::internal::AUGRUCell::get_type_info_static()) {
        wIdx = 2; rIdx = 3; bIdx = 4; aIdx = 5;
    } else if (one_of(op->get_type_info(),
                      ov::op::v0::LSTMCell::get_type_info_static(),
                      ov::op::v4::LSTMCell::get_type_info_static())) {
        wIdx = 3; rIdx = 4; bIdx = 5;
        yIdx = hoIdx = 0; coIdx = 1;
    } else if (one_of(op->get_type_info(),
                      ov::op::v5::RNNSequence::get_type_info_static(),
                      ov::op::v5::GRUSequence::get_type_info_static())) {
        sIdx = 2; wIdx = 3; rIdx = 4; bIdx = 5;
        yIdx = 0; hoIdx = 1;
    } else if (op->get_type_info() == ov::op::internal::AUGRUSequence::get_type_info_static()) {
        sIdx = 2; wIdx = 3; rIdx = 4; bIdx = 5; aIdx = 6;
        yIdx = 0; hoIdx = 1;
    } else if (one_of(op->get_type_info(),
                      ov::op::v0::LSTMSequence::get_type_info_static(),
                      ov::op::v5::LSTMSequence::get_type_info_static())) {
        sIdx = 3; wIdx = 4; rIdx = 5; bIdx = 6;
        yIdx = 0; hoIdx = 1; coIdx = 2;
    }

    auto rnnCellBase = std::dynamic_pointer_cast<ngraph::op::util::RNNCellBase>(op);
    if (!rnnCellBase)
        THROW_ERROR << "does not have original layer for RNNCell.";

    cell_type = ie2dnnl(op);
    if (!rnnCellBase->get_activations().empty())
        cell_act = ie2dnnl(rnnCellBase->get_activations()[0]);  // Works only for RNN with one gate

    G = gatesCount(cell_type);
    Gb = (cell_type != dnnl::algorithm::lbr_gru) ? G : G + 1;
    S = statesCount(cell_type);
    SC = rnnCellBase->get_hidden_size();
    N = {getInputShapeAtPort(0).getMinDims()[0], getInputShapeAtPort(0).getMaxDims()[0]};

    const auto& rtInfo = op->get_rt_info();

    if (rtInfo.count("inputScale"))
        inputScale = rtInfo.at("inputScale").as<float>();

    if (rtInfo.count("inputShift"))
        inputShift = rtInfo.at("inputShift").as<float>();

    if (rtInfo.count("weightsScales"))
        weightsScales = rtInfo.at("weightsScales").as<std::vector<float>>();

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
    inDataTypes[xIdx] = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(0));
    inDataTypes[hIdx] = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(1));
    if (haveCellState(cell_type))
        inDataTypes[cIdx] = memory::data_type::f32; // @todo bf16 is also allowed, should be tried out
    if (!is_cell)
        inDataTypes[sIdx] = memory::data_type::s32;
    inDataTypes[wIdx] = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(wIdx));
    inDataTypes[rIdx] = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(rIdx));

    inDataTypes[bIdx] = memory::data_type::f32; // @todo bf16 is also allowed, should be tried out
    if (haveAttention(cell_type))
        inDataTypes[aIdx] = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalInputPrecisionAtPort(aIdx));

    if (!is_cell)
        outDataTypes[yIdx] = DnnlExtensionUtils::IEPrecisionToDataType(getOriginalOutputPrecisionAtPort(0));

    outDataTypes[hoIdx] = inDataTypes[hIdx]; // required by oneDNN. Output hidden state is a input hidden state for the next iteration

    if (haveCellState(cell_type))
        outDataTypes[coIdx] = inDataTypes[cIdx]; // required by oneDNN.

    if (one_of(memory::data_type::bf16, inDataTypes[xIdx], inDataTypes[hIdx]))
        inDataTypes[xIdx] = outDataTypes[yIdx] = outDataTypes[hoIdx] = inDataTypes[hIdx] = memory::data_type::bf16; // required by oneDNN.

    if (one_of(memory::data_type::f16, inDataTypes[xIdx], inDataTypes[hIdx]))
        // onednn doesn't have fp16 instance
        inDataTypes[xIdx] = outDataTypes[yIdx] = outDataTypes[hoIdx] = inDataTypes[hIdx] = memory::data_type::f32; // required by oneDNN.

    if (outDataTypes[yIdx] == memory::data_type::bf16 && one_of(inDataTypes[xIdx], memory::data_type::s8, memory::data_type::u8))
        outDataTypes[yIdx] = memory::data_type::f32; // oneDNN does not support bf16 output precision for quantized rnn primitive yet
}

void RNN::getSupportedDescriptors() {
    configurePortDataTypes();

    if (is_cell)
        fillCellDesc();
    else
        fillSequenceDesc();
}

void RNN::initCell() {
    if (getInputShapeAtPort(0).getRank() != 2lu || getInputShapeAtPort(1).getRank() != 2lu)
        THROW_ERROR << "has incorrect input ranks. Data rank: " << getInputShapeAtPort(0).getRank() <<
                "; Hidden state rank: " << getInputShapeAtPort(1).getRank();
    if (is_augru && getInputShapeAtPort(5).getRank() != 2lu)
        THROW_ERROR << "has incorrect input ranks. Attention rank: " << getInputShapeAtPort(2).getRank();

    T = {1, 1};
    if (cell_type == algorithm::vanilla_lstm)
        DC = getInputShapeAtPort(3).getDims()[1];
    else
        DC = getInputShapeAtPort(2).getDims()[1];

    if (N.isStatic()) {
        // Expected shapes.
        const auto B = N.minVal;
        const Shape shapeD{B, DC}, shapeS{B, SC};

        if ((getInputShapeAtPort(0).isStatic() && getInputShapeAtPort(0) != shapeD) ||
                (getInputShapeAtPort(1).isStatic() && getInputShapeAtPort(1) != shapeS) ||
                (getOutputShapeAtPort(0).isStatic() && getOutputShapeAtPort(0) != shapeS)) {
            THROW_ERROR << "has incorrect input/output shapes. Data shape: " << getInputShapeAtPort(0).toString() <<
                    "; Hidden state input: " << getInputShapeAtPort(1).toString() << "; Hidden state output: " << getOutputShapeAtPort(0).toString();
        }

        if (S == 2) {
            if ((getInputShapeAtPort(2).isStatic() && getInputShapeAtPort(2) != shapeS) ||
                (getOutputShapeAtPort(1).isStatic() && getOutputShapeAtPort(1) != shapeS))
                THROW_ERROR << "has incorrect input/output shapes. Cell state input: " << getInputShapeAtPort(2).toString() <<
                        "; Cell state output: " << getOutputShapeAtPort(1).toString();
        }

        if (is_augru) {
            const Shape shapeA{B, 1};
            if (getInputShapeAtPort(5).isStatic() && getInputShapeAtPort(5) != shapeA) {
                THROW_ERROR << "has incorrect input shapes. Attention shape: " << getInputShapeAtPort(5).toString();
            }
        }
    }
}

void RNN::fillCellDesc() {
    const Shape shapeS_4D = MemoryDescUtils::makeDummyShape({{L, D, N.minVal, SC}, {L, D, N.maxVal, SC}});
    const Shape inShape   = MemoryDescUtils::makeDummyShape({{T.minVal, N.minVal, DC}, {T.maxVal, N.maxVal, DC}});
    const Shape outShape  = MemoryDescUtils::makeDummyShape({{T.minVal, N.minVal, D * SC}, {T.maxVal, N.maxVal, D * SC}});

    // layer input plus states
    if (haveAttention(cell_type)) {
        inDataDescs.reserve(S + 2);
    } else {
        inDataDescs.reserve(S + 1);
    }
    outDataDescs.reserve(S + 1);

    // @todo use indexies instead of emplacing back, since order matters
    inDataDescs.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(inShape, inDataTypes[xIdx], memory::format_tag::tnc));
    outDataDescs.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(outShape, outDataTypes[yIdx], memory::format_tag::tnc));

    inDataDescs.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, inDataTypes[hIdx], memory::format_tag::ldnc));
    outDataDescs.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, outDataTypes[hoIdx], memory::format_tag::ldnc));

    if (haveCellState(cell_type)) {
        inDataDescs.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, inDataTypes[cIdx], memory::format_tag::ldnc));
        outDataDescs.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, outDataTypes[coIdx], memory::format_tag::ldnc));
    } else if (haveAttention(cell_type)) {
        const Shape attnShape = MemoryDescUtils::makeDummyShape({{T.minVal, N.minVal, 1}, {T.maxVal, N.maxVal, 1}});
        inDataDescs.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(attnShape, inDataTypes[aIdx], memory::format_tag::tnc));
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

    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeD, inDataTypes[xIdx], memory::format_tag::nc));

    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeS, inDataTypes[hIdx], memory::format_tag::nc));
    outCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeS, outDataTypes[hoIdx], memory::format_tag::nc));

    if (haveCellState(cell_type)) {
        inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeS, inDataTypes[cIdx], memory::format_tag::nc));
        outCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeS, outDataTypes[coIdx], memory::format_tag::nc));
    }

    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(WShape, inDataTypes[wIdx], memory::format_tag::nc));
    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(RShape, inDataTypes[rIdx], memory::format_tag::nc));
    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(BShape, inDataTypes[bIdx], memory::format_tag::x));

    if (haveAttention(cell_type)) {
        Shape shapeAttn{{N.minVal, 1}, {N.maxVal, 1}};
        inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeAttn, inDataTypes[aIdx], memory::format_tag::nc));
    }

    createDescriptor(inCandidate, outCandidate);
}

void RNN::initSequence() {
    const auto& inDataShape = getInputShapeAtPort(0);
    const auto& outDataShape = getOutputShapeAtPort(0);

    if (inDataShape.getRank() != 3lu || outDataShape.getRank() != 4lu)
        THROW_ERROR << "has incorrect input/output shapes. Input data shape: " << inDataShape.toString() <<
                " Output shape: " << outDataShape.toString();

    if (!one_of(getOriginalInputsNumber(), 6u, 7u))
        THROW_ERROR << "has incorrect number of input ports: " << getOriginalInputsNumber();
    if (!one_of(getOriginalOutputsNumber(), 2u, 3u))
        THROW_ERROR << "has incorrect number of output ports: " << getOriginalOutputsNumber();

    T = {inDataShape.getMinDims()[1], inDataShape.getMaxDims()[1]};
    if (cell_type == algorithm::vanilla_lstm)
        DC = getInputShapeAtPort(4).getDims()[2];
    else
        DC = getInputShapeAtPort(3).getDims()[2];

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
    const Shape inShape   = MemoryDescUtils::makeDummyShape({{T.minVal, N.minVal, DC}, {T.maxVal, N.maxVal, DC}});
    const Shape outShape  = MemoryDescUtils::makeDummyShape({{T.minVal, N.minVal, D * SC}, {T.maxVal, N.maxVal, D * SC}});

    // Try to create descriptor and corresponding configuration
    inDataDescs.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(inShape,  inDataTypes[xIdx], memory::format_tag::tnc));
    outDataDescs.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(outShape, outDataTypes[yIdx], memory::format_tag::tnc));

    inDataDescs.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, inDataTypes[hIdx], memory::format_tag::ldnc));
    outDataDescs.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, outDataTypes[hoIdx], memory::format_tag::ldnc));

    if (haveCellState(cell_type)) {
        inDataDescs.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, inDataTypes[cIdx], memory::format_tag::ldnc));
        outDataDescs.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, outDataTypes[coIdx], memory::format_tag::ldnc));
    } else if (haveAttention(cell_type)) {
        const Shape attnShape = MemoryDescUtils::makeDummyShape({{T.minVal, N.minVal, 1}, {T.maxVal, N.maxVal, 1}});
        inDataDescs.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(attnShape, inDataTypes[aIdx], memory::format_tag::tnc));
    }

    copyWeightsData();

    const Shape shapeNDSC {{N.minVal, D, SC}, {N.maxVal, D, SC}};
    Shape shapeNTSC {{N.minVal, T.minVal, SC}, {N.maxVal, T.maxVal, SC}};
    const Shape shapeNTDC {{N.minVal, T.minVal, DC}, {N.maxVal, T.maxVal, DC}};
    const Shape TShape {VectorDims{N.minVal}, VectorDims{N.maxVal}};
    const Shape WShape {D, G * SC, DC};
    const Shape RShape {D, G * SC, SC};
    const Shape BShape {D, Gb * SC};

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

    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeNTDC, inDataTypes[xIdx], srcLayerMemoryFormat));
    outCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeNTSC, outDataTypes[yIdx], dstLayerMemoryFormat));

    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeNDSC,  inDataTypes[hIdx],    memory::format_tag::tnc));
    outCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeNDSC, outDataTypes[hoIdx], memory::format_tag::tnc));

    // initial cell state
    if (haveCellState(cell_type)) {
        inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeNDSC, inDataTypes[cIdx], memory::format_tag::tnc));
        outCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeNDSC, outDataTypes[coIdx], memory::format_tag::tnc));
    }

    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(TShape, inDataTypes[sIdx], memory::format_tag::x)); // sequence lengths
    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(WShape, inDataTypes[wIdx], memory::format_tag::ntc)); // W
    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(RShape, inDataTypes[rIdx], memory::format_tag::ntc)); // R
    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(BShape, inDataTypes[bIdx], memory::format_tag::nc)); // B

    if (haveAttention(cell_type)) {
        Shape shapeAttn{{N.minVal, T.minVal, 1}, {N.maxVal, T.maxVal, 1}};
        inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeAttn, inDataTypes[aIdx], memory::format_tag::ntc));
    }

    createDescriptor(inCandidate, outCandidate);
}

template <typename Prec>
void RNN::fillWeights(const int *gate_map, const size_t wIdx, const size_t rIdx) {
    const auto& weightPrec       = DnnlExtensionUtils::DataTypeToIEPrecision(inDataTypes[wIdx]);
    const auto& targetWeightPrec = DnnlExtensionUtils::DataTypeToIEPrecision(weightsByinputDataType.at(inDataTypes[xIdx]));

    // create weight blobs (data and state part)
    const VectorDims dims_w = { L, D, DC, G, SC };
    TensorDesc w_data_desc(targetWeightPrec, dims_w, getWeightsLayoutByDims(dims_w, false));

    Blob::Ptr w_data_mem = make_shared_blob<Prec>(w_data_desc);
    w_data_mem->allocate();
    auto w_ptr = static_cast<Prec*>(w_data_mem->buffer());
    if (w_ptr == nullptr)
        IE_THROW(NotAllocated) << "Internal blob was not allocated for node " << getName() << ".";

    const VectorDims dims_s = { L, D, SC, G, SC };
    TensorDesc w_state_desc(targetWeightPrec, dims_s, getWeightsLayoutByDims(dims_s, false));
    Blob::Ptr w_state_mem = make_shared_blob<Prec>(w_state_desc);
    w_state_mem->allocate();
    auto r_ptr = static_cast<Prec*>(w_state_mem->buffer());
    if (r_ptr == nullptr)
        IE_THROW(NotAllocated) << "Internal blob was not allocated for node " << getName() << ".";

    const size_t ie_w_vec_size = getInputShapeAtPort(wIdx).getElementsCount();
    const size_t ie_r_vec_size = getInputShapeAtPort(rIdx).getElementsCount();

    auto *wInputNode = dynamic_cast<Input *>(getParentEdgesAtPort(wIdx)[0]->getParent().get());
    auto wConstBlob = wInputNode->getMemoryPtr();

    auto *rInputNode = dynamic_cast<Input *>(getParentEdgesAtPort(rIdx)[0]->getParent().get());
    auto rConstBlob = rInputNode->getMemoryPtr();

    std::vector<Prec> ie_w_vec(ie_w_vec_size), ie_r_vec(ie_r_vec_size);

    auto ie_w_ptr = ie_w_vec.data();
    auto ie_r_ptr = ie_r_vec.data();

    cpu_convert(wConstBlob->getData(), ie_w_ptr, weightPrec, targetWeightPrec, ie_w_vec_size);
    cpu_convert(rConstBlob->getData(), ie_r_ptr, weightPrec, targetWeightPrec, ie_r_vec_size);

    const int step = SC * G;

    for (size_t g = 0; g < G; g++) {
        for (size_t out_i = 0; out_i < SC; out_i++) {
            Prec *l_w_ptr = w_ptr + gate_map[g] * SC + out_i;
            for (size_t in_i = 0; in_i < DC; in_i++) {
                *l_w_ptr = *ie_w_ptr;
                ie_w_ptr++;
                l_w_ptr += step;
            }

            Prec *l_r_ptr = r_ptr + gate_map[g] * SC + out_i;
            for (size_t in_i = 0; in_i < SC; in_i++) {
                *l_r_ptr = *ie_r_ptr;
                ie_r_ptr++;
                l_r_ptr += step;
            }
        }
    }

    internalBlobs.push_back(w_data_mem);
    internalBlobs.push_back(w_state_mem);
}

template <Precision::ePrecision Prec>
void RNN::fillBiases(const int *gate_map) {
    using dataType = typename PrecisionTrait<Prec>::value_type;

    if (inDataTypes[bIdx] != memory::data_type::f32) {
        THROW_ERROR << "doesn't support bias data type: " << DnnlExtensionUtils::DataTypeToIEPrecision(inDataTypes[bIdx]);
    }

    VectorDims dims_b = { L, D, Gb, SC };
    TensorDesc w_bias_data_desc(Prec, dims_b, getWeightsLayoutByDims(dims_b, false));
    Blob::Ptr w_bias_data_mem = make_shared_blob<dataType>(w_bias_data_desc);
    w_bias_data_mem->allocate();
    auto b_ptr = static_cast<dataType*>(w_bias_data_mem->buffer());
    if (b_ptr == nullptr)
        IE_THROW(NotAllocated) << "Internal blob was not allocated for node " << getName() << ".";

    auto *constInputNode = dynamic_cast<Input *>(getParentEdgesAtPort(bIdx)[0]->getParent().get());
    auto constBlob = constInputNode->getMemoryPtr();
    auto const elementsCount = constBlob->getSize() / constBlob->getDesc().getPrecision().size();

    std::vector<dataType> ie_b_vec(elementsCount);
    cpu_convert(constBlob->getData(),
                &ie_b_vec[0],
                DnnlExtensionUtils::DataTypeToIEPrecision(constBlob->getDataType()),
                Prec,
                elementsCount);

    for (size_t g = 0; g < Gb; g++) {
        dataType *l_b_ptr = b_ptr + gate_map[g] * SC;
        const dataType *l_ie_b_ptr = &ie_b_vec[g * SC];
        cpu_memcpy(l_b_ptr, l_ie_b_ptr, SC * sizeof(typename PrecisionTrait<Prec>::value_type));
    }
    // @todo replace push_back with copy assignment by index, since order matters
    internalBlobs.push_back(w_bias_data_mem);
}

void RNN::copyWeightsData() {
    /* Copy Weight data
     * IE format:
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
     *   IE    - FICO, onednn - IFCO
     *
     *   ====== GRU ======
     *   IE - URO, onednn - URO
     */
    const int gate_map_lstm[] = {1, 0, 2, 3};  // FICO -> IFCO
    const int gate_map_gru[]  = {0, 1, 2, 3};
    const int gate_map_rnn[]  = {0};
    const int *gate_map;
    const int gate_map_lstm_size = sizeof(gate_map_lstm) / sizeof(int);
    const int gate_map_gru_size = sizeof(gate_map_gru) / sizeof(int);
    const int gate_map_rnn_size = sizeof(gate_map_rnn) / sizeof(int);
    if (cell_type == dnnl::algorithm::vanilla_lstm) {
        gate_map = gate_map_lstm;
        if (G > gate_map_lstm_size) {
            THROW_ERROR << ". G isn't equal to the size of gate_map.";
        }
    } else if (cell_type == dnnl::algorithm::vanilla_gru || cell_type == dnnl::algorithm::vanilla_augru) {
        gate_map = gate_map_gru;
        if (G > gate_map_gru_size) {
            THROW_ERROR << ". G isn't equal to the size of gate_map";
        }
    } else if (cell_type == dnnl::algorithm::lbr_gru || cell_type == dnnl::algorithm::lbr_augru) {
        gate_map = gate_map_gru;
        if (G > gate_map_gru_size) {
            THROW_ERROR << ". G isn't equal to the size of gate_map.";
        }
    } else if (cell_type == dnnl::algorithm::vanilla_rnn) {
        gate_map = gate_map_rnn;
        if (G > gate_map_rnn_size) {
            THROW_ERROR << ". G isn't equal to the size of gate_map.";
        }
    } else {
        gate_map = gate_map_gru;
        if (G > gate_map_gru_size) {
            THROW_ERROR << ". G isn't equal to the size of gate_map.";
        }
    }

    const auto& dataType = inDataTypes[xIdx];
    if (one_of(dataType, memory::data_type::bf16, memory::data_type::f16)) {
        fillWeights<uint16_t>(gate_map, wIdx, rIdx);
    } else if (dataType == memory::data_type::f32) {
        // WA To avoid different weights layer and iter formats in FP32 case
        if (T.minVal > 1 || N.maxVal < optimalBatchSize)
            wFormat = dnnl::memory::format_tag::ldigo;
        fillWeights<float>(gate_map, wIdx, rIdx);
    } else if (dataType == memory::data_type::u8 || dataType == memory::data_type::s8) {
        fillWeights<int8_t>(gate_map, wIdx, rIdx);
    } else {
        THROW_ERROR << "has unsupported data type: " << DnnlExtensionUtils::DataTypeToIEPrecision(dataType);
    }

    fillBiases<Precision::FP32>(gate_map);
}

namespace {
dnnl::primitive_desc createPrimitiveDescriptor(const dnnl::engine        engine,
                                               const dnnl::algorithm     cellType,
                                               const dnnl::algorithm     cellAct,
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
        return dnnl::gru_forward::primitive_desc(
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
        IE_THROW() << "RNN. Unknown cell type";
    }
}
} // namespace

void RNN::fillDescs() {
    descs.clear();

    const auto attr = initPrimitiveAttr();

    auto desc = createPrimitiveDescriptor(
        getEngine(),
        cell_type,
        cell_act,
        direction,
        inDataDescs,
        outDataDescs,
        wDescs,
        *attr);

    descs.emplace_back(desc);
}

void RNN::createDescriptor(const std::vector<MemoryDescPtr> &inputDesc,
                           const std::vector<MemoryDescPtr> &outputDesc) {
    if (descs.empty()) {
        wDescs.resize(3);

        /* for descriptor configuration use the same type which is used for internalBlobs
           since internalBlobs are used for the execution, not the initial weights */
        const auto& targetWeightDataType = weightsByinputDataType.at(inDataTypes[xIdx]);
        auto weightsDims = DnnlExtensionUtils::convertToDnnlDims(VectorDims{ L, D, DC, G, SC });
        wDescs[0] = dnnl::memory::desc(weightsDims, targetWeightDataType, wFormat);
        auto statesDims = DnnlExtensionUtils::convertToDnnlDims(VectorDims{ L, D, SC, G, SC });
        wDescs[1] = dnnl::memory::desc(statesDims, targetWeightDataType, wFormat);
        auto biasDims = DnnlExtensionUtils::convertToDnnlDims(VectorDims{ L, D, Gb, SC });
        wDescs[2] = dnnl::memory::desc(biasDims, inDataTypes[bIdx], memory::format_tag::ldgo);

        fillDescs();
    }

    // Fill supported config
    NodeConfig config;
    for (const auto &desc : inputDesc) {
        PortConfig dataConfig;
        dataConfig.inPlace(-1);
        dataConfig.constant(false);
        dataConfig.setMemDesc(desc);
        config.inConfs.push_back(dataConfig);
    }

    for (const auto &desc : outputDesc) {
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
        const int weightsScaleMask = 0
            + (1 << 3) // bit, indicating the unique scales for `g` dim in `ldigo`
            + (1 << 4); // bit, indicating the unique scales for `o` dim in `ldigo`

        DEBUG_LOG(getName(), ": inputScale: ", inputScale, ", inputShift: ", inputShift,
                  ", weightsScaleMask: ", weightsScaleMask, ", weightsScales[0]: ", weightsScales[0]);

        attr->set_rnn_weights_qparams(weightsScaleMask, weightsScales);
        attr->set_rnn_data_qparams(inputScale, inputShift);
    }

    return attr;
}

void RNN::prepareParams() {
    for (size_t i = 0; i < wIdx; i++) {
        auto memPtr = getParentEdgesAtPort(i).front()->getMemoryPtr();
        if (!memPtr || !memPtr->isAllocated())
            THROW_ERROR << "has uninitialized memory at port " << i;
    }
    if ((is_cell && DC != getParentEdgesAtPort(0)[0]->getMemory().getDesc().getShape().getStaticDims()[1]) ||
        (!is_cell && DC != getParentEdgesAtPort(0)[0]->getMemory().getDesc().getShape().getStaticDims()[2]))
            THROW_ERROR << "has incorrect input size value in the first input.";

    auto dataMemPtr = getParentEdgesAtPort(0).front()->getMemoryPtr();
    const size_t B = dataMemPtr->getShape().getStaticDims()[0];
    const size_t SL = is_cell ? 1lu : dataMemPtr->getShape().getStaticDims()[1];
    const Shape shapeS_4D{L, D, B, SC};

    inDataDescs[0] = std::make_shared<DnnlBlockedMemoryDesc>(Shape{SL, B, DC}, inDataTypes[xIdx], memory::format_tag::tnc);
    outDataDescs[0] = std::make_shared<DnnlBlockedMemoryDesc>(Shape{SL, B, D * SC}, outDataTypes[yIdx], memory::format_tag::tnc);

    inDataDescs[1] = std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, inDataTypes[hIdx], memory::format_tag::ldnc);
    outDataDescs[1] = std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, outDataTypes[hoIdx], memory::format_tag::ldnc);

    if (haveCellState(cell_type)) {
        inDataDescs[2] = std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, inDataTypes[cIdx], memory::format_tag::ldnc);
        outDataDescs[2] = std::make_shared<DnnlBlockedMemoryDesc>(shapeS_4D, outDataTypes[coIdx], memory::format_tag::ldnc);
    } else if (haveAttention(cell_type)) {
        inDataDescs[2] = std::make_shared<DnnlBlockedMemoryDesc>(Shape{SL, B, 1}, inDataTypes[aIdx], memory::format_tag::tnc);
    }

    bool wFormatWasChanged = false;
    // WA To avoid different weights layer and iter formats in FP32 case.
    if (one_of(inDataTypes[xIdx], memory::data_type::f32) &&
        (SL != 1 || B < optimalBatchSize)) {
        if (wFormat != dnnl::memory::format_tag::ldigo) {
            wFormat = dnnl::memory::format_tag::ldigo;
            wFormatWasChanged = true;
        }
    } else if (wFormat != dnnl::memory::format_tag::any) {
        wFormat = dnnl::memory::format_tag::any;
        wFormatWasChanged = true;
    }

    if (wFormatWasChanged) {
        auto weightsDims = DnnlExtensionUtils::convertToDnnlDims(VectorDims{ L, D, DC, G, SC });
        const auto& targetWeightDataType = weightsByinputDataType.at(inDataTypes[xIdx]);
        wDescs[0] = dnnl::memory::desc(weightsDims, targetWeightDataType, wFormat);
        auto statesDims = DnnlExtensionUtils::convertToDnnlDims(VectorDims{ L, D, SC, G, SC });
        wDescs[1] = dnnl::memory::desc(statesDims, targetWeightDataType, wFormat);
    }

    const auto attr = initPrimitiveAttr();
    RNNKey key = { inDataDescs, outDataDescs, wDescs, cell_type, cell_act, direction, *attr };

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
        IE_THROW() << "Primitive descriptor was not found for node " << getName() << ".";
    }

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
    (void) prim_desc;
    return supportedPrimitiveDescriptors[0].getConfig().inConfs[idx].getMemDesc();
}

std::shared_ptr<MemoryDesc> RNN::getDstMemDesc(const dnnl::primitive_desc& prim_desc, size_t idx) const {
    (void) prim_desc;
    return supportedPrimitiveDescriptors[0].getConfig().outConfs[idx].getMemDesc();
}

void RNN::execute(dnnl::stream strm) {
    if (!execPtr)
        THROW_ERROR << "does not have initialized primitive to execute.";

    const auto src_data_mem = getParentEdgeAt(0)->getMemoryPtr();
    const auto dst_data_mem = getChildEdgeAt(0)->getMemoryPtr();

    auto args = primArgs;

    args[DNNL_ARG_SRC_LAYER] = src_data_mem->getPrimitive();
    args[DNNL_ARG_DST_LAYER] = dst_data_mem->getPrimitive();

    int state_i_tags[] {DNNL_ARG_SRC_ITER, DNNL_ARG_SRC_ITER_C};
    int state_o_tags[] {DNNL_ARG_DST_ITER, DNNL_ARG_DST_ITER_C};
    for (size_t s = 0; s < S; s++) {
        args[state_i_tags[s]] = getParentEdgeAt(s+1)->getMemoryPtr()->getPrimitive();
    }
    if (is_augru) {
        const auto atten_port = is_cell ? 5 : 6;
        args[DNNL_ARG_AUGRU_ATTENTION] = getParentEdgeAt(atten_port)->getMemoryPtr()->getPrimitive();
    }

    if (is_cell) {
        for (size_t s = 0; s < S; s++) {
            args[state_o_tags[s]] = getChildEdgesAtPort(s)[0]->getMemoryPtr()->getPrimitive();
        }
    } else {
        size_t n_ports_with_init_states = outputShapes.size() - 1; // first is a sequence data
        for (size_t s = 0; s < std::min(S, n_ports_with_init_states); s++) {
            if (s < outputShapes.size()) {
                args[state_o_tags[s]] = getChildEdgesAtPort(s+1)[0]->getMemoryPtr()->getPrimitive();
            }
        }
    }

    execPtr->exec(args, strm);
}

void RNN::executeDynamicImpl(dnnl::stream strm) {
    execute(strm);
}

void RNN::cleanup() {
    if (!isDynamicNode()) {
        internalBlobs.clear();
    }

    for (auto it : fusedWith) {
        it->cleanup();
    }

    for (auto it : mergedWith) {
        it->cleanup();
    }
}

RNN::RnnDnnlExecutor::RnnDnnlExecutor(const dnnl::primitive_desc& pd) : DnnlExecutor(pd) {
    wghts_iter_md = DnnlExtensionUtils::makeDescriptor(pd.weights_desc(1));
    bias_md = DnnlExtensionUtils::makeDescriptor(pd.weights_desc(2));
}

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
