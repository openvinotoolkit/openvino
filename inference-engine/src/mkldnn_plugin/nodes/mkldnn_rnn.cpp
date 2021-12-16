// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_rnn.h"
#include <utils/general_utils.h>
#include "nodes/common/cpu_memcpy.h"
#include "nodes/common/cpu_convert.h"
#include "utils/bfloat16.hpp"
#include "mkldnn_input_node.h"
#include <mkldnn_extension_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"

#include <ngraph/node.hpp>

#include <string>
#include <utility>

#define THROW_ERROR IE_THROW() << NameFromType(getType()) << " node with name '" << getName() << "' "

using namespace mkldnn;
using namespace InferenceEngine;

namespace MKLDNNPlugin {

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
         : rnn_direction::unidirectional;
}

static mkldnn::algorithm ie2dnnl(const std::string& act_type) {
    return act_type == "sigmoid" ? mkldnn::algorithm::eltwise_logistic
         : act_type == "tanh"    ? mkldnn::algorithm::eltwise_tanh
         : act_type == "relu"    ? mkldnn::algorithm::eltwise_relu
         : mkldnn::algorithm::undef;
}

static mkldnn::algorithm ie2dnnl(const std::shared_ptr<const ov::Node>& op) {
    if (one_of(op->get_type_info(),
            ov::op::v3::GRUCell::get_type_info_static(),
            ov::op::v5::GRUSequence::get_type_info_static())) {
        auto gruCellOp = ov::as_type_ptr<const ov::op::v3::GRUCell>(op);
        auto gruSeqOp = ov::as_type_ptr<const ov::op::v5::GRUSequence>(op);
        if ((gruCellOp && gruCellOp->get_linear_before_reset()) ||
                (gruSeqOp && gruSeqOp->get_linear_before_reset()))
            return mkldnn::algorithm::lbr_gru;
        else
            return mkldnn::algorithm::vanilla_gru;
    } else if (one_of(op->get_type_info(),
            ov::op::v0::LSTMCell::get_type_info_static(),
            ov::op::v4::LSTMCell::get_type_info_static(),
            ov::op::v0::LSTMSequence::get_type_info_static(),
            ov::op::v5::LSTMSequence::get_type_info_static())) {
        return mkldnn::algorithm::vanilla_lstm;
    } else if (one_of(op->get_type_info(),
            ov::op::v0::RNNCell::get_type_info_static(),
            ov::op::v5::RNNSequence::get_type_info_static())) {
        return mkldnn::algorithm::vanilla_rnn;
    } else {
        IE_THROW() << "Operation " << op->get_type_name() << " with name '" << op->get_friendly_name() << "' has unsupported cell type.";
    }
}

inline size_t gatesCount(const mkldnn::algorithm& alg) {
    switch (alg) {
        case mkldnn::algorithm::vanilla_rnn:     return 1;
        case mkldnn::algorithm::vanilla_gru:
        case mkldnn::algorithm::lbr_gru:         return 3;
        case mkldnn::algorithm::vanilla_lstm:    return 4;
        default:
            IE_THROW() << "Unsupported cell type";
            return 0;
    }
}

inline size_t statesCount(const mkldnn::algorithm& alg) {
    switch (alg) {
        case mkldnn::algorithm::vanilla_rnn:
        case mkldnn::algorithm::vanilla_gru:
        case mkldnn::algorithm::lbr_gru:         return 1;
        case mkldnn::algorithm::vanilla_lstm:    return 2;
        default:
            IE_THROW() << "Unsupported cell type";
            return 0;
    }
}

inline bool haveCellState(const mkldnn::algorithm& alg) {
    return alg == mkldnn::algorithm::vanilla_lstm;
}

const std::map<Precision, Precision> MKLDNNRNN::weightsByLayerPrec {
    // layer precision,                weights precision
    {Precision::FP32, Precision::FP32},
    {Precision::BF16, Precision::BF16},
    // FP16 and U8 are not supported yet
    // {Precision::FP16, Precision::FP16},
    // {Precision::U8,   Precision::I8},
};

bool MKLDNNRNN::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                ov::op::v3::GRUCell::get_type_info_static(),
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
            if (!ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(2)) ||
                    !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(3)) ||
                    (op->get_input_size() > 4 && !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(4)))) {
                errorMessage = "Node expects constants as W, R, B inputs.";
                return false;
            }
        } else if (one_of(op->get_type_info(),
                ov::op::v0::LSTMCell::get_type_info_static(),
                ov::op::v4::LSTMCell::get_type_info_static(),
                ov::op::v5::GRUSequence::get_type_info_static(),
                ov::op::v5::RNNSequence::get_type_info_static())) {
            // Plug-in does not support dynamism on weights.
            if (!ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(3)) ||
                    !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(4)) ||
                    (op->get_input_size() > 5 && !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(5)))) {
                errorMessage = "Node expects constants as W, R, B inputs.";
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
            if (!ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(4)) ||
                    !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(5)) ||
                    !ov::is_type<ov::op::v0::Constant>(op->get_input_node_ptr(6))) {
                errorMessage = "Node expects constants as W, R, B inputs.";
                return false;
            }
        }

        auto rnnCellBase = ov::as_type_ptr<const ov::op::util::RNNCellBase>(op);
        if (rnnCellBase && rnnCellBase->get_clip() != 0.f) {
            errorMessage = "Clipping is not supported for RNN primitive.";
            return false;
        }

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
        if (!one_of(direction, ov::op::RecurrentSequenceDirection::FORWARD, ov::op::RecurrentSequenceDirection::REVERSE)) {
            errorMessage = "Unsupported sequence direction.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNRNN::MKLDNNRNN(const std::shared_ptr<ov::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    internalBlobDesc.emplace_back([&](primitive_desc_iterator& primitive_desc_it, size_t idx) -> DnnlMemoryDescPtr {
        return MKLDNNExtensionUtils::makeDescriptor(primitive_desc_it.weights_desc(0));
    });
    internalBlobDesc.emplace_back([&](primitive_desc_iterator& primitive_desc_it, size_t idx) -> DnnlMemoryDescPtr {
        return MKLDNNExtensionUtils::makeDescriptor(primitive_desc_it.weights_desc(1));
    });
    internalBlobDesc.emplace_back([&](primitive_desc_iterator& primitive_desc_it, size_t idx) -> DnnlMemoryDescPtr {
        return MKLDNNExtensionUtils::makeDescriptor(primitive_desc_it.weights_desc(2));
    });

    is_cell = one_of(op->get_type_info(),
            ov::op::v0::RNNCell::get_type_info_static(),
            ov::op::v3::GRUCell::get_type_info_static(),
            ov::op::v0::LSTMCell::get_type_info_static(),
            ov::op::v4::LSTMCell::get_type_info_static());

    if (one_of(op->get_type_info(),
               ov::op::v0::RNNCell::get_type_info_static(),
               ov::op::v3::GRUCell::get_type_info_static())) {
        wIdx = 2; rIdx = 3; bIdx = 4;
    } else if (one_of(op->get_type_info(),
                      ov::op::v5::RNNSequence::get_type_info_static(),
                      ov::op::v0::LSTMCell::get_type_info_static(),
                      ov::op::v4::LSTMCell::get_type_info_static(),
                      ov::op::v5::GRUSequence::get_type_info_static())) {
        wIdx = 3; rIdx = 4; bIdx = 5;
    } else if (one_of(op->get_type_info(),
                      ov::op::v0::LSTMSequence::get_type_info_static(),
                      ov::op::v5::LSTMSequence::get_type_info_static())) {
        wIdx = 4; rIdx = 5; bIdx = 6;
    }

    auto rnnCellBase = std::dynamic_pointer_cast<ngraph::op::util::RNNCellBase>(op);
    if (!rnnCellBase)
        THROW_ERROR << "does not have original layer for RNNCell.";

    runtimePrecision = getOriginalInputPrecisionAtPort(0);

    cell_type = ie2dnnl(op);
    cell_act = mkldnn::algorithm::undef;
    if (!rnnCellBase->get_activations().empty())
        cell_act = ie2dnnl(rnnCellBase->get_activations()[0]);  // Works only for RNN with one gate

    G = gatesCount(cell_type);
    Gb = (cell_type != mkldnn::algorithm::lbr_gru) ? G : G + 1;
    S = statesCount(cell_type);
    SC = rnnCellBase->get_hidden_size();
    N = getInputShapeAtPort(0).getInterval(0);

    if (is_cell) {
        initCell();
    } else {
        direction = ieDirection2dnnl(op);

        nativeOrder = false;
        const auto& rtInfo = op->get_rt_info();
        if (rtInfo.count("seqAxis")) {
            nativeOrder = rtInfo.at("seqAxis").as<int64_t>() == 0;
        }

        initSequence();
    }
}

bool MKLDNNRNN::created() const {
    return getType() == (is_cell ? RNNCell : RNNSeq);
}

void MKLDNNRNN::getSupportedDescriptors() {
    if (is_cell)
        fillCellDesc();
    else
        fillSequenceDesc();
}

void MKLDNNRNN::initCell() {
    if (getInputShapeAtPort(0).getRank() != 2lu || getInputShapeAtPort(1).getRank() != 2lu)
        THROW_ERROR << "has incorrect input ranks. Data rank: " << getInputShapeAtPort(0).getRank() <<
                "; Hidden state rank: " << getInputShapeAtPort(1).getRank();

    T = 1;
    DC = getInputShapeAtPort(0).getDims()[1];

    const Shape shapeD{N, DC}, shapeS{N, SC};

    if (getInputShapeAtPort(0) != shapeD || getInputShapeAtPort(1) != shapeS || getOutputShapeAtPort(0) != shapeS)
        THROW_ERROR << "has incorrect input/output shapes. Data shape: " << getInputShapeAtPort(0).toString() <<
                "; Hidden state input: " << getInputShapeAtPort(1).toString() << "; Hidden state output: " << getOutputShapeAtPort(0).toString();

    if (S == 2) {
        if (getInputShapeAtPort(2) != shapeS || getOutputShapeAtPort(1) != shapeS)
            THROW_ERROR << "has incorrect input/output shapes. Cell state input: " << getInputShapeAtPort(2).toString() <<
                    "; Cell state output: " << getOutputShapeAtPort(1).toString();
    }
}

void MKLDNNRNN::fillCellDesc() {
    const auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(runtimePrecision);
    const size_t B = N.isStatic() ? N.getMaxValue() : 64lu; // Dummy value
    const Shape shapeS_4D {L, D, B, SC};

    // layer input plus states
    inDataDescs.reserve(S + 1);
    outDataDescs.reserve(S + 1);

    inDataDescs.emplace_back(Shape{T, B, DC}, dataType, memory::format_tag::tnc);
    outDataDescs.emplace_back(Shape{T, B, SC}, dataType, memory::format_tag::tnc);

    inDataDescs.emplace_back(shapeS_4D, dataType, memory::format_tag::ldnc);
    outDataDescs.emplace_back(shapeS_4D, dataType, memory::format_tag::ldnc);

    if (haveCellState(cell_type)) {
        inDataDescs.emplace_back(shapeS_4D, memory::data_type::f32, memory::format_tag::ldnc);
        outDataDescs.emplace_back(shapeS_4D, memory::data_type::f32, memory::format_tag::ldnc);
    }

    copyWeightsData();

    // Expected shapes
    Shape shapeD{N, DC}, shapeS{N, SC}, WShape(VectorDims{SC * G, DC}), RShape(VectorDims{SC * G, SC}), BShape(VectorDims{SC * Gb});
    std::vector<MemoryDescPtr> inCandidate, outCandidate;
    inCandidate.reserve(6);

    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeD, dataType, memory::format_tag::nc));
    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeS, dataType, memory::format_tag::nc));
    outCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeS, dataType, memory::format_tag::nc));

    if (haveCellState(cell_type)) {
        inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeS, memory::data_type::f32, memory::format_tag::nc));
        outCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(shapeS, memory::data_type::f32, memory::format_tag::nc));
    }
    if (one_of(cell_type, mkldnn::algorithm::vanilla_rnn, mkldnn::algorithm::vanilla_gru, mkldnn::algorithm::lbr_gru, mkldnn::algorithm::vanilla_lstm)) {
        inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(WShape, memory::data_type::f32, memory::format_tag::nc));
        inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(RShape, memory::data_type::f32, memory::format_tag::nc));
        inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(BShape, memory::data_type::f32, memory::format_tag::x));
    }

    createDescriptor(inCandidate, outCandidate);
}

void MKLDNNRNN::initSequence() {
    const auto& inDataShape = getInputShapeAtPort(0);
    const auto& outDataShape = getOutputShapeAtPort(0);

    if (inDataShape.getRank() != 3lu || outDataShape.getRank() != 4lu)
        THROW_ERROR << "has incorrect input/output shapes. Input data shape: " << inDataShape.toString() <<
                " Output shape: " << outDataShape.toString();

    if (!one_of(getOriginalInputsNumber(), 6, 7))
        THROW_ERROR << "has incorrect number of input ports: " << getOriginalInputsNumber();
    if (!one_of(getOriginalOutputsNumber(), 2, 3))
        THROW_ERROR << "has incorrect number of output ports: " << getOriginalOutputsNumber();

    T = inDataShape.getInterval(1);
    DC = inDataShape.getDims()[2];

    // layer input plus states
    inDataDescs.reserve(S + 1);
    outDataDescs.reserve(S + 1);
}

void MKLDNNRNN::fillSequenceDesc() {
    const auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(runtimePrecision);
    const size_t B = N.isStatic() ? N.getMaxValue() : 64lu; // Dummy value
    const size_t SL = T.isStatic() ? T.getMaxValue() : 1lu; // Dummy value
    const Shape shapeS_4D {L, D, B, SC};

    // Try to create descriptor and corresponding configuration
    inDataDescs.emplace_back(Shape{SL, B, DC},  dataType, memory::format_tag::tnc);
    outDataDescs.emplace_back(Shape{SL, B, SC}, dataType, memory::format_tag::tnc);

    inDataDescs.emplace_back(shapeS_4D, dataType, memory::format_tag::ldnc);
    outDataDescs.emplace_back(shapeS_4D, dataType, memory::format_tag::ldnc);

    if (haveCellState(cell_type)) {
        inDataDescs.emplace_back(shapeS_4D, memory::data_type::f32, memory::format_tag::ldnc);
        outDataDescs.emplace_back(shapeS_4D, memory::data_type::f32, memory::format_tag::ldnc);
    }

    copyWeightsData();

    std::vector<MemoryDescPtr> inCandidate;
    inCandidate.reserve(7);

    if (nativeOrder)
        inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(inputShapes[RNNInOutKind::Layer], dataType, memory::format_tag::tnc));
    else if (N.isStatic() && N.getMaxValue() == 1)
        // WA to avoid reorder before sequence for some models.
        inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(Shape{N, T, DC}, dataType, memory::format_tag::tnc));
    else
        inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(Shape{N, T, DC}, dataType, memory::format_tag::ntc));

    // Initial hidden state.
    // WA to avoid reorder before.
    if (D == 1)
        inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(Shape{N, D, SC}, dataType, memory::format_tag::tnc));
    else
        inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(Shape{N, D, SC}, dataType, memory::format_tag::ntc));

    // initial cell state
    if (haveCellState(cell_type)) {
        if (D == 1)
            inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(Shape{N, D, SC}, memory::data_type::f32, memory::format_tag::tnc));
        else
            inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(Shape{N, D, SC}, memory::data_type::f32, memory::format_tag::ntc));
    }

    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(Shape{N}, memory::data_type::s32, memory::format_tag::x)); // sequence lengths
    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(Shape{D, G * SC, DC}, memory::data_type::f32, memory::format_tag::ntc)); // W
    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(Shape{D, G * SC, SC}, memory::data_type::f32, memory::format_tag::ntc)); // R
    inCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(Shape{D, Gb * SC}, memory::data_type::f32, memory::format_tag::nc)); // B

    std::vector<MemoryDescPtr> outCandidate;
    outCandidate.reserve(3);

    if (nativeOrder) {
        outCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(outDataDescs[RNNInOutKind::Layer]));
    } else if (N.isStatic() && N.getMaxValue() == 1) {
        // WA to avoid reorder after sequence for some models
        outCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(Shape{N, T, SC}, dataType, memory::format_tag::tnc));
    } else {
        outCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(Shape{N, T, SC}, dataType, memory::format_tag::ntc));
    }

    // WA to avoid reorder after
    if (D == 1)
        outCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(Shape{N, D, SC}, dataType, memory::format_tag::tnc));
    else
        outCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(Shape{N, D, SC}, dataType, memory::format_tag::ntc));

    if (haveCellState(cell_type)) {
        if (D == 1)
            outCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(Shape{N, D, SC}, memory::data_type::f32, memory::format_tag::tnc));
        else
            outCandidate.emplace_back(std::make_shared<DnnlBlockedMemoryDesc>(Shape{N, D, SC}, memory::data_type::f32, memory::format_tag::ntc));
    }

    createDescriptor(inCandidate, outCandidate);
}

bool MKLDNNRNN::verifyWeightsPrecision(const Precision &layerPrec, const Precision &weightsPrec) {
    if (!weightsByLayerPrec.count(layerPrec))
        THROW_ERROR << "has unsupported layer precision " << layerPrec;
    return weightsPrec == weightsByLayerPrec.at(layerPrec);
}

template <typename Prec>
void MKLDNNRNN::fillWeights(const int *gate_map, const size_t wIdx, const size_t rIdx) {
    const auto weightPrec = getOriginalInputPrecisionAtPort(wIdx);
    if (!verifyWeightsPrecision(runtimePrecision, weightPrec) && runtimePrecision != Precision::BF16 && weightPrec != Precision::FP32) {
        THROW_ERROR << "doesn't support combination of weights precision: " << weightPrec << " and runtime precision: " << runtimePrecision;
    }
    // create weight blobs (data and state part)
    const VectorDims dims_w = { L, D, DC, G, SC };
    TensorDesc w_data_desc(runtimePrecision, dims_w, getWeightsLayoutByDims(dims_w, false));
    Blob::Ptr w_data_mem = make_shared_blob<Prec>(w_data_desc);
    w_data_mem->allocate();
    auto w_ptr = static_cast<Prec*>(w_data_mem->buffer());
    if (w_ptr == nullptr)
        IE_THROW(NotAllocated) << "Internal blob was not allocated for node " << getName() << ".";

    const VectorDims dims_s = { L, D, SC, G, SC };
    TensorDesc w_state_desc(runtimePrecision, dims_s, getWeightsLayoutByDims(dims_s, false));
    Blob::Ptr w_state_mem = make_shared_blob<Prec>(w_state_desc);
    w_state_mem->allocate();
    auto r_ptr = static_cast<Prec*>(w_state_mem->buffer());
    if (r_ptr == nullptr)
        IE_THROW(NotAllocated) << "Internal blob was not allocated for node " << getName() << ".";

    const size_t ie_w_vec_size = getInputShapeAtPort(wIdx).getElementsCount();
    const size_t ie_r_vec_size = getInputShapeAtPort(rIdx).getElementsCount();

    auto *wInputNode = dynamic_cast<MKLDNNInputNode *>(getParentEdgesAtPort(wIdx)[0]->getParent().get());
    auto wConstBlob = wInputNode->getMemoryPtr();

    auto *rInputNode = dynamic_cast<MKLDNNInputNode *>(getParentEdgesAtPort(rIdx)[0]->getParent().get());
    auto rConstBlob = rInputNode->getMemoryPtr();

    std::vector<Prec> ie_w_vec(ie_w_vec_size), ie_r_vec(ie_r_vec_size);

    auto ie_w_ptr = ie_w_vec.data();
    auto ie_r_ptr = ie_r_vec.data();
    cpu_convert(wConstBlob->GetPtr(), ie_w_ptr, weightPrec, runtimePrecision, ie_w_vec_size);
    cpu_convert(rConstBlob->GetPtr(), ie_r_ptr, weightPrec, runtimePrecision, ie_r_vec_size);

    const int step = SC * G;

    for (int g = 0; g < G; g++) {
        for (int out_i = 0; out_i < SC; out_i++) {
            Prec *l_w_ptr = w_ptr + gate_map[g] * SC + out_i;
            for (int in_i = 0; in_i < DC; in_i++) {
                *l_w_ptr = *ie_w_ptr;
                ie_w_ptr++;
                l_w_ptr += step;
            }

            Prec *l_r_ptr = r_ptr + gate_map[g] * SC + out_i;
            for (int in_i = 0; in_i < SC; in_i++) {
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
void MKLDNNRNN::fillBiases(const int *gate_map) {
    using dataType = typename PrecisionTrait<Prec>::value_type;

    if (getOriginalInputPrecisionAtPort(bIdx) != Precision::FP32) {
        THROW_ERROR << "doesn't support bias precision: " << getOriginalInputPrecisionAtPort(bIdx);
    }

    VectorDims dims_b = { L, D, Gb, SC };
    TensorDesc w_bias_data_desc(Prec, dims_b, getWeightsLayoutByDims(dims_b, false));
    Blob::Ptr w_bias_data_mem = make_shared_blob<dataType>(w_bias_data_desc);
    w_bias_data_mem->allocate();
    auto b_ptr = static_cast<dataType*>(w_bias_data_mem->buffer());
    if (b_ptr == nullptr)
        IE_THROW(NotAllocated) << "Internal blob was not allocated for node " << getName() << ".";

    auto *constInputNode = dynamic_cast<MKLDNNInputNode *>(getParentEdgesAtPort(bIdx)[0]->getParent().get());
    auto constBlob = constInputNode->getMemoryPtr();
    auto const elementsCount = constBlob->GetSize() / constBlob->getDesc().getPrecision().size();

    std::vector<dataType> ie_b_vec(elementsCount);
    cpu_convert(constBlob->GetPtr(),
                &ie_b_vec[0],
                MKLDNNExtensionUtils::DataTypeToIEPrecision(constBlob->GetDataType()),
                Prec,
                elementsCount);

    for (int g = 0; g < Gb; g++) {
        dataType *l_b_ptr = b_ptr + gate_map[g] * SC;
        const dataType *l_ie_b_ptr = &ie_b_vec[g * SC];
        cpu_memcpy(l_b_ptr, l_ie_b_ptr, SC * sizeof(typename PrecisionTrait<Prec>::value_type));
    }
    internalBlobs.push_back(w_bias_data_mem);
}

void MKLDNNRNN::copyWeightsData() {
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
     *   IE    - FICO, mkldnn - IFCO
     *
     *   ====== GRU ======
     *   IE - URO, mkldnn - URO
     */
    const int gate_map_lstm[] = {1, 0, 2, 3};  // FICO -> IFCO
    const int gate_map_gru[]  = {0, 1, 2, 3};
    const int gate_map_rnn[]  = {0};
    const int *gate_map;
    const int gate_map_lstm_size = sizeof(gate_map_lstm) / sizeof(int);
    const int gate_map_gru_size = sizeof(gate_map_gru) / sizeof(int);
    const int gate_map_rnn_size = sizeof(gate_map_rnn) / sizeof(int);
    if (cell_type == mkldnn::algorithm::vanilla_lstm) {
        gate_map = gate_map_lstm;
        if (G > gate_map_lstm_size) {
            THROW_ERROR << ". G isn't equal to the size of gate_map.";
        }
    } else if (cell_type == mkldnn::algorithm::vanilla_gru) {
        gate_map = gate_map_gru;
        if (G > gate_map_gru_size) {
            THROW_ERROR << ". G isn't equal to the size of gate_map";
        }
    } else if (cell_type == mkldnn::algorithm::lbr_gru) {
        gate_map = gate_map_gru;
        if (G > gate_map_gru_size) {
            THROW_ERROR << ". G isn't equal to the size of gate_map.";
        }
    } else if (cell_type == mkldnn::algorithm::vanilla_rnn) {
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

    if (runtimePrecision == Precision::BF16) {
        fillWeights<uint16_t>(gate_map, wIdx, rIdx);
    } else if (runtimePrecision == Precision::FP32) {
        // WA To avoid different weights layer and iter formats in FP32 case
        if ((T.isStatic() && T.getMaxValue() != 1) || (N.isStatic() && N.getMaxValue() < optimalBatchSize))
            wFormat = mkldnn::memory::format_tag::ldigo;
        fillWeights<float>(gate_map, wIdx, rIdx);
    } else {// TODO FP16 and INT8 support
        THROW_ERROR << "has unsupported data type: " << runtimePrecision;
    }

    if (runtimePrecision == Precision::BF16 || runtimePrecision == Precision::FP32)
        fillBiases<Precision::FP32>(gate_map);
}

void MKLDNNRNN::createDescriptor(const std::vector<MemoryDescPtr> &inputDesc,
                                 const std::vector<MemoryDescPtr> &outputDesc) {
std::cout << "MKLDNNRNN::createDescriptor() +" << std::endl;
    if (!descs.empty())
        return;
    wDescs.resize(3);
    auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(runtimePrecision);
    auto weightsDims = MKLDNNExtensionUtils::convertToDnnlDims(VectorDims{ L, D, DC, G, SC });
    wDescs[0] = mkldnn::memory::desc(weightsDims, dataType, wFormat);
    auto statesDims = MKLDNNExtensionUtils::convertToDnnlDims(VectorDims{ L, D, SC, G, SC });
    wDescs[1] = mkldnn::memory::desc(statesDims, dataType, wFormat);
    auto biasDims = MKLDNNExtensionUtils::convertToDnnlDims(VectorDims{ L, D, Gb, SC });
    wDescs[2] = mkldnn::memory::desc(biasDims, memory::data_type::f32, memory::format_tag::ldgo);

    switch (cell_type) {
        case mkldnn::algorithm::vanilla_rnn: {
            MKLDNNDescriptor desc(std::shared_ptr<vanilla_rnn_forward::desc>(
                    new vanilla_rnn_forward::desc(prop_kind::forward_scoring, cell_act, direction,
                            /* In Data       */ inDataDescs[RNNInOutKind::Layer].getDnnlDesc(),
                            /* In State      */ inDataDescs[RNNInOutKind::HiddenState].getDnnlDesc(),
                            /* Weights data  */ wDescs[0],
                            /* Weights state */ wDescs[1],
                            /* Bias          */ wDescs[2],
                            /* Out Data      */ outDataDescs[RNNInOutKind::Layer].getDnnlDesc(),
                            /* Out State     */ outDataDescs[RNNInOutKind::HiddenState].getDnnlDesc())));
            descs.push_back(desc);
        } break;
        case mkldnn::algorithm::vanilla_gru: {
            MKLDNNDescriptor desc(std::shared_ptr<gru_forward::desc>(
                    new gru_forward::desc(prop_kind::forward_scoring, direction,
                            /* In Data       */ inDataDescs[RNNInOutKind::Layer].getDnnlDesc(),
                            /* In State      */ inDataDescs[RNNInOutKind::HiddenState].getDnnlDesc(),
                            /* Weights data  */ wDescs[0],
                            /* Weights state */ wDescs[1],
                            /* Bias          */ wDescs[2],
                            /* Out Data      */ outDataDescs[RNNInOutKind::Layer].getDnnlDesc(),
                            /* Out State     */ outDataDescs[RNNInOutKind::HiddenState].getDnnlDesc())));
            descs.push_back(desc);
        } break;
        case mkldnn::algorithm::lbr_gru: {
            MKLDNNDescriptor desc(std::shared_ptr<lbr_gru_forward::desc>(
                    new lbr_gru_forward::desc(prop_kind::forward_scoring, direction,
                            /* In Data       */ inDataDescs[RNNInOutKind::Layer].getDnnlDesc(),
                            /* In State      */ inDataDescs[RNNInOutKind::HiddenState].getDnnlDesc(),
                            /* Weights data  */ wDescs[0],
                            /* Weights state */ wDescs[1],
                            /* Bias          */ wDescs[2],
                            /* Out Data      */ outDataDescs[RNNInOutKind::Layer].getDnnlDesc(),
                            /* Out State     */ outDataDescs[RNNInOutKind::HiddenState].getDnnlDesc())));
            descs.push_back(desc);
        } break;
        case mkldnn::algorithm::vanilla_lstm: {
            MKLDNNDescriptor desc(std::shared_ptr<lstm_forward::desc>(
                    new lstm_forward::desc(prop_kind::forward_scoring, direction,
                            /* In Data       */ inDataDescs[RNNInOutKind::Layer].getDnnlDesc(),
                            /* In State      */ inDataDescs[RNNInOutKind::HiddenState].getDnnlDesc(),
                            /* In State C    */ inDataDescs[RNNInOutKind::CellState].getDnnlDesc(),
                            /* Weights data  */ wDescs[0],
                            /* Weights state */ wDescs[1],
                            /* Bias          */ wDescs[2],
                            /* Out Data      */ outDataDescs[RNNInOutKind::Layer].getDnnlDesc(),
                            /* Out State     */ outDataDescs[RNNInOutKind::HiddenState].getDnnlDesc(),
                            /* Out State C   */ outDataDescs[RNNInOutKind::CellState].getDnnlDesc())));
            descs.push_back(desc);
        } break;
        default:
            THROW_ERROR << "has unknown cell type.";
    }

    // Fill supported config
    NodeConfig config;
    config.dynBatchSupport = false;
    for (size_t i = 0; i < inputDesc.size(); i++) {
        PortConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;
        dataConfig.desc = inputDesc[i];
        config.inConfs.push_back(dataConfig);
    }

    for (size_t i = 0; i < outputDesc.size(); i++) {
        PortConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;
        dataConfig.desc = outputDesc[i];
        config.outConfs.push_back(dataConfig);
    }

    supportedPrimitiveDescriptors.emplace_back(config, ref_any);
std::cout << "MKLDNNRNN::createDescriptor() -" << std::endl;
}

void MKLDNNRNN::createPrimitive() {
std::cout << "MKLDNNRNN::createPrimitive() +" << std::endl;
    if (!isDynamicNode()) {
        if (cell_type == mkldnn::algorithm::vanilla_rnn) {
            auto prim_desc = createPrimitiveDescriptor<vanilla_rnn_forward::primitive_desc, vanilla_rnn_forward::desc>();
            prim.reset(new vanilla_rnn_forward(prim_desc));
        } else if (cell_type == mkldnn::algorithm::vanilla_gru) {
            auto prim_desc = createPrimitiveDescriptor<gru_forward::primitive_desc, gru_forward::desc>();
            prim.reset(new gru_forward(prim_desc));
        } else if (cell_type == mkldnn::algorithm::lbr_gru) {
            auto prim_desc = createPrimitiveDescriptor<lbr_gru_forward::primitive_desc, lbr_gru_forward::desc>();
            prim.reset(new lbr_gru_forward(prim_desc));
        } else if (cell_type == mkldnn::algorithm::vanilla_lstm) {
            auto prim_desc = createPrimitiveDescriptor<lstm_forward::primitive_desc, lstm_forward::desc>();
            prim.reset(new lstm_forward(prim_desc));
        } else {
            THROW_ERROR << "has unknown cell type.";
        }
    }

    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
std::cout << "MKLDNNRNN::createPrimitive() -" << std::endl;
}

void MKLDNNRNN::prepareParams() {
std::cout << "MKLDNNRNN::prepareParams() +" << std::endl;
    const auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(runtimePrecision);

    const size_t B = getParentEdgesAtPort(0).front()->getMemory().GetShape().getStaticDims()[0];
    const size_t SL = is_cell ? 1lu : getParentEdgesAtPort(0).front()->getMemory().GetShape().getStaticDims()[1];
    Shape shapeS_4D{L, D, B, SC};
    std::vector<DnnlBlockedMemoryDesc> inDataD, outDataD;

    inDataD.reserve(S + 1);
    outDataD.reserve(S + 1);

    inDataD.emplace_back(Shape{SL, B, DC}, dataType, memory::format_tag::tnc);
    outDataD.emplace_back(Shape{SL, B, SC}, dataType, memory::format_tag::tnc);

    inDataD.emplace_back(shapeS_4D, dataType, memory::format_tag::ldnc);
    outDataD.emplace_back(shapeS_4D, dataType, memory::format_tag::ldnc);

    if (haveCellState(cell_type)) {
        inDataD.emplace_back(shapeS_4D, memory::data_type::f32, memory::format_tag::ldnc);
        outDataD.emplace_back(shapeS_4D, memory::data_type::f32, memory::format_tag::ldnc);
    }

    bool wFormatWasChanged = false;
    // WA To avoid different weights layer and iter formats in FP32 case.
    if (runtimePrecision == Precision::FP32) {
        if (SL != 1 || B < optimalBatchSize) {
            if (wFormat != mkldnn::memory::format_tag::ldigo) {
                wFormat = mkldnn::memory::format_tag::ldigo;
                wFormatWasChanged = true;
            }
        } else if (wFormat != mkldnn::memory::format_tag::any) {
            wFormat = mkldnn::memory::format_tag::any;
            wFormatWasChanged = true;
        }
    }
    if (wFormatWasChanged) {
        auto weightsDims = MKLDNNExtensionUtils::convertToDnnlDims(VectorDims{ L, D, DC, G, SC });
        wDescs[0] = mkldnn::memory::desc(weightsDims, dataType, wFormat);
        auto statesDims = MKLDNNExtensionUtils::convertToDnnlDims(VectorDims{ L, D, SC, G, SC });
        wDescs[1] = mkldnn::memory::desc(statesDims, dataType, wFormat);
    }

    primitive_desc_iterator itpd;
    const mkldnn::primitive_attr attr = mkldnn::primitive_attr();
    if (cell_type == mkldnn::algorithm::vanilla_rnn) {
        auto desc = std::make_shared<vanilla_rnn_forward::desc>(
                                            prop_kind::forward_scoring,
                                            cell_act,
                                            direction,
                        /* In Data       */ inDataD[RNNInOutKind::Layer].getDnnlDesc(),
                        /* In State      */ inDataD[RNNInOutKind::HiddenState].getDnnlDesc(),
                        /* Weights data  */ wDescs[0],
                        /* Weights state */ wDescs[1],
                        /* Bias          */ wDescs[2],
                        /* Out Data      */ outDataD[RNNInOutKind::Layer].getDnnlDesc(),
                        /* Out State     */ outDataD[RNNInOutKind::HiddenState].getDnnlDesc());
        prim.reset(new vanilla_rnn_forward(vanilla_rnn_forward::primitive_desc(*desc, getEngine())));
        itpd = mkldnn::primitive_desc_iterator(&desc->data, &attr, getEngine(), nullptr, true);
    } else if (cell_type == mkldnn::algorithm::vanilla_gru) {
        auto desc = std::make_shared<gru_forward::desc>(
                                            prop_kind::forward_scoring,
                                            direction,
                        /* In Data       */ inDataD[RNNInOutKind::Layer].getDnnlDesc(),
                        /* In State      */ inDataD[RNNInOutKind::HiddenState].getDnnlDesc(),
                        /* Weights data  */ wDescs[0],
                        /* Weights state */ wDescs[1],
                        /* Bias          */ wDescs[2],
                        /* Out Data      */ outDataD[RNNInOutKind::Layer].getDnnlDesc(),
                        /* Out State     */ outDataD[RNNInOutKind::HiddenState].getDnnlDesc());
        prim.reset(new gru_forward(gru_forward::primitive_desc(*desc, getEngine())));
        itpd = mkldnn::primitive_desc_iterator(&desc->data, &attr, getEngine(), nullptr, true);
    } else if (cell_type == mkldnn::algorithm::lbr_gru) {
        auto desc = std::make_shared<lbr_gru_forward::desc>(
                                                prop_kind::forward_scoring,
                                                direction,
                            /* In Data       */ inDataD[RNNInOutKind::Layer].getDnnlDesc(),
                            /* In State      */ inDataD[RNNInOutKind::HiddenState].getDnnlDesc(),
                            /* Weights data  */ wDescs[0],
                            /* Weights state */ wDescs[1],
                            /* Bias          */ wDescs[2],
                            /* Out Data      */ outDataD[RNNInOutKind::Layer].getDnnlDesc(),
                            /* Out State     */ outDataD[RNNInOutKind::HiddenState].getDnnlDesc());
        prim.reset(new lbr_gru_forward(lbr_gru_forward::primitive_desc(*desc, getEngine())));
        itpd = mkldnn::primitive_desc_iterator(&desc->data, &attr, getEngine(), nullptr, true);
    } else if (cell_type == mkldnn::algorithm::vanilla_lstm) {
        auto desc = std::make_shared<lstm_forward::desc>(
                                                prop_kind::forward_scoring,
                                                direction,
                            /* In Data       */ inDataD[RNNInOutKind::Layer].getDnnlDesc(),
                            /* In State      */ inDataD[RNNInOutKind::HiddenState].getDnnlDesc(),
                            /* In State C    */ inDataD[RNNInOutKind::CellState].getDnnlDesc(),
                            /* Weights data  */ wDescs[0],
                            /* Weights state */ wDescs[1],
                            /* Bias          */ wDescs[2],
                            /* Out Data      */ outDataD[RNNInOutKind::Layer].getDnnlDesc(),
                            /* Out State     */ outDataD[RNNInOutKind::HiddenState].getDnnlDesc(),
                            /* Out State C   */ outDataD[RNNInOutKind::CellState].getDnnlDesc());
        prim.reset(new lstm_forward(lstm_forward::primitive_desc(*desc, getEngine())));
        itpd = mkldnn::primitive_desc_iterator(&desc->data, &attr, getEngine(), nullptr, true);
    }

    if (wFormatWasChanged) {
        const NodeDesc* selectedPd = getSelectedPrimitiveDescriptor();
        if (selectedPd == nullptr)
            THROW_ERROR << "does not have preferable primitive descriptor.";

        prepareMemory(selectedPd, itpd);
    }
std::cout << "MKLDNNRNN::prepareParams() -" << std::endl;
}

std::shared_ptr<MemoryDesc> MKLDNNRNN::getSrcMemDesc(mkldnn::primitive_desc_iterator& primitive_desc_it, size_t idx) {
    auto desc = supportedPrimitiveDescriptors[0].getConfig().inConfs[idx].desc;
    return desc->as<BlockedMemoryDesc>()->cloneWithUndefStridesAndOffset();
}

std::shared_ptr<MemoryDesc> MKLDNNRNN::getDstMemDesc(mkldnn::primitive_desc_iterator& primitive_desc_it, size_t idx) {
    auto desc = supportedPrimitiveDescriptors[0].getConfig().outConfs[idx].desc;
    return desc->as<BlockedMemoryDesc>()->cloneWithUndefStridesAndOffset();
}

void MKLDNNRNN::execute(mkldnn::stream strm) {
std::cout << "MKLDNNRNN::execute() +" << std::endl;
    if (!prim)
        THROW_ERROR << "does not have initialized primitive to execute.";

    const auto src_data_mem = getParentEdgeAt(0)->getMemoryPtr();
    const auto dst_data_mem = getChildEdgeAt(0)->getMemoryPtr();
std::cout << "MKLDNNRNN::execute() 1" << std::endl;

    const auto &wgh_data_mem = internalBlobMemory[0];
    const auto &wgh_stat_mem = internalBlobMemory[1];
    const auto &wgh_bias_mem = internalBlobMemory[2];
std::cout << "MKLDNNRNN::execute() 2; src_data_mem: " << src_data_mem << "; wgh_data_mem: " << wgh_data_mem <<
        "; wgh_stat_mem: " << wgh_stat_mem << "; wgh_bias_mem: " << wgh_bias_mem << "; dst_data_mem: " << dst_data_mem << std::endl;

    std::unordered_map<int, memory> args {
        {DNNL_ARG_SRC_LAYER,     src_data_mem->GetPrimitive()},
        {DNNL_ARG_WEIGHTS_LAYER, wgh_data_mem->GetPrimitive()},
        {DNNL_ARG_WEIGHTS_ITER,  wgh_stat_mem->GetPrimitive()},
        {DNNL_ARG_BIAS,          wgh_bias_mem->GetPrimitive()},
        {DNNL_ARG_DST_LAYER,     dst_data_mem->GetPrimitive()},
    };
std::cout << "MKLDNNRNN::execute() 3" << std::endl;

    int state_i_tags[] {DNNL_ARG_SRC_ITER, DNNL_ARG_SRC_ITER_C};
    int state_o_tags[] {DNNL_ARG_DST_ITER, DNNL_ARG_DST_ITER_C};
    for (size_t s = 0; s < S; s++) {
        args[state_i_tags[s]] = getParentEdgeAt(s+1)->getMemoryPtr()->GetPrimitive();
    }
std::cout << "MKLDNNRNN::execute() 4" << std::endl;

    if (is_cell) {
        for (size_t s = 0; s < S; s++) {
            args[state_o_tags[s]] = getChildEdgesAtPort(s)[0]->getMemoryPtr()->GetPrimitive();
        }
    } else {
        size_t n_ports_with_init_states = outputShapes.size() - 1; // first is a sequence data
        for (size_t s = 0; s < std::min(S, n_ports_with_init_states); s++) {
            if (s < outputShapes.size()) {
                args[state_o_tags[s]] = getChildEdgesAtPort(s+1)[0]->getMemoryPtr()->GetPrimitive();
            }
        }
    }

std::cout << "MKLDNNRNN::execute() 5" << std::endl;
    (*prim).execute(strm, args);
std::cout << "MKLDNNRNN::execute() -" << std::endl;
}

void MKLDNNRNN::executeDynamicImpl(mkldnn::stream strm) {
    execute(strm);
}

std::vector<VectorDims> MKLDNNRNN::shapeInfer() const {
    auto originOutputShapes = MKLDNNNode::shapeInfer();

    if (!hasNativeOrder() && originOutputShapes[0].size() == 4lu && originOutputShapes[0][1] == 1lu) {
        originOutputShapes[0].erase(originOutputShapes[0].begin() + 1);
    }
    return originOutputShapes;
}
}  // namespace MKLDNNPlugin

REG_MKLDNN_PRIM_FOR(MKLDNNRNN, RNNCell);
REG_MKLDNN_PRIM_FOR(MKLDNNRNN, RNNSeq);
