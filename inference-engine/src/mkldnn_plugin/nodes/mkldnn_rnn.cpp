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

#include <ngraph/node.hpp>

#include <string>
#include <utility>

using namespace mkldnn;
using namespace InferenceEngine;

namespace MKLDNNPlugin {

static rnn_direction ieDirection2dnnl(const std::shared_ptr<const ngraph::Node>& op) {
    ngraph::op::RecurrentSequenceDirection direction = ngraph::op::RecurrentSequenceDirection::FORWARD;
    if (op->get_type_info() == ngraph::op::v5::GRUSequence::type_info) {
        direction = ngraph::as_type_ptr<const ngraph::op::v5::GRUSequence>(op)->get_direction();
    } else if (op->get_type_info() == ngraph::op::v0::LSTMSequence::type_info) {
        direction = ngraph::as_type_ptr<const ngraph::op::v0::LSTMSequence>(op)->get_direction();
    } else if (op->get_type_info() == ngraph::op::v5::LSTMSequence::type_info) {
        direction = ngraph::as_type_ptr<const ngraph::op::v5::LSTMSequence>(op)->get_direction();
    } else if (op->get_type_info() == ngraph::op::v5::RNNSequence::type_info) {
        direction = ngraph::as_type_ptr<const ngraph::op::v5::RNNSequence>(op)->get_direction();
    }
    return direction == ngraph::op::RecurrentSequenceDirection::FORWARD ? rnn_direction::unidirectional_left2right
         : direction == ngraph::op::RecurrentSequenceDirection::REVERSE ? rnn_direction::unidirectional_right2left
         : direction == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL ? rnn_direction::bidirectional_concat
         : rnn_direction::unidirectional;
}

static mkldnn::algorithm ie2dnnl(std::string act_type) {
    return act_type == "sigmoid" ? mkldnn::algorithm::eltwise_logistic
         : act_type == "tanh"    ? mkldnn::algorithm::eltwise_tanh
         : act_type == "relu"    ? mkldnn::algorithm::eltwise_relu
         : mkldnn::algorithm::undef;
}

static mkldnn::algorithm ie2dnnl(const std::shared_ptr<const ngraph::Node>& op) {
    if (one_of(op->get_type_info(),
            ngraph::op::v3::GRUCell::type_info,
            ngraph::op::v5::GRUSequence::type_info)) {
        auto gruCellOp = ngraph::as_type_ptr<const ngraph::op::v3::GRUCell>(op);
        auto gruSeqOp = ngraph::as_type_ptr<const ngraph::op::v5::GRUSequence>(op);
        if ((gruCellOp && gruCellOp->get_linear_before_reset()) ||
                (gruSeqOp && gruSeqOp->get_linear_before_reset()))
            return mkldnn::algorithm::lbr_gru;
        else
            return mkldnn::algorithm::vanilla_gru;
    } else if (one_of(op->get_type_info(),
            ngraph::op::v0::LSTMCell::type_info,
            ngraph::op::v4::LSTMCell::type_info,
            ngraph::op::v0::LSTMSequence::type_info,
            ngraph::op::v5::LSTMSequence::type_info)) {
        return mkldnn::algorithm::vanilla_lstm;
    } else if (one_of(op->get_type_info(),
            ngraph::op::v0::RNNCell::type_info,
            ngraph::op::v5::RNNSequence::type_info)) {
        return mkldnn::algorithm::vanilla_rnn;
    } else {
        IE_THROW() << "Unsupported cell type";
    }
}

size_t gatesCount(mkldnn::algorithm alg) {
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

size_t statesCount(mkldnn::algorithm alg) {
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

bool haveCellState(mkldnn::algorithm alg) {
    return alg == mkldnn::algorithm::vanilla_lstm;
}

const std::map<InferenceEngine::Precision, InferenceEngine::Precision> MKLDNNRNN::weightsByLayerPrec {
    // layer precision,                weights precision
    {InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP32},
    {InferenceEngine::Precision::BF16, InferenceEngine::Precision::BF16},
    // FP16 and U8 are not supported yet
    // {InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP16},
    // {InferenceEngine::Precision::U8,   InferenceEngine::Precision::I8},
};

bool MKLDNNRNN::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                ngraph::op::v3::GRUCell::type_info,
                ngraph::op::v0::LSTMCell::type_info,
                ngraph::op::v4::LSTMCell::type_info,
                ngraph::op::v0::RNNCell::type_info,
                ngraph::op::v5::GRUSequence::type_info,
                ngraph::op::v0::LSTMSequence::type_info,
                ngraph::op::v5::LSTMSequence::type_info,
                ngraph::op::v5::RNNSequence::type_info)) {
            errorMessage = "Unsupported RNN operation.";
            return false;
        }

        if (one_of(op->get_type_info(), ngraph::op::v0::RNNCell::type_info, ngraph::op::v3::GRUCell::type_info)) {
            if (op->get_input_size() != 5) {
                errorMessage = "Node expects 5 inputs. Actual: " + std::to_string(op->get_input_size());
                return false;
            }
            if (op->get_input_node_ptr(2)->get_type_info() != ngraph::op::v0::Constant::type_info ||
                    op->get_input_node_ptr(3)->get_type_info() != ngraph::op::v0::Constant::type_info ||
                    op->get_input_node_ptr(4)->get_type_info() != ngraph::op::v0::Constant::type_info) {
                errorMessage = "Node expects constants as W, R, B inputs.";
                return false;
            }
        } else if (one_of(op->get_type_info(),
                ngraph::op::v0::LSTMCell::type_info,
                ngraph::op::v4::LSTMCell::type_info,
                ngraph::op::v5::GRUSequence::type_info,
                ngraph::op::v5::RNNSequence::type_info)) {
            if (op->get_input_size() != 6) {
                errorMessage = "Node expects 6 inputs. Actual: " + std::to_string(op->get_input_size());
                return false;
            }
            if (op->get_input_node_ptr(3)->get_type_info() != ngraph::op::v0::Constant::type_info ||
                    op->get_input_node_ptr(4)->get_type_info() != ngraph::op::v0::Constant::type_info ||
                    op->get_input_node_ptr(5)->get_type_info() != ngraph::op::v0::Constant::type_info) {
                errorMessage = "Node expects constants as W, R, B inputs.";
                return false;
            }
        } else if (one_of(op->get_type_info(),
                ngraph::op::v0::LSTMSequence::type_info,
                ngraph::op::v5::LSTMSequence::type_info)) {
            if (op->get_input_size() != 7) {
                errorMessage = "Node expects 7 inputs. Actual: " + std::to_string(op->get_input_size());
                return false;
            }
            if (op->get_input_node_ptr(4)->get_type_info() != ngraph::op::v0::Constant::type_info ||
                    op->get_input_node_ptr(5)->get_type_info() != ngraph::op::v0::Constant::type_info ||
                    op->get_input_node_ptr(6)->get_type_info() != ngraph::op::v0::Constant::type_info) {
                errorMessage = "Node expects constants as W, R, B inputs.";
                return false;
            }
        }

        auto rnnCellBase = std::dynamic_pointer_cast<const ngraph::op::util::RNNCellBase>(op);
        if (rnnCellBase && rnnCellBase->get_clip() != 0.0f) {
            errorMessage = "Clipping is not supported for RNN primitive.";
            return false;
        }

        ngraph::op::RecurrentSequenceDirection direction = ngraph::op::RecurrentSequenceDirection::FORWARD;
        if (op->get_type_info() == ngraph::op::v5::GRUSequence::type_info) {
            direction = ngraph::as_type_ptr<const ngraph::op::v5::GRUSequence>(op)->get_direction();
        } else if (op->get_type_info() == ngraph::op::v0::LSTMSequence::type_info) {
            direction = ngraph::as_type_ptr<const ngraph::op::v0::LSTMSequence>(op)->get_direction();
        } else if (op->get_type_info() == ngraph::op::v5::LSTMSequence::type_info) {
            direction = ngraph::as_type_ptr<const ngraph::op::v5::LSTMSequence>(op)->get_direction();
        } else if (op->get_type_info() == ngraph::op::v5::RNNSequence::type_info) {
            direction = ngraph::as_type_ptr<const ngraph::op::v5::RNNSequence>(op)->get_direction();
        }
        if (!one_of(direction, ngraph::op::RecurrentSequenceDirection::FORWARD, ngraph::op::RecurrentSequenceDirection::REVERSE)) {
            errorMessage = "Unsupported sequence direction.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNRNN::MKLDNNRNN(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    is_cell = one_of(op->get_type_info(),
            ngraph::op::v0::RNNCell::type_info,
            ngraph::op::v3::GRUCell::type_info,
            ngraph::op::v0::LSTMCell::type_info,
            ngraph::op::v4::LSTMCell::type_info);

    if (one_of(op->get_type_info(),
               ngraph::op::v0::RNNCell::type_info,
               ngraph::op::v3::GRUCell::type_info)) {
        wIdx = 2; rIdx = 3; bIdx = 4;
    } else if (one_of(op->get_type_info(),
                      ngraph::op::v5::RNNSequence::type_info,
                      ngraph::op::v0::LSTMCell::type_info,
                      ngraph::op::v4::LSTMCell::type_info,
                      ngraph::op::v5::GRUSequence::type_info)) {
        wIdx = 3; rIdx = 4; bIdx = 5;
    } else if (one_of(op->get_type_info(),
                      ngraph::op::v0::LSTMSequence::type_info,
                      ngraph::op::v5::LSTMSequence::type_info)) {
        wIdx = 4; rIdx = 5; bIdx = 6;
    }

    if (is_cell)
        initCell(op);
    else
        initSeq(op);
}

bool MKLDNNRNN::created() const {
    return getType() == (is_cell ? RNNCell : RNNSeq);
}

void MKLDNNRNN::getSupportedDescriptors() {
    if (is_cell)
        fillCellDesc();
    else
        fillSeqDesc();
}

void MKLDNNRNN::initCell(const std::shared_ptr<ngraph::Node>& op) {
    auto rnnCellBase = std::dynamic_pointer_cast<ngraph::op::util::RNNCellBase>(op);
    if (!rnnCellBase)
        IE_THROW() << "No original layer for RNNCell.";

    cell_type = ie2dnnl(op);
    cell_act = ie2dnnl(rnnCellBase->get_activations()[0]);  // Works only for RNN with one gate

    auto in_data_dims = op->get_input_shape(0);
    auto in_h_state_dims = op->get_input_shape(1);
    auto out_h_state_dims = op->get_output_shape(0);

    if (in_data_dims.size() != 2 || in_h_state_dims.size() != 2)
        IE_THROW() << "Incorrect shape of input/output ports for layer " << getName();

    G = gatesCount(cell_type);
    S = statesCount(cell_type);
    T = 1;
    N  = in_data_dims[0];
    DC = in_data_dims[1];
    SC = in_h_state_dims[1];

    Gb = (cell_type != mkldnn::algorithm::lbr_gru) ? G : G + 1;

    // Expected shapes
    MKLDNNDims D_shape {N, DC}, S_shape {N, SC}, S_4D_shape {L, D, N, SC};

    if (in_data_dims != D_shape.ToSizeVector()
        || in_h_state_dims != S_shape.ToSizeVector()
        || out_h_state_dims != S_shape.ToSizeVector())
        IE_THROW() << "Incorrect shape of input/output ports for layer " << getName();

    if (S == 2) {
        auto in_c_state_dims = op->get_input_shape(2);
        auto out_c_state_dims = op->get_output_shape(1);

        if (in_c_state_dims != S_shape.ToSizeVector()
            || out_c_state_dims != S_shape.ToSizeVector())
            IE_THROW() << "Incorrect shape of input/output ports for layer " << getName();
    }
}

void MKLDNNRNN::fillCellDesc() {
    runtimePrecision = getOriginalInputPrecisionAtPort(0);
    auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(runtimePrecision);

    MKLDNNDims S_4D_shape {L, D, N, SC};

    // layer input plus states
    in_data_d.resize(S + 1);
    out_data_d.resize(S + 1);

    // Shapes and Attributes are correct. Can start internal stuff initialization.
    in_data_d[RNNInOutKind::Layer]  = {MKLDNNDims{T, N, DC}, dataType, memory::format_tag::tnc};
    out_data_d[RNNInOutKind::Layer] = {MKLDNNDims{T, N, SC}, dataType, memory::format_tag::tnc};

    in_data_d[RNNInOutKind::HiddenState]  = {S_4D_shape, dataType, memory::format_tag::ldnc};
    out_data_d[RNNInOutKind::HiddenState] = {S_4D_shape, dataType, memory::format_tag::ldnc};

    if (haveCellState(cell_type)) {
        in_data_d[RNNInOutKind::CellState] =  {S_4D_shape, memory::data_type::f32, memory::format_tag::ldnc};
        out_data_d[RNNInOutKind::CellState] = {S_4D_shape, memory::data_type::f32, memory::format_tag::ldnc};
    }

    w_data_d   = {{L, D, DC, G, SC}, dataType, memory::format_tag::ldigo};
    w_state_d  = {{L, D, SC, G, SC}, dataType, memory::format_tag::ldigo};

    // Add 5th input
    w_bias_d = {{L, D, Gb, SC}, memory::data_type::f32, memory::format_tag::ldgo};

    copyWeightsData();

    // Expected shapes
    MKLDNNDims D_shape {N, DC}, S_shape {N, SC}, WShape {SC * G, DC}, RShape {SC * G, SC}, BShape {SC * Gb};
    std::vector<TensorDesc> in_candidate, out_candidate;
    in_candidate.reserve(6);

    in_candidate.emplace_back(MKLDNNMemoryDesc {D_shape, dataType, memory::format_tag::nc});
    in_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, dataType, memory::format_tag::nc});
    out_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, dataType, memory::format_tag::nc});

    if (haveCellState(cell_type)) {
        in_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::data_type::f32, memory::format_tag::nc});
        out_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::data_type::f32, memory::format_tag::nc});
    }
    if (one_of(cell_type, mkldnn::algorithm::vanilla_rnn, mkldnn::algorithm::vanilla_gru, mkldnn::algorithm::lbr_gru, mkldnn::algorithm::vanilla_lstm)) {
        in_candidate.emplace_back(MKLDNNMemoryDesc {WShape, memory::data_type::f32, memory::format_tag::nc});
        in_candidate.emplace_back(MKLDNNMemoryDesc {RShape, memory::data_type::f32, memory::format_tag::nc});
        in_candidate.emplace_back(MKLDNNMemoryDesc {BShape, memory::data_type::f32, memory::format_tag::x});
    }

    createDescriptor(in_candidate, out_candidate);
}

void MKLDNNRNN::initSeq(const std::shared_ptr<ngraph::Node>& op) {
    auto rnnCellBase = std::dynamic_pointer_cast<ngraph::op::util::RNNCellBase>(op);
    if (!rnnCellBase)
        IE_THROW() << "No original layer for RNNCell.";

    cell_type = ie2dnnl(op);
    cell_act = mkldnn::algorithm::undef;
    if (!rnnCellBase->get_activations().empty())
        cell_act = ie2dnnl(rnnCellBase->get_activations()[0]);  // Works only for RNN with one gate

    direction = ieDirection2dnnl(op);

    if (!one_of(op->get_input_size(), 6, 7))
        IE_THROW() << "Incorrect number of input ports for layer " << getName();
    if (!one_of(op->get_output_size(), 2, 3))
        IE_THROW() << "Incorrect number of output ports for layer " << getName();

    in_data_dims = op->get_input_shape(0);
    out_data_dims = op->get_output_shape(0);

    if (in_data_dims.size() != 3 || out_data_dims.size() != 4)
        IE_THROW() << "Incorrect shape of input/output ports for layer " << getName();

    N = op->get_input_shape(1)[0];
    nativeOrder = false;
    const auto rtInfo = op->get_rt_info();

    if (rtInfo.count("seqAxis")) {
        nativeOrder = std::dynamic_pointer_cast<ngraph::VariantWrapper<int64_t>>(rtInfo.at("seqAxis"))->get() == 0;
    }
    out_data_dims.erase(out_data_dims.begin() + 1);

    std::swap(in_data_dims[0], in_data_dims[1]);
    std::swap(out_data_dims[0], out_data_dims[1]);

    G = gatesCount(cell_type);
    S = statesCount(cell_type);
    T = in_data_dims[0];
    DC = in_data_dims[2];
    SC = rnnCellBase->get_hidden_size();

    Gb = (cell_type != mkldnn::algorithm::lbr_gru) ? G : G + 1;

    // layer input plus states
    in_data_d.resize(S + 1);
    out_data_d.resize(S + 1);
}

void MKLDNNRNN::fillSeqDesc() {
    runtimePrecision = getOriginalInputPrecisionAtPort(0);
    auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(runtimePrecision);

    MKLDNNDims S_4D_shape {L, D, N, SC};

    // Try to create descriptor and corresponding configuration
    in_data_d[RNNInOutKind::Layer]  = {MKLDNNDims{in_data_dims},  dataType, memory::format_tag::tnc};
    out_data_d[RNNInOutKind::Layer] = {MKLDNNDims{out_data_dims}, dataType, memory::format_tag::tnc};

    in_data_d[RNNInOutKind::HiddenState]  = {MKLDNNDims{S_4D_shape}, dataType, memory::format_tag::ldnc};
    out_data_d[RNNInOutKind::HiddenState] = {MKLDNNDims{S_4D_shape}, dataType, memory::format_tag::ldnc};

    if (haveCellState(cell_type)) {
        in_data_d[RNNInOutKind::CellState] = {MKLDNNDims{S_4D_shape}, memory::data_type::f32, memory::format_tag::ldnc};
        out_data_d[RNNInOutKind::CellState] = {MKLDNNDims{S_4D_shape}, memory::data_type::f32, memory::format_tag::ldnc};
    }

    w_data_d  = {{L, D, DC, G, SC}, dataType, memory::format_tag::ldigo};
    w_state_d = {{L, D, SC, G, SC}, dataType, memory::format_tag::ldigo};

    w_bias_d = {{L, D, Gb, SC}, memory::data_type::f32, memory::format_tag::ldgo};

    copyWeightsData();

    std::vector<TensorDesc> in_candidate;

    if (nativeOrder)
        in_candidate.push_back(MKLDNNMemoryDesc{inDims[RNNInOutKind::Layer], dataType, memory::format_tag::tnc});
    else
        in_candidate.push_back(MKLDNNMemoryDesc{{N, T, DC}, dataType, memory::format_tag::ntc});

    in_candidate.push_back(MKLDNNMemoryDesc{{N, D, SC}, dataType, memory::format_tag::ntc}); // initial hidden state
    if (haveCellState(cell_type))
        in_candidate.push_back(MKLDNNMemoryDesc{{N, D, SC}, memory::data_type::f32, memory::format_tag::ntc}); // initial cell state
    in_candidate.push_back(MKLDNNMemoryDesc{{N}, memory::data_type::s32, memory::format_tag::x}); // sequence lengths
    in_candidate.push_back(MKLDNNMemoryDesc{{D, G * SC, DC}, memory::data_type::f32, memory::format_tag::ntc}); // W
    in_candidate.push_back(MKLDNNMemoryDesc{{D, G * SC, SC}, memory::data_type::f32, memory::format_tag::ntc}); // R
    in_candidate.push_back(MKLDNNMemoryDesc{{D, Gb * SC}, memory::data_type::f32, memory::format_tag::nc}); // B

    std::vector<TensorDesc> out_candidate;

    if (nativeOrder) {
        out_candidate.push_back(out_data_d[RNNInOutKind::Layer]);
    } else {
        // TODO reorder ntc -> ndtc does not work, thus use tnc(plain) + transformation reshape-transpose-reshape for now.
        out_candidate.push_back(MKLDNNMemoryDesc{{T, N, SC}, dataType, memory::format_tag::tnc});
    }

    out_candidate.push_back(MKLDNNMemoryDesc{{N, D, SC}, dataType, memory::format_tag::ntc});
    if (haveCellState(cell_type))
        out_candidate.push_back(MKLDNNMemoryDesc{{N, D, SC}, memory::data_type::f32, memory::format_tag::ntc});

    createDescriptor(in_candidate, out_candidate);
}

bool MKLDNNRNN::verifyWeightsPrecision(const Precision &layerPrec, const Precision &weightsPrec) {
    if (!weightsByLayerPrec.count(layerPrec))
        IE_THROW() << "Unsupported layer precision " << layerPrec;
    return weightsPrec == weightsByLayerPrec.at(layerPrec);
}

template <typename Prec>
void MKLDNNRNN::fillWeights(const int *gate_map, const size_t wIdx, const size_t rIdx) {
    const auto weightPrec = getOriginalInputPrecisionAtPort(wIdx);
    if (!verifyWeightsPrecision(runtimePrecision, weightPrec) && runtimePrecision != Precision::BF16 && weightPrec != Precision::FP32) {
        IE_THROW() << "Doesn't support combination of weights precision: " << weightPrec << " and runtime precision: " << runtimePrecision;
    }
    // create weight blobs (data and state part)
    auto w_data_mem = std::make_shared<MKLDNNMemory>(getEngine());
    w_data_mem->Create(w_data_d);
    internalBlobMemory.push_back(w_data_mem);
    auto w_state_mem = std::make_shared<MKLDNNMemory>(getEngine());
    w_state_mem->Create(w_state_d);
    internalBlobMemory.push_back(w_state_mem);

    const size_t ie_w_vec_size = getParentEdgesAtPort(wIdx)[0]->getDims().size();
    const size_t ie_r_vec_size = getParentEdgesAtPort(rIdx)[0]->getDims().size();

    auto *wInputNode = dynamic_cast<MKLDNNInputNode *>(getParentEdgesAtPort(wIdx)[0]->getParent().get());
    auto wConstBlob = wInputNode->getMemoryPtr();

    auto *rInputNode = dynamic_cast<MKLDNNInputNode *>(getParentEdgesAtPort(rIdx)[0]->getParent().get());
    auto rConstBlob = rInputNode->getMemoryPtr();

    std::vector<Prec> ie_w_vec(ie_w_vec_size), ie_r_vec(ie_r_vec_size);

    auto ie_w_ptr = ie_w_vec.data();
    auto ie_r_ptr = ie_r_vec.data();
    cpu_convert(wConstBlob->GetPtr(), ie_w_ptr, weightPrec, runtimePrecision, ie_w_vec_size);
    cpu_convert(rConstBlob->GetPtr(), ie_r_ptr, weightPrec, runtimePrecision, ie_r_vec_size);

    auto w_ptr = static_cast<Prec*>(w_data_mem->GetData());
    auto r_ptr = static_cast<Prec*>(w_state_mem->GetData());
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
}

template <InferenceEngine::Precision::ePrecision Prec>
void MKLDNNRNN::fillBiases(const int *gate_map) {
    using dataType = typename PrecisionTrait<Prec>::value_type;

    if (!w_bias_d)
        return;

    if (getOriginalInputPrecisionAtPort(bIdx) != Precision::FP32) {
        IE_THROW() << "Doesn't support bias precision: " << getOriginalInputPrecisionAtPort(bIdx);
    }

    auto w_bias_mem = std::make_shared<MKLDNNMemory>(getEngine());
    w_bias_mem->Create(w_bias_d);
    internalBlobMemory.push_back(w_bias_mem);

    auto *constInputNode = dynamic_cast<MKLDNNInputNode *>(getParentEdgesAtPort(bIdx)[0]->getParent().get());
    auto constBlob = constInputNode->getMemoryPtr();
    auto const elementsCount = constBlob->GetElementsCount();

    std::vector<dataType> ie_b_vec(elementsCount);
    cpu_convert(constBlob->GetPtr(),
                &ie_b_vec[0],
                MKLDNNExtensionUtils::DataTypeToIEPrecision(constBlob->GetDataType()),
                Prec,
                elementsCount);

    auto b_ptr = static_cast<dataType*>(w_bias_mem->GetData());
    for (int g = 0; g < Gb; g++) {
        dataType *l_b_ptr = b_ptr + gate_map[g] * SC;
        const dataType *l_ie_b_ptr = &ie_b_vec[g * SC];
        cpu_memcpy(l_b_ptr, l_ie_b_ptr, SC * sizeof(typename PrecisionTrait<Prec>::value_type));
    }
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
            IE_THROW() << "G isn't equal to the size of gate_map";
        }
    } else if (cell_type == mkldnn::algorithm::vanilla_gru) {
        gate_map = gate_map_gru;
        if (G > gate_map_gru_size) {
            IE_THROW() << "G isn't equal to the size of gate_map";
        }
    } else if (cell_type == mkldnn::algorithm::lbr_gru) {
        gate_map = gate_map_gru;
        if (G > gate_map_gru_size) {
            IE_THROW() << "G isn't equal to the size of gate_map";
        }
    } else if (cell_type == mkldnn::algorithm::vanilla_rnn) {
        gate_map = gate_map_rnn;
        if (G > gate_map_rnn_size) {
            IE_THROW() << "G isn't equal to the size of gate_map";
        }
    } else {
        gate_map = gate_map_gru;
        if (G > gate_map_gru_size) {
            IE_THROW() << "G isn't equal to the size of gate_map";
        }
    }

    if (runtimePrecision == Precision::BF16)
        fillWeights<bfloat16_t>(gate_map, wIdx, rIdx);
    else if (runtimePrecision == Precision::FP32)
        fillWeights<float>(gate_map, wIdx, rIdx);
    else // TODO FP16 and INT8 support
        IE_THROW() << "Unsupported data type";

    if (runtimePrecision == Precision::BF16 || runtimePrecision == Precision::FP32)
        fillBiases<Precision::FP32>(gate_map);
}

void MKLDNNRNN::createDescriptor(const std::vector<TensorDesc> &inputDesc,
                                 const std::vector<TensorDesc> &outputDesc) {
    switch (cell_type) {
        case mkldnn::algorithm::vanilla_rnn: {
            MKLDNNDescriptor desc(std::shared_ptr<vanilla_rnn_forward::desc>(
                    new vanilla_rnn_forward::desc(prop_kind::forward_scoring, cell_act, direction,
                            /* In Data       */ in_data_d[RNNInOutKind::Layer],
                            /* In State      */ in_data_d[RNNInOutKind::HiddenState],
                            /* Weights data  */ w_data_d,
                            /* Weights state */ w_state_d,
                            /* Bias          */ w_bias_d,
                            /* Out Data      */ out_data_d[RNNInOutKind::Layer],
                            /* Out State     */ out_data_d[RNNInOutKind::HiddenState])));
            descs.push_back(desc);
        } break;
        case mkldnn::algorithm::vanilla_gru: {
            MKLDNNDescriptor desc(std::shared_ptr<gru_forward::desc>(
                    new gru_forward::desc(prop_kind::forward_scoring, direction,
                            /* In Data       */ in_data_d[RNNInOutKind::Layer],
                            /* In State      */ in_data_d[RNNInOutKind::HiddenState],
                            /* Weights data  */ w_data_d,
                            /* Weights state */ w_state_d,
                            /* Bias          */ w_bias_d,
                            /* Out Data      */ out_data_d[RNNInOutKind::Layer],
                            /* Out State     */ out_data_d[RNNInOutKind::HiddenState])));
            descs.push_back(desc);
        } break;
        case mkldnn::algorithm::lbr_gru: {
            MKLDNNDescriptor desc(std::shared_ptr<lbr_gru_forward::desc>(
                    new lbr_gru_forward::desc(prop_kind::forward_scoring, direction,
                            /* In Data       */ in_data_d[RNNInOutKind::Layer],
                            /* In State      */ in_data_d[RNNInOutKind::HiddenState],
                            /* Weights data  */ w_data_d,
                            /* Weights state */ w_state_d,
                            /* Bias          */ w_bias_d,
                            /* Out Data      */ out_data_d[RNNInOutKind::Layer],
                            /* Out State     */ out_data_d[RNNInOutKind::HiddenState])));
            descs.push_back(desc);
        } break;
        case mkldnn::algorithm::vanilla_lstm: {
            MKLDNNDescriptor desc(std::shared_ptr<lstm_forward::desc>(
                    new lstm_forward::desc(prop_kind::forward_scoring, direction,
                            /* In Data       */ in_data_d[RNNInOutKind::Layer],
                            /* In State      */ in_data_d[RNNInOutKind::HiddenState],
                            /* In State C    */ in_data_d[RNNInOutKind::CellState],
                            /* Weights data  */ w_data_d,
                            /* Weights state */ w_state_d,
                            /* Bias          */ w_bias_d,
                            /* Out Data      */ out_data_d[RNNInOutKind::Layer],
                            /* Out State     */ out_data_d[RNNInOutKind::HiddenState],
                            /* Out State C   */ out_data_d[RNNInOutKind::CellState])));
            descs.push_back(desc);
        } break;
        default:
            IE_THROW() << "Unknown cell type";
    }

    // Fill supported config
    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = false;
    for (size_t i = 0; i < inputDesc.size(); i++) {
        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;
        dataConfig.desc = inputDesc[i];
        config.inConfs.push_back(dataConfig);
    }

    for (size_t i = 0; i < outputDesc.size(); i++) {
        InferenceEngine::DataConfig dataConfig;
        dataConfig.inPlace = -1;
        dataConfig.constant = false;
        dataConfig.desc = outputDesc[i];
        config.outConfs.push_back(dataConfig);
    }

    supportedPrimitiveDescriptors.emplace_back(config, ref_any);
}

void MKLDNNRNN::createPrimitive() {
    auto pd = descs[0].createPrimitiveDescriptorIterator(getEngine());
    prim.reset(new mkldnn::primitive(pd));
}

void MKLDNNRNN::execute(mkldnn::stream strm) {
    if (!prim)
        IE_THROW() << "No initialized primitive to execute";

    const auto src_data_mem = getParentEdgeAt(0)->getMemoryPtr();
    const auto dst_data_mem = getChildEdgeAt(0)->getMemoryPtr();

    const auto &wgh_data_mem = internalBlobMemory[0];
    const auto &wgh_stat_mem = internalBlobMemory[1];
    const auto &wgh_bias_mem = internalBlobMemory[2];

    std::unordered_map<int, memory> args {
        {DNNL_ARG_SRC_LAYER,     src_data_mem->GetPrimitive()},
        {DNNL_ARG_WEIGHTS_LAYER, wgh_data_mem->GetPrimitive()},
        {DNNL_ARG_WEIGHTS_ITER,  wgh_stat_mem->GetPrimitive()},
        {DNNL_ARG_BIAS,          wgh_bias_mem->GetPrimitive()},
        {DNNL_ARG_DST_LAYER,     dst_data_mem->GetPrimitive()},
    };

    int state_i_tags[] {DNNL_ARG_SRC_ITER, DNNL_ARG_SRC_ITER_C};
    int state_o_tags[] {DNNL_ARG_DST_ITER, DNNL_ARG_DST_ITER_C};
    for (size_t s = 0; s < S; s++) {
        args[state_i_tags[s]] = getParentEdgeAt(s+1)->getMemoryPtr()->GetPrimitive();
    }

    if (is_cell) {
        for (size_t s = 0; s < S; s++) {
            args[state_o_tags[s]] = getChildEdgesAtPort(s)[0]->getMemoryPtr()->GetPrimitive();
        }
    } else {
        ptrdiff_t n_ports_with_init_states = outDims.size() - 1; // first is a sequence data
        for (size_t s = 0; s < std::min(S, n_ports_with_init_states); s++) {
            if (s < inDims.size()) {
                args[state_o_tags[s]] = getChildEdgesAtPort(s+1)[0]->getMemoryPtr()->GetPrimitive();
            }
        }
    }

    (*prim).execute(strm, args);
}

REG_MKLDNN_PRIM_FOR(MKLDNNRNN, RNNCell);
REG_MKLDNN_PRIM_FOR(MKLDNNRNN, RNNSeq);
}  // namespace MKLDNNPlugin
