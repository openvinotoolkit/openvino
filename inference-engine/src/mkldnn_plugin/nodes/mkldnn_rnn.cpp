// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_rnn.h"
#include "mkldnn_extension_utils.h"
#include "desc_iterator.hpp"

#include <string>
#include <utility>

using namespace mkldnn;
using namespace InferenceEngine;

namespace MKLDNNPlugin {

template <typename T, typename P>
inline bool one_of(T val, P item) { return val == item; }
template <typename T, typename P, typename... Args>
inline bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

using _RNN = RNNSequenceLayer;  // alias

static rnn_direction ie2mkl(_RNN::Direction &direction) {
    return direction == _RNN::FWD ? unidirectional_left2right
         : direction == _RNN::BWD ? unidirectional_right2left
         : direction == _RNN::BDR ? bidirectional_concat
         : unidirectional;
}

static algorithm ie2mkl(std::string act_type) {
    return act_type == "sigmoid" ? eltwise_logistic
         : act_type == "tanh"    ? eltwise_tanh
         : act_type == "relu"    ? eltwise_relu
         : algorithm_undef;
}

static algorithm ie2mkl(RNNCellBase::CellType cell_type) {
    switch (cell_type) {
        case RNNCellBase::LSTM: return vanilla_lstm;
        case RNNCellBase::GRU:  return vanilla_gru;
        case RNNCellBase::GRU_LBR:  return gru_linear_before_reset;
        case RNNCellBase::RNN:  return vanilla_rnn;
        default:
            THROW_IE_EXCEPTION << "Unsoupported cell type";
            return algorithm_undef;
    }
}

MKLDNNRNN::MKLDNNRNN(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(layer, eng, cache) {
    is_cell = one_of(layer->type, "LSTMCell", "GRUCell", "RNNCell");
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

void MKLDNNRNN::fillCellDesc() {
    if (!descs.empty()) return;
    auto cellLayer = std::dynamic_pointer_cast<RNNCellBase>(getCnnLayer());

    if (!cellLayer)
        THROW_IE_EXCEPTION << "No original layer for RNNCell.";

    algorithm cell_type = ie2mkl(cellLayer->cellType);
    algorithm cell_act = ie2mkl(cellLayer->activations[0]);  // Works only for RNN with one gate

    cell_desc = {cell_type, cell_act};
    if (cellLayer->clip != 0.0f)
        cell_desc.set_clipping(cellLayer->clip);

    auto &ins = cellLayer->insData;
    auto &outs = cellLayer->outData;

    if (!one_of(ins.size(), 3, 2))
        THROW_IE_EXCEPTION << "Incorrect number of input ports for layer " << getName();
    if (!one_of(outs.size(), 2, 1))
        THROW_IE_EXCEPTION << "Incorrect number of output ports for layer " << getName();

    auto in_data_dims = getParentEdgeAt(0)->getDims();
    auto in_h_state_dims = getParentEdgeAt(1)->getDims();
    auto out_h_state_dims = getChildEdgeAt(0)->getDims();

    if (in_data_dims.ndims() != 2 || in_h_state_dims.ndims() != 2)
        THROW_IE_EXCEPTION << "Incorrect shape of input/output ports for layer " << getName();

    G = cell_desc.get_gates_count();
    S = cell_desc.get_state_count();
    T = 1;
    N  = in_data_dims[0];
    DC = in_data_dims[1];
    SC = in_h_state_dims[1];

    Gb = (cell_type != gru_linear_before_reset) ? G : G + 1;

    // Expected shapes
    MKLDNNDims D_shape {N, DC}, S_shape {N, SC};

    if (in_data_dims != D_shape
        || in_h_state_dims != S_shape
        || out_h_state_dims != S_shape)
        THROW_IE_EXCEPTION << "Incorrect shape of input/output ports for layer " << getName();

    if (S == 2) {
        auto in_c_state_dims = getParentEdgeAt(2)->getDims();
        auto out_c_state_dims = getChildEdgeAt(1)->getDims();

        if (in_c_state_dims != S_shape
            || out_c_state_dims != S_shape)
            THROW_IE_EXCEPTION << "Incorrect shape of input/output ports for layer " << getName();
    }

    auto blobs = cellLayer->blobs;
    Blob::Ptr weights, bias;
    if (blobs.find("weights") != blobs.end()) weights = blobs["weights"];
    if (blobs.find("biases") != blobs.end()) bias = blobs["biases"];

    if (!weights)
        THROW_IE_EXCEPTION << "RNN Layer. Weights do not present.";

    if (weights->size() != G*SC*(SC+DC))
        THROW_IE_EXCEPTION << "RNN Layer. Weights size is not correct. Expected size:" << G*SC*(SC+DC);

    if (bias && bias->size() != Gb*SC)
        THROW_IE_EXCEPTION << "RNN Layer. Biases size is not correct. Expected size:" << G*SC;

    // Shapes and Attributes are correct. Can start internal stuff initialization.

    in_state_d  = {{L, D, S, N, SC}, memory::f32, memory::ldsnc};
    out_state_d = {{L, D, S, N, SC}, memory::f32, memory::ldsnc};

    in_data_d  = {{T, N, DC}, memory::f32, memory::tnc};;
    out_data_d = {{T, N, SC}, memory::f32, memory::tnc};;

    w_data_d   = {{L, D, DC, G, SC}, memory::f32, memory::ldigo};
    w_state_d  = {{L, D, SC, G, SC}, memory::f32, memory::ldigo};

    if (bias)
        w_bias_d = {{L, D, Gb, SC}, memory::f32, memory::ldgo};

    std::vector<TensorDesc> in_candidate, out_candidate;
    std::vector<memory::format> outputFormats;
    in_candidate.emplace_back(MKLDNNMemoryDesc {D_shape, memory::f32, memory::nc});
    in_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::f32, memory::nc});
    out_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::f32, memory::nc});
    outputFormats.emplace_back(memory::nc);

    if (S == 2) {
        in_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::f32, memory::nc});
        out_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::f32, memory::nc});
        outputFormats.emplace_back(memory::nc);
    }

    createDescriptor(in_candidate, out_candidate, outputFormats);
}

void MKLDNNRNN::fillSeqDesc() {
    if (!descs.empty()) return;
    auto rnnLayer = std::dynamic_pointer_cast<RNNSequenceLayer>(getCnnLayer());

    if (!rnnLayer)
        THROW_IE_EXCEPTION << "Wrong RNN layer representation. Cannot cast to RNNSequenceLayer.";

    if (!one_of(rnnLayer->cellType, _RNN::LSTM, _RNN::GRU, _RNN::GRU_LBR, _RNN::RNN))
        THROW_IE_EXCEPTION << "RNN layer supports only LSTM/GRU/RNN cell";

    algorithm cell_type = ie2mkl(rnnLayer->cellType);
    algorithm cell_act = algorithm_undef;
    if (!rnnLayer->activations.empty())
        cell_act = ie2mkl(rnnLayer->activations[0]);  // Works only for RNN with one gate

    cell_desc = {cell_type, cell_act};

    if (rnnLayer->clip != 0.0f)
        cell_desc.set_clipping(rnnLayer->clip);

    if (!one_of(rnnLayer->axis, 0, 1))
        THROW_IE_EXCEPTION << "RNN layer supports only sequence axis 0 or 1";
    nativeOrder = rnnLayer->axis == 0;

    if (!one_of(rnnLayer->direction, _RNN::FWD, _RNN::BWD))
        THROW_IE_EXCEPTION << "RNN layer supports only unidirectional RNN layer";
    direction = ie2mkl(rnnLayer->direction);

    auto &ins = rnnLayer->insData;
    auto &outs = rnnLayer->outData;

    if (!one_of(ins.size(), 3, 2, 1))
        THROW_IE_EXCEPTION << "Incorrect number of input ports for layer " << getName();
    if (!one_of(outs.size(), 3, 2, 1))
        THROW_IE_EXCEPTION << "Incorrect number of output ports for layer " << getName();

    auto in_data_dims = getParentEdgeAt(0)->getDims();
    auto out_data_dims = getChildEdgeAt(0)->getDims();

    if (in_data_dims.ndims() != 3 || out_data_dims.ndims() != 3)
        THROW_IE_EXCEPTION << "Incorrect shape of input/output ports for layer " << getName();

    if (!nativeOrder) {
        std::swap(in_data_dims[0], in_data_dims[1]);
        std::swap(out_data_dims[0], out_data_dims[1]);
    }

    G = cell_desc.get_gates_count();
    S = cell_desc.get_state_count();
    T = in_data_dims[0];
    N = in_data_dims[1];
    DC = in_data_dims[2];
    SC = out_data_dims[2];

    Gb = (cell_type != gru_linear_before_reset) ? G : G + 1;

    MKLDNNDims ID_shape {T, N, DC}, OD_shape {T, N, SC}, S_shape {N, SC};

    if (out_data_dims != OD_shape)
        THROW_IE_EXCEPTION << "Incorrect shape of input/output ports for layer " << getName();

    if (ins.size() > 1) {
        for (int i = 1; i < ins.size(); i++)
            if (getParentEdgeAt(i)->getDims() != S_shape)
                THROW_IE_EXCEPTION << "Incorrect shape of state ports for layer " << getName();

        in_state_d = {{L, D, S, N, SC}, memory::f32, memory::ldsnc};
    }

    if (outs.size() > 1) {
        for (int i = 1; i < outs.size(); i++)
            if (getChildEdgeAt(i)->getDims() != S_shape)
                THROW_IE_EXCEPTION << "Incorrect shape of state ports for layer " << getName();

        out_state_d = {{L, D, S, N, SC}, memory::f32, memory::ldsnc};
    }

    auto blobs = rnnLayer->blobs;
    Blob::Ptr weights, bias;
    if (blobs.find("weights") != blobs.end()) weights = blobs["weights"];
    if (blobs.find("biases") != blobs.end()) bias = blobs["biases"];

    if (!weights)
        THROW_IE_EXCEPTION << "RNN Layer. Weights do not present.";

    if (weights->size() != G*SC*(SC+DC))
        THROW_IE_EXCEPTION << "RNN Layer. Weights size is not correct. Expected size:" << G*SC*(SC+DC);

    w_data_d  = {{L, D, DC, G, SC}, memory::f32, memory::ldigo};
    w_state_d = {{L, D, SC, G, SC}, memory::f32, memory::ldigo};

    if (bias && bias->size() != Gb*SC)
        THROW_IE_EXCEPTION << "RNN Layer. Biases size is not correct. Expected size:" << G*SC;

    if (bias)
        w_bias_d = {{L, D, Gb, SC}, memory::f32, memory::ldgo};

    // Try to create descriptor and corresponding configuration
    in_data_d = {in_data_dims, memory::f32, memory::tnc};
    out_data_d = {out_data_dims, memory::f32, memory::tnc};

    std::vector<TensorDesc> in_candidate;
    if (nativeOrder)
        in_candidate.push_back(in_data_d);
    else
        in_candidate.push_back(MKLDNNMemoryDesc{{N, T, DC}, memory::f32, memory::ntc});

    for (int i = 1; i < ins.size(); i++)
        in_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::f32, memory::nc});

    std::vector<TensorDesc> out_candidate;
    std::vector<memory::format> outputFormats;
    if (nativeOrder) {
        out_candidate.push_back(out_data_d);
        outputFormats.push_back(out_data_d.getFormat());
    } else {
        out_candidate.push_back(MKLDNNMemoryDesc{{N, T, SC}, memory::f32, memory::ntc});
        outputFormats.push_back(memory::ntc);
    }

    for (int i = 1; i < outs.size(); i++) {
        out_candidate.emplace_back(MKLDNNMemoryDesc{S_shape, memory::f32, memory::nc});
        outputFormats.push_back(memory::nc);
    }

    createDescriptor(in_candidate, out_candidate, outputFormats);
}

void MKLDNNRNN::createDescriptor(const std::vector<TensorDesc> &inputDesc,
                                 const std::vector<TensorDesc> &outputDesc,
                                 const std::vector<memory::format> &outputFormats) {
    MKLDNNDescriptor desc(std::shared_ptr<rnn_forward::desc>(
            new rnn_forward::desc(forward_scoring, cell_desc,
                    direction,
                    /* In Data       */ in_data_d,
                    /* In State      */ in_state_d,
                    /* Weights data  */ w_data_d,
                    /* Weights state */ w_state_d,
                    /* Bias          */ w_bias_d,
                    /* Out Data      */ out_data_d,
                    /* Out State     */ out_state_d)));
    descs.push_back(desc);

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

    supportedPrimitiveDescriptors.emplace_back(config, ref_any, outputFormats);
}

void MKLDNNRNN::createPrimitive() {
    if (prim) return;

    std::string errorPrefix =  "RNN layer '" + getCnnLayer()->name + "'";
    auto weightsIt = getCnnLayer()->blobs.find("weights");
    if (weightsIt == getCnnLayer()->blobs.end())
        THROW_IE_EXCEPTION << errorPrefix << " does not have weights blob.";
    if (weightsIt->second->getTensorDesc().getPrecision() != Precision::FP32)
        THROW_IE_EXCEPTION << errorPrefix << " has invalid weights precision: " << weightsIt->second->getTensorDesc().getPrecision();
    if (getCnnLayer()->blobs.find("biases") != getCnnLayer()->blobs.end()
            && getCnnLayer()->blobs["biases"]->getTensorDesc().getPrecision() != Precision::FP32)
        THROW_IE_EXCEPTION << errorPrefix << " has invalid biases precision: " << getCnnLayer()->blobs["biases"]->getTensorDesc().getPrecision();

    std::shared_ptr<rnn_forward::desc> d = descs[0];
    rnn_forward::primitive_desc pd(*d, getEngine());

    auto src_data_mem = getParentEdgeAt(0)->getMemoryPtr();
    auto dst_data_mem = getChildEdgeAt(0)->getMemoryPtr();

    // create weight blobs (data and state part)
    auto w_data_mem = std::make_shared<MKLDNNMemory>(getEngine());
    w_data_mem->Create(w_data_d);
    internalBlobMemory.push_back(w_data_mem);

    auto w_state_mem = std::make_shared<MKLDNNMemory>(getEngine());
    w_state_mem->Create(w_state_d);
    internalBlobMemory.push_back(w_state_mem);

    auto w_bias_mem = std::make_shared<MKLDNNMemory>(getEngine());
    w_bias_mem->Create(w_bias_d);
    internalBlobMemory.push_back(w_bias_mem);

    {
        /* Copy Weight data
         * IE format:
         *   W - [gates, out_state_size, in_data_size + in_state_size]
         *   B - [gates, out_state_size]
         *
         * MKLDNN format:
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
        if (cell_desc.get_cell_kind() == vanilla_lstm) {
            gate_map = gate_map_lstm;
            if (G > gate_map_lstm_size) {
                THROW_IE_EXCEPTION << "G isn't equal to the size of gate_map";
            }
        } else if (cell_desc.get_cell_kind() == vanilla_gru) {
            gate_map = gate_map_gru;
            if (G > gate_map_gru_size) {
                THROW_IE_EXCEPTION << "G isn't equal to the size of gate_map";
            }
        } else if (cell_desc.get_cell_kind() == gru_linear_before_reset) {
            gate_map = gate_map_gru;
            if (G > gate_map_gru_size) {
                THROW_IE_EXCEPTION << "G isn't equal to the size of gate_map";
            }
        } else if (cell_desc.get_cell_kind() == vanilla_rnn) {
            gate_map = gate_map_rnn;
            if (G > gate_map_rnn_size) {
                THROW_IE_EXCEPTION << "G isn't equal to the size of gate_map";
            }
        } else {
            gate_map = gate_map_gru;
            if (G > gate_map_gru_size) {
                THROW_IE_EXCEPTION << "G isn't equal to the size of gate_map";
            }
        }

        auto ie_w_ptr = getCnnLayer()->blobs["weights"]->buffer().as<const float*>();
        auto w_ptr = static_cast<float*>(w_data_mem->GetData());
        auto r_ptr = static_cast<float*>(w_state_mem->GetData());
        const int step = SC * G;

        for (int g = 0; g < G; g++) {
            for (int out_i = 0; out_i < SC; out_i++) {
                float *l_w_ptr = w_ptr + gate_map[g]*SC + out_i;
                float *l_r_ptr = r_ptr + gate_map[g]*SC+ out_i;
                for (int in_i = 0; in_i < DC; in_i++) {
                    *l_w_ptr = *ie_w_ptr;
                    ie_w_ptr++;
                    l_w_ptr += step;
                }

                for (int in_i = 0; in_i < SC; in_i++) {
                    *l_r_ptr = *ie_w_ptr;
                    ie_w_ptr++;
                    l_r_ptr += step;
                }
            }
        }

        if (w_bias_d) {
            auto ie_b_ptr = getCnnLayer()->blobs["biases"]->buffer().as<const float*>();
            auto b_ptr = static_cast<float*>(w_bias_mem->GetData());
            for (int g = 0; g < Gb; g++) {
                float *l_b_ptr = b_ptr + gate_map[g]*SC;
                for (int out_i = 0; out_i < SC; out_i++) {
                    *l_b_ptr = *ie_b_ptr;
                    ie_b_ptr++;
                    l_b_ptr++;
                }
            }
        }
    }

    auto src_state_mem = std::make_shared<MKLDNNMemory>(getEngine());
    src_state_mem->Create(in_state_d);
    internalBlobMemory.push_back(src_state_mem);
    if (in_state_d) {
        int offset = 0;
        for (int i = 0; i < S; i++) {
            /* create copy/concat primitive */
            auto src_stat = getParentEdgeAt(i+1)->getMemory().GetPrimitive();

            auto state_mem = std::make_shared<MKLDNNMemory>(getEngine());
            state_mem->Create(
                    src_stat.get_primitive_desc().desc(),
                    static_cast<uint8_t *>(src_state_mem->GetPrimitive().get_data_handle()) + offset);
            offset += src_stat.get_primitive_desc().get_size();

            internalBlobMemory.push_back(state_mem);

            exec_before.emplace_back(src_stat, state_mem->GetPrimitive());
        }
    }

    auto dst_state_mem = std::make_shared<MKLDNNMemory>(getEngine());
    dst_state_mem->Create(out_state_d);
    internalBlobMemory.push_back(dst_state_mem);
    if (out_state_d) {
        int offset = 0;
        int idx_start = is_cell ? 0 : 1;
        for (int i = 0; i < S; i++) {
            /* create copy/split primitive */
            auto dst_stat = getChildEdgeAt(idx_start + i)->getMemory().GetPrimitive();

            auto state_mem = std::make_shared<MKLDNNMemory>(getEngine());
            state_mem->Create(
                    dst_stat.get_primitive_desc().desc(),
                    static_cast<uint8_t *>(dst_state_mem->GetPrimitive().get_data_handle()) + offset);
            offset += dst_stat.get_primitive_desc().get_size();

            internalBlobMemory.push_back(state_mem);

            if (is_cell && i == 0) continue;
            exec_after.emplace_back(state_mem->GetPrimitive(), dst_stat);
        }
    }

    auto workspace_mem = std::make_shared<MKLDNNMemory>(getEngine());
    workspace_mem->Create({}, memory::f32, memory::format_undef, nullptr);  // stub, not in use
    internalBlobMemory.push_back(workspace_mem);

    auto p = new rnn_forward(pd,
            /* In Data       */ src_data_mem ->GetPrimitive(),
            /* In State      */ src_state_mem->GetPrimitive(),
            /* Weights data  */ w_data_mem   ->GetPrimitive(),
            /* Weights state */ w_state_mem  ->GetPrimitive(),
            /* Bias          */ w_bias_mem   ->GetPrimitive(),
            /* Out Data      */ dst_data_mem ->GetPrimitive(),
            /* Out State     */ dst_state_mem->GetPrimitive(),
            /* Workspace     */ workspace_mem->GetPrimitive());

    prim.reset(p);
}

void MKLDNNRNN::execute(mkldnn::stream strm) {
    if (!exec_before.empty())
        strm.submit({exec_before.begin(), exec_before.end()});

    if (prim)
        strm.submit({*prim});

    if (!exec_after.empty())
        strm.submit({exec_after.begin(), exec_after.end()});
}

REG_MKLDNN_PRIM_FOR(MKLDNNRNN, RNN);
}  // namespace MKLDNNPlugin
