// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_rnn.h"
#include "mkldnn_extension_utils.h"

#include "utils/general_utils.h"
#include "nodes/common/cpu_memcpy.h"

#include <string>
#include <utility>

using namespace mkldnn;
using namespace InferenceEngine;

namespace MKLDNNPlugin {

using _RNN = RNNSequenceLayer;  // alias

static rnn_direction ie2mkl(_RNN::Direction &direction) {
    return direction == _RNN::FWD ? rnn_direction::unidirectional_left2right
         : direction == _RNN::BWD ? rnn_direction::unidirectional_right2left
         : direction == _RNN::BDR ? rnn_direction::bidirectional_concat
         : rnn_direction::unidirectional;
}

static algorithm ie2mkl(std::string act_type) {
    return act_type == "sigmoid" ? algorithm::eltwise_logistic
         : act_type == "tanh"    ? algorithm::eltwise_tanh
         : act_type == "relu"    ? algorithm::eltwise_relu
         : algorithm::undef;
}

static algorithm ie2mkl(RNNCellBase::CellType cell_type) {
    switch (cell_type) {
        case RNNCellBase::RNN:     return algorithm::vanilla_rnn;
        case RNNCellBase::LSTM:    return algorithm::vanilla_lstm;
        case RNNCellBase::GRU:     return algorithm::vanilla_gru;
        case RNNCellBase::GRU_LBR: return algorithm::lbr_gru;
        default:
            THROW_IE_EXCEPTION << "Unsupported cell type";
            return algorithm::undef;
    }
}

size_t gatesCount(algorithm alg) {
    switch (alg) {
        case algorithm::vanilla_rnn:     return 1;
        case algorithm::vanilla_gru:
        case algorithm::lbr_gru:         return 3;
        case algorithm::vanilla_lstm:    return 4;
        default:
            THROW_IE_EXCEPTION << "Unsupported cell type";
            return 0;
    }
}

size_t statesCount(algorithm alg) {
    switch (alg) {
        case algorithm::vanilla_rnn:
        case algorithm::vanilla_gru:
        case algorithm::lbr_gru:         return 1;
        case algorithm::vanilla_lstm:    return 2;
        default:
            THROW_IE_EXCEPTION << "Unsupported cell type";
            return 0;
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

    cell_type = ie2mkl(cellLayer->cellType);
    cell_act = ie2mkl(cellLayer->activations[0]);  // Works only for RNN with one gate

    if (cellLayer->clip != 0.0f) {
        // TODO [oneDNN]: No more supported
        THROW_IE_EXCEPTION << "Clipping is not supported for RNN primitive";
//        cell_desc.set_clipping(cellLayer->clip);
    }

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

    G = gatesCount(cell_type);
    S = statesCount(cell_type);
    T = 1;
    N  = in_data_dims[0];
    DC = in_data_dims[1];
    SC = in_h_state_dims[1];

    Gb = (cell_type != mkldnn::algorithm::lbr_gru) ? G : G + 1;

    // Expected shapes
    MKLDNNDims D_shape {N, DC}, S_shape {N, SC}, S_4D_shape {L, D, N, SC};

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
    for (size_t i = 0; i < S; i++) {
        in_states_d.emplace_back(S_4D_shape, memory::data_type::f32, memory::format_tag::ldnc);
        out_states_d.emplace_back(S_4D_shape, memory::data_type::f32, memory::format_tag::ldnc);
    }

    in_data_d  = {{T, N, DC}, memory::data_type::f32, memory::format_tag::tnc};;
    out_data_d = {{T, N, SC}, memory::data_type::f32, memory::format_tag::tnc};;

    w_data_d   = {{L, D, DC, G, SC}, memory::data_type::f32, memory::format_tag::ldigo};
    w_state_d  = {{L, D, SC, G, SC}, memory::data_type::f32, memory::format_tag::ldigo};

    if (bias)
        w_bias_d = {{L, D, Gb, SC}, memory::data_type::f32, memory::format_tag::ldgo};

    std::vector<TensorDesc> in_candidate, out_candidate;
    std::vector<memory::format_tag> outputFormats;
    in_candidate.emplace_back(MKLDNNMemoryDesc {D_shape, memory::data_type::f32, memory::format_tag::nc});
    in_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::data_type::f32, memory::format_tag::nc});
    out_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::data_type::f32, memory::format_tag::nc});
    outputFormats.emplace_back(memory::format_tag::nc);

    if (S == 2) {
        in_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::data_type::f32, memory::format_tag::nc});
        out_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::data_type::f32, memory::format_tag::nc});
        outputFormats.emplace_back(memory::format_tag::nc);
    }

    createDescriptor(in_candidate, out_candidate);
}

void MKLDNNRNN::fillSeqDesc() {
    if (!descs.empty()) return;
    auto rnnLayer = std::dynamic_pointer_cast<RNNSequenceLayer>(getCnnLayer());

    if (!rnnLayer)
        THROW_IE_EXCEPTION << "Wrong RNN layer representation. Cannot cast to RNNSequenceLayer.";

    if (!one_of(rnnLayer->cellType, _RNN::LSTM, _RNN::GRU, _RNN::GRU_LBR, _RNN::RNN))
        THROW_IE_EXCEPTION << "RNN layer supports only LSTM/GRU/RNN cell";

    cell_type = ie2mkl(rnnLayer->cellType);
    cell_act = algorithm::undef;
    if (!rnnLayer->activations.empty())
        cell_act = ie2mkl(rnnLayer->activations[0]);  // Works only for RNN with one gate

    // TODO [oneDNN]: No more supported
    if (rnnLayer->clip != 0.0f) {
        THROW_IE_EXCEPTION << "Clipping is not supported for RNN primitive";
//        cell_desc.set_clipping(rnnLayer->clip);
    }

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

    G = gatesCount(cell_type);
    S = statesCount(cell_type);
    T = in_data_dims[0];
    N = in_data_dims[1];
    DC = in_data_dims[2];
    SC = out_data_dims[2];

    Gb = (cell_type != mkldnn::algorithm::lbr_gru) ? G : G + 1;

    MKLDNNDims ID_shape {T, N, DC}, OD_shape {T, N, SC}, S_shape {N, SC}, S_4D_shape {L, D, N, SC};

    if (out_data_dims != OD_shape)
        THROW_IE_EXCEPTION << "Incorrect shape of input/output ports for layer " << getName();

    in_states_d.resize(S);
    out_states_d.resize(S);

    for (int i = 1; i < ins.size(); i++) {
        if (getParentEdgeAt(i)->getDims() != S_shape)
            THROW_IE_EXCEPTION << "Incorrect shape of state ports for layer " << getName();
        in_states_d[i - 1] = {S_4D_shape, memory::data_type::f32, memory::format_tag::ldnc};
    }

    for (int i = 1; i < outs.size(); i++) {
        if (getChildEdgeAt(i)->getDims() != S_shape)
            THROW_IE_EXCEPTION << "Incorrect shape of state ports for layer " << getName();
        out_states_d[i - 1] = {S_4D_shape, memory::data_type::f32, memory::format_tag::ldnc};
    }

    auto blobs = rnnLayer->blobs;
    Blob::Ptr weights, bias;
    if (blobs.find("weights") != blobs.end()) weights = blobs["weights"];
    if (blobs.find("biases") != blobs.end()) bias = blobs["biases"];

    if (!weights)
        THROW_IE_EXCEPTION << "RNN Layer. Weights do not present.";

    if (weights->size() != G*SC*(SC+DC))
        THROW_IE_EXCEPTION << "RNN Layer. Weights size is not correct. Expected size:" << G*SC*(SC+DC);

    w_data_d  = {{L, D, DC, G, SC}, memory::data_type::f32, memory::format_tag::ldigo};
    w_state_d = {{L, D, SC, G, SC}, memory::data_type::f32, memory::format_tag::ldigo};

    if (bias && bias->size() != Gb*SC)
        THROW_IE_EXCEPTION << "RNN Layer. Biases size is not correct. Expected size:" << G*SC;

    if (bias)
        w_bias_d = {{L, D, Gb, SC}, memory::data_type::f32, memory::format_tag::ldgo};

    // Try to create descriptor and corresponding configuration
    in_data_d = {in_data_dims, memory::data_type::f32, memory::format_tag::tnc};
    out_data_d = {out_data_dims, memory::data_type::f32, memory::format_tag::tnc};

    std::vector<TensorDesc> in_candidate;
    if (nativeOrder)
        in_candidate.push_back(in_data_d);
    else
        in_candidate.push_back(MKLDNNMemoryDesc{{N, T, DC}, memory::data_type::f32, memory::format_tag::ntc});

    for (int i = 1; i < ins.size(); i++)
        in_candidate.emplace_back(MKLDNNMemoryDesc {S_shape, memory::data_type::f32, memory::format_tag::nc});

    std::vector<TensorDesc> out_candidate;
    if (nativeOrder) {
        out_candidate.push_back(out_data_d);
    } else {
        out_candidate.push_back(MKLDNNMemoryDesc{{N, T, SC}, memory::data_type::f32, memory::format_tag::ntc});
    }

    for (int i = 1; i < outs.size(); i++) {
        out_candidate.emplace_back(MKLDNNMemoryDesc{S_shape, memory::data_type::f32, memory::format_tag::nc});
    }

    createDescriptor(in_candidate, out_candidate);
}

void MKLDNNRNN::createDescriptor(const std::vector<TensorDesc> &inputDesc,
                                 const std::vector<TensorDesc> &outputDesc) {
    switch (cell_type) {
        case mkldnn::algorithm::vanilla_rnn: {
            MKLDNNDescriptor desc(std::shared_ptr<vanilla_rnn_forward::desc>(
                    new vanilla_rnn_forward::desc(prop_kind::forward_scoring, cell_act, direction,
                            /* In Data       */ in_data_d,
                            /* In State      */ in_states_d[0],
                            /* Weights data  */ w_data_d,
                            /* Weights state */ w_state_d,
                            /* Bias          */ w_bias_d,
                            /* Out Data      */ out_data_d,
                            /* Out State     */ out_states_d[0])));
            descs.push_back(desc);
        } break;
        case mkldnn::algorithm::vanilla_gru: {
            MKLDNNDescriptor desc(std::shared_ptr<gru_forward::desc>(
                    new gru_forward::desc(prop_kind::forward_scoring, direction,
                            /* In Data       */ in_data_d,
                            /* In State      */ in_states_d[0],
                            /* Weights data  */ w_data_d,
                            /* Weights state */ w_state_d,
                            /* Bias          */ w_bias_d,
                            /* Out Data      */ out_data_d,
                            /* Out State     */ out_states_d[0])));
            descs.push_back(desc);
        } break;
        case mkldnn::algorithm::lbr_gru: {
            MKLDNNDescriptor desc(std::shared_ptr<lbr_gru_forward::desc>(
                    new lbr_gru_forward::desc(prop_kind::forward_scoring, direction,
                            /* In Data       */ in_data_d,
                            /* In State      */ in_states_d[0],
                            /* Weights data  */ w_data_d,
                            /* Weights state */ w_state_d,
                            /* Bias          */ w_bias_d,
                            /* Out Data      */ out_data_d,
                            /* Out State     */ out_states_d[0])));
            descs.push_back(desc);
        } break;
        case mkldnn::algorithm::vanilla_lstm: {
            MKLDNNDescriptor desc(std::shared_ptr<lstm_forward::desc>(
                    new lstm_forward::desc(prop_kind::forward_scoring, direction,
                            /* In Data       */ in_data_d,
                            /* In State H    */ in_states_d[0],
                            /* In State C    */ in_states_d[1],
                            /* Weights data  */ w_data_d,
                            /* Weights state */ w_state_d,
                            /* Bias          */ w_bias_d,
                            /* Out Data      */ out_data_d,
                            /* Out State H   */ out_states_d[0],
                            /* Out State C   */ out_states_d[1])));
            descs.push_back(desc);
        } break;
        default:
            THROW_IE_EXCEPTION << "Unknown cell type";
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

    auto pd = descs[0].createPrimitiveDescriptorIterator(getEngine());

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
        if (cell_type == algorithm::vanilla_lstm) {
            gate_map = gate_map_lstm;
            if (G > gate_map_lstm_size) {
                THROW_IE_EXCEPTION << "G isn't equal to the size of gate_map";
            }
        } else if (cell_type == algorithm::vanilla_gru) {
            gate_map = gate_map_gru;
            if (G > gate_map_gru_size) {
                THROW_IE_EXCEPTION << "G isn't equal to the size of gate_map";
            }
        } else if (cell_type == algorithm::lbr_gru) {
            gate_map = gate_map_gru;
            if (G > gate_map_gru_size) {
                THROW_IE_EXCEPTION << "G isn't equal to the size of gate_map";
            }
        } else if (cell_type == algorithm::vanilla_rnn) {
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
                const float *l_ie_b_ptr = ie_b_ptr + g * SC;
                cpu_memcpy(l_b_ptr, l_ie_b_ptr, SC * sizeof(float));
            }
        }
    }

    prim.reset(new mkldnn::primitive(pd));
}

void MKLDNNRNN::execute(mkldnn::stream strm) {
    if (!prim)
        THROW_IE_EXCEPTION << "No initialized primitive to execute";

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
