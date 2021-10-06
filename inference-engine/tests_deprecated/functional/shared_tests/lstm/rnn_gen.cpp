// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn_gen.hpp"
#include "rnn_referee.hpp"
#include "rnn_util.hpp"
#include "xml_net_builder.hpp"
#include <ie_core.hpp>

#include <vector>
#include <string>

using namespace InferenceEngine;
using std::map;
using std::pair;
using std::vector;
using std::string;

using Shape = InferenceEngine::SizeVector;

RNNGen::RNNGen(size_t batch, size_t seq, CellDesc cell, Mode mode, Direction dir, int axis) :
        N(batch), T(seq), cell(cell), mode(mode), dir(dir),
        axis(axis), neg(mode == TI_CSTM) {
    size_t effective_T = (mode == DYN_SEQ) ? T - 1 : T;

    referee = RNN_Referee::create_referee(cell, N, effective_T, D, S);

    st_dim = {N, S};
    id_dim = (axis == 1) ? Shape{N, T, D} : Shape{T, N, D};
    od_dim = (axis == 1) ? Shape{N, T, S} : Shape{T, N, S};
    seq_l_dim = {N};

    state_num = referee->stateNum();
    wSzB = referee->wSize() * sizeof(float);
    bSzB = referee->bSize() * sizeof(float);

    weights = std::make_shared<TBlob<uint8_t>>(TensorDesc(Precision::U8, SizeVector{(wSzB + bSzB)}, Layout::C));
    weights->allocate();

    auto ptr = weights->buffer().as<float *>();
    SizeVector w_dims{referee->wSize()};
    SizeVector b_dims{referee->bSize()};
    w_blob = make_shared_blob<float>({Precision::FP32, w_dims, TensorDesc::getLayoutByDims(w_dims)}, ptr);
    b_blob = make_shared_blob<float>({Precision::FP32, b_dims, TensorDesc::getLayoutByDims(b_dims)},
                                     ptr + referee->wSize());
}

string RNNGen::model() {
    auto net_b = CommonTestUtils::V2NetBuilder::buildNetworkWithOneInput("RNN_Net", id_dim, "FP32");
    for (int i = 0; i < state_num; i++)
        net_b.addInputLayer("FP32", st_dim);

    if (mode == DYN_SEQ)
        net_b.addInputLayer("FP32", seq_l_dim);

    if (mode == CELL)
        add_CELL(net_b);
    else if (mode == SEQ || mode == DYN_SEQ)
        add_SEQ(net_b);
    else {
        add_TI(net_b);
    }

    size_t num_input = 1 + state_num + (mode == DYN_SEQ ? 1 : 0);
    vector<pair<string, string>> edges;

    switch (num_input) {
        case 4:
            edges = {
                    {"0,0", "4,4"},
                    {"1,1", "4,5"},
                    {"2,2", "4,6"},
                    {"3,3", "4,7"},
            };
            break;
        case 3:
            edges = {
                    {"0,0", "3,3"},
                    {"1,1", "3,4"},
                    {"2,2", "3,5"},
            };
            break;
        case 2:
            edges = {
                    {"0,0", "2,2"},
                    {"1,1", "2,3"},
            };
            break;
    }
    return net_b.finish(&edges);
}

static const std::string cell_type(Cell cell) {
    return cell == LSTM ? "LSTM" :
           cell == GRU ? "GRU" :
           cell == GRU_lbr ? "GRU" :
           cell == RNN ? "RNN" : "Unknown";
}

static const std::string cell_layer_type(CellDesc cell) {
    return cell_type(cell.type) + "Cell";
}

map<string, string> RNNGen::basic_cell_attr() {
    map<string, string> attr{};

    // Prepare activations attributes
    string algs, alpha, beta;
    for (auto &act : cell.acts) {
        algs += act.alg + ',';
        alpha += std::to_string(act.alpha) + ',';
        beta += std::to_string(act.beta) + ',';
    }
    algs.pop_back(); // remove last comma
    alpha.pop_back();
    beta.pop_back();

    attr["activations"] = algs;
    attr["activations_alpha"] = alpha;
    attr["activations_beta"] = beta;

    attr["clip"] = std::to_string(cell.clip);
    attr["hidden_size"] = std::to_string(S);

    if (cell.type == GRU_lbr)
        attr["linear_before_reset"] = std::to_string(true);

    return attr;
}

void RNNGen::add_TI(CommonTestUtils::V2NetBuilder &builder) {
    /// Generate TI body
    Shape id_ti = id_dim;
    Shape od_ti = od_dim;
    id_ti[axis] = 1;
    od_ti[axis] = 1;

    std::map<std::string, std::string>
            cell_attr = basic_cell_attr(),
            rsh1_attr{{"dim", "-1," + std::to_string(D)}},
            rsh2_attr{{"dim", (axis == 1 ? "-1,1," : "1,-1,") + std::to_string(S)}},
            negt_attr{{"scale", "-1"},
                      {"shift", "0"},
                      {"power", "1"}};

    CommonTestUtils::InOutShapes cell_inout{{{N, D}},
                                              {}};
    for (int i = 0; i < state_num; i++) {
        cell_inout.inDims.push_back({N, S});
        cell_inout.outDims.push_back({N, S});
    }

    auto body_bilder = CommonTestUtils::V2NetBuilder::buildBody();
    body_bilder.addLayer("Reshape", "FP32", &rsh1_attr, {{id_ti},
                                                         {{N, D}}});
    body_bilder.addLayer(cell_layer_type(cell), "FP32", &cell_attr, cell_inout, wSzB, bSzB);
    body_bilder.addLayer("Reshape", "FP32", &rsh2_attr, {{{N, S}},
                                                         {od_ti}});
    if (neg)
        body_bilder.addLayer("Power", "FP32", &negt_attr, {{od_ti},
                                                           {od_ti}});

    // body edges
    int last_l = 2, last_p = 6;
    vector<pair<string, string>> body_edges{
            {"0,1", "1,2"},
            {"1,4", "2,5"}};

    if (state_num == 2) {
        body_edges[1] = {"1,5", "2,7"};
        last_p += 2;
    }

    if (neg) {
        using std::to_string;
        body_edges.push_back({to_string(last_l) + ',' + to_string(last_p),
                              to_string(last_l + 1) + ',' + to_string(last_p + 1)});
        last_l += 1;
        last_p += 2;
    }

    auto body = body_bilder.finish(&body_edges);
    /// body is generated

    bool fwd = (dir == FWD);

    int st = fwd ? 1 : -1;
    int bgn = fwd ? 0 : -1;
    int end = fwd ? -1 : 0;

    CommonTestUtils::InOutShapes ti_inout{{id_dim},
                                            {od_dim}};
    for (int i = 0; i < state_num; i++) {
        ti_inout.inDims.push_back({N, S});
        ti_inout.outDims.push_back({N, S});
    }

    int &ll = last_l, lp = last_p;
    if (state_num == 2) {
        builder.TILayer(ti_inout, body,
                /* frm_l | frm_p | to_l | to_p | axis | step | start | end */
                        {{3, 3, 0, 0, axis, st, bgn, end},
                         {3, 4, 1, 3, -1},
                         {3, 5, 1, 4, -1}},
                        {{3, 6, ll, lp, axis, st, bgn, end},
                         {3, 7, 1,  5,  -1},
                         {3, 8, 1,  6,  -1}},
                        {{1, 5, 1, 3},
                         {1, 6, 1, 4}});
    } else {
        builder.TILayer(ti_inout, body,
                /* frm_l | frm_p | to_l | to_p | axis | step | start | end */
                        {{2, 2, 0, 0, axis, st, bgn, end},
                         {2, 3, 1, 3, -1}},
                        {{2, 4, ll, lp, axis, st, bgn, end},
                         {2, 5, 1,  4,  -1}},
                        {{1, 4, 1, 3}});
    }
}

void RNNGen::add_SEQ(CommonTestUtils::V2NetBuilder &builder) {
    map<string, string> seq_attr = basic_cell_attr();

    string direction = dir == FWD ? "Forward" :
                       dir == BWD ? "Backward" :
                       dir == BDR ? "Bidirectional" :
                       "Unknown";

    seq_attr["direction"] = direction;
    seq_attr["axis"] = std::to_string(axis);

    CommonTestUtils::InOutShapes inout{{id_dim},
                                         {od_dim}};
    for (int i = 0; i < state_num; i++) {
        inout.inDims.push_back({N, S});
        inout.outDims.push_back({N, S});
    }

    if (mode == DYN_SEQ) {
        inout.inDims.push_back(seq_l_dim);
    }

    auto seq_type = cell_type(cell.type) + "Sequence";
    builder.addLayer(seq_type, "FP32", &seq_attr, inout, wSzB, bSzB);
}

void RNNGen::add_CELL(CommonTestUtils::V2NetBuilder &builder) {
    auto id = Shape{N, D};
    auto od = Shape{N, S};

    map<string, string> cell_p = {{"hidden_size", std::to_string(S)}};
    builder.addLayer("LSTMCell", "FP32", &cell_p,
                     {{id, {N, S}, {N, S}},
                      {od, {N, S}, {N, S}}},
                     wSzB, bSzB);
}

CNNNetwork RNNGen::net() {
    referee->wFiller(w_blob);
    referee->bFiller(b_blob);

    Core ie;
    return ie.ReadNetwork(model(), weights);
}

const std::vector<Filler> RNNGen::fillers() const {
    auto fillers = referee->getDataFillers();

    if (dir == BWD)
        // Reverse seq dim for input and output
        fillers[0] = reverse(fillers[0], 1);

    if (axis == 0)
        // Swap N and T dims
        fillers[0] = permute(fillers[0], {1, 0, 2});

    // filler for sequence length tensor
    if (mode == DYN_SEQ) {
        using namespace std::placeholders;
        fillers.push_back(std::bind(scalar_filler, _1, SizeVector{N}, T - 1));

        auto zero_shape = id_dim;
        zero_shape[axis] = 1;
        Filler zero_filler(std::bind(scalar_filler, _1, zero_shape, 0.0f));

        fillers[0] = concat(fillers[0], zero_filler, axis);
    }
    return fillers;
}

const std::vector<Checker> RNNGen::checkers() const {
    auto checkers = referee->getDataChecker();

    if (mode == TI_CSTM)
        // Negative data blob checker. Customization is negative Power layer at the end of TI body
        checkers[0] = negative(checkers[0]);

    if (dir == BWD)
        // Reverse seq dim for input and output
        checkers[0] = reverse(checkers[0], 1);

    if (axis == 0)
        // Swap N and T dims
        checkers[0] = permute(checkers[0], {1, 0, 2});

    if (mode == DYN_SEQ) {
        using namespace std::placeholders;

        auto zero_shape = od_dim;
        zero_shape[axis] = 1;
        Checker zero_checker(std::bind(scalar_checker, _1, zero_shape, 0.0f));

        checkers[0] = concat(checkers[0], zero_checker, axis);
    }

    return checkers;
}

