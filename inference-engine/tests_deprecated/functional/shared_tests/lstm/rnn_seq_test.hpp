// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn_gen.hpp"
#include "plg_test.hpp"

#include <cmath>
#include <vector>
#include <string>

using namespace ::testing;
using namespace InferenceEngine;
using std::map;
using std::pair;
using std::vector;
using std::string;

enum Reshape {
    RESH_NO = 0, /**< No reshape step */
    RESH_B  = 1, /**< Reshape for batch dim */
    RESH_T  = 2, /**< Reshape for time dim */
    RESH_BT = 3  /**< Reshape for both batch and time dims */
};

using rnn_param = std::tuple<
    CellDesc,    /**< cell  - Descriptor of RNN cell */
    float,       /**< clip  - Clip value */
    Direction,   /**< fwd   - Direction */
    Mode,        /**< mode  - Modes of LSTM representation */
    size_t,      /**< N     - Batch size */
    size_t,      /**< T     - Sequence length */
    size_t,      /**< axis  - Dimension with T */
    Reshape      /**< shape - Apply reshape. +1 to original dim */
>;

const Named<rnn_param> test_name( [] (const rnn_param &p) {
    CellDesc _cell; Direction _dir; Mode _mode; Reshape _resh;
    size_t _N, _S, _axis;
    float _clip;

    std::tie(_cell,_clip,_dir,_mode,_N,_S,_axis,_resh) = p;

    string res = _cell.type == LSTM ? "LSTM_" : _cell.type == GRU  ? "GRU__"  : "RNN__";
    for (auto &act : _cell.acts) res += act.alg[0];

    res += _dir == FWD ? "_FWD" : _dir == BWD ? "_BWD" : _dir == BDR ? "_BDR" : "_XXX";
    res += _mode == SEQ ? "_SEQ" : _mode == TI ? "__TI" : _mode == TI_CSTM ? "_TIX" : "_XXX";
    res += (_clip == 0.0f) ? "_c0" : "_cX";
    res += "_b" + std::to_string(_N);
    res += "_s" + std::to_string(_S);
    res += "_axis" + std::to_string(_axis);

    res += _resh == RESH_NO ? "_reshNO" :
           _resh == RESH_B  ? "__reshB" :
           _resh == RESH_T  ? "__reshT" :
           _resh == RESH_BT ? "_reshBT" : "_X";
    return res;
});

using RNNSeqTest = PlgTest<rnn_param>;

// disabled due to transition to ngraph transformation
// DO NOT DELETE, part of the functionality is still needed
TEST_P(RNNSeqTest, DISABLED_SingleRNN) {
    auto p = param();

    auto cell = std::get<0>(p);
    auto clip = std::get<1>(p);
    auto dir  = std::get<2>(p);
    auto mode = std::get<3>(p);
    auto N    = std::get<4>(p);
    auto T    = std::get<5>(p);
    auto axis = std::get<6>(p);
    auto resh = std::get<7>(p);

    if (device_name == "GPU" && cell.type != LSTM)
        GTEST_SKIP();

    cell.clip = clip;

    /************ Test Body  *****************************/

    RNNGen topology(N, T, cell, mode , dir, axis);
    auto net = topology.net();
    auto fillers = topology.fillers();
    auto checkers = topology.checkers();

    // Reshape if requested
    if (resh != RESH_NO) {
        const bool resh_b = resh & RESH_B;
        const bool resh_t = resh & RESH_T;

        auto shapes = net.getInputShapes();
        for (auto &pair : shapes) {
            // Blobs with data
            if (pair.second.size() == 3) {
                if (resh_b) pair.second[(axis+1)%2]++;
                if (resh_t) pair.second[axis]++;
            }
            // Blobs with state or Seq Len
            if (pair.second.size() == 1 || pair.second.size() == 2) {
                if (resh_b) pair.second[0]++;
            }
        }
        net.reshape(shapes);

        // Also need new fillers/checkers for new shapes
        RNNGen resh_topology(resh_b ? N+1 : N, resh_t ? T+1 : T, cell, mode , dir, axis);
        fillers = resh_topology.fillers();
        checkers = resh_topology.checkers();
    }

    Core ie;
    auto execNet = ie.LoadNetwork(net, device_name);
    auto req = execNet.CreateInferRequest();

    ASSERT_TRUE(net.getInputsInfo().size() == fillers.size());
    ASSERT_TRUE(net.getOutputsInfo().size() == checkers.size());

    int i = 0;
    for (auto &info : net.getInputsInfo())
        fillers[i++](req.GetBlob(info.first));

    req.Infer();

    i = 0;
    for (auto &info : net.getOutputsInfo())
        EXPECT_TRUE(checkers[i++](req.GetBlob(info.first))) << "Error with #" << i << " output";
}

const std::vector<CellDesc> cells = {
  /** LSTM modifications */
  {LSTM, {{"sigmoid",0,0}, {"tanh",0,0}, {"tanh",0,0}} }, // default
  {LSTM, {{"tanh",0,0}, {"sigmoid",0,0}, {"relu",0,0}} },
  /** GRU modifications */
  {GRU , {{"sigmoid",0,0}, {"tanh",0,0}} }, // default
  {GRU , {{"tanh",0,0}, {"relu",0,0}} },
  /** GRU linear_before_reset modifications */
  {GRU_lbr , {{"sigmoid",0,0}, {"tanh",0,0}} }, // default
  {GRU_lbr , {{"tanh",0,0}, {"relu",0,0}} },
  /** RNN modifications */
  {RNN , {{"tanh",0,0}} },   // default
  {RNN , {{"sigmoid",0,0}} },
  {RNN , {{"relu",0,0}} },
};

#if 0
// All combination of next parameters
const auto workload = Combine(
    ValuesIn(cells),          // Cell desc
    Values(0.0f, 0.7f),       // Clip arg
    Values(FWD, BWD),         // Direction
    Values(SEQ, DYN_SEQ,      // Representation mode
           TI, TI_CSTM),      //
    Values(1, 3),             // Batch
    Values(3),                // Sequence size
    Values(0, 1),             // Axis of sequence
    Values(RESH_NO, RESH_B,   // Reshape mode for batch, sequence or both
           RESH_T, RESH_BT)   //
);
#else
// All combination of next parameters ( small subset for fast CI testing)
const auto workload = Combine(
        ValuesIn(cells.begin(),     // Cell desc (only first 5)
                 cells.begin()+7),  //
        Values(0.0f, 0.7f),         // Clip arg
        Values(FWD, BWD),           // Direction
        Values(SEQ, TI),            // Representation mode
        Values(2),                  // Batch
        Values(3),                  // Sequence size
        Values(0, 1),               // Axis of sequence
        Values(RESH_NO /*, RESH_B TODO: migrate to ngraph reshape */)  // Reshape mode for batch, sequence or both
);
#endif

// All combination of next parameters ( small subset for fast CI testing)
const auto dyn_seq_workload = Combine(
        ValuesIn(std::vector<CellDesc> {cells[0], cells[2], cells[4], cells[6]}),
        Values(0.0f),               // Clip arg
        Values(FWD, BWD, BDR),      // Direction
        Values(DYN_SEQ),            // Representation mode
        Values(1, 8),               // Batch
        Values(3, 100),             // Sequence size
        Values(0, 1),               // Axis of sequence
        Values(RESH_NO /*, RESH_B TODO: migrate to ngraph reshape */)     // Reshape mode for batch, sequence or both
);
