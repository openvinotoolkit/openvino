// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "xml_net_builder.hpp"
#include "rnn_referee.hpp"
#include <cpp/ie_cnn_network.h>

#include <vector>
#include <string>

enum Mode {
    CELL,       /**< single LSTMCell layer */
    SEQ,        /**< single LSTMSeq layer */
    DYN_SEQ,    /**< single LSTMSeq layer with seq length input*/
    TI,         /**< TI layer with LSTM body */
    TI_CSTM     /**< TI layer with LSTM plus negative at the body */
};

enum Direction {
    FWD,        /**< Forward. With stride 1  */
    BWD,        /**< Backward. WIth stride -1 */
    BDR         /**< Bidirectional. With stride 1 and -1 */
};

/**
 *  Topology generator for some RNN specific cases
 */
class RNNGen {
public:
    /** Sequence topology */
    RNNGen(size_t batch, size_t seq, CellDesc cell, Mode mode, Direction dir, int axis);

    const std::vector<Filler> fillers() const;
    const std::vector<Checker> checkers() const;

    InferenceEngine::CNNNetwork net();

private:
    const size_t D = 10;  // Data size
    const size_t S = 5;   // State size
    const size_t G = 4;   // Number of gate

    const size_t N;  // Batch
    const size_t T;  // Sequence
    const int axis;  // Axis of sequence

    const Mode mode;
    const CellDesc cell;
    const Direction dir;
    const bool neg;

    size_t state_num = 0;

    size_t wSzB = 0;
    size_t bSzB = 0;

    InferenceEngine::SizeVector seq_l_dim, st_dim, id_dim, od_dim;

    InferenceEngine::TBlob<uint8_t>::Ptr weights;
    InferenceEngine::Blob::Ptr w_blob, b_blob;

    std::shared_ptr<RNN_Referee> referee;

private:
    std::string model();

    void add_TI(CommonTestUtils::V2NetBuilder &builder);
    void add_SEQ(CommonTestUtils::V2NetBuilder &builder);
    void add_CELL(CommonTestUtils::V2NetBuilder &builder);

    std::map<std::string, std::string> basic_cell_attr();
};
