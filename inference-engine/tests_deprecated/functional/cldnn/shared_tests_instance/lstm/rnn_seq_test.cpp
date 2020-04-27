// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rnn_seq_test.hpp"

RUN_CASE_CP_WITH_SUFFIX(GPU, _smoke, RNNSeqTest, workload);

RUN_CASE_CP_WITH_SUFFIX(GPU, _smoke_seq, RNNSeqTest, dyn_seq_workload);
