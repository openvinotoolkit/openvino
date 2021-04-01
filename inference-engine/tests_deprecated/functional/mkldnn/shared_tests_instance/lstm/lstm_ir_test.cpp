// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lstm_ir_test.hpp"

RUN_CASE_P_WITH_SUFFIX(CPU, _smoke, LSTM_IR_Test, workload);

static std::vector<ModelInfo> hetero_workload { workload };
RUN_CASE_P_WITH_SUFFIX(HETERO_CPU, _smoke, LSTM_IR_Test, hetero_workload);
