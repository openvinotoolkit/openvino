// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gna/gna_config.hpp>
#include "lstm_cell_test.hpp"

#if defined GNA_LIB_VER && GNA_LIB_VER == 2
# define DISABLE_TEST_ON_GNA2 GTEST_SKIP();
#else
# define DISABLE_TEST_ON_GNA2
#endif

TEST_P(LSTMCellTestBase, GNA_sw_fp32_single_lstm_test) {
    runSingleLSTMTest({{"GNA_DEVICE_MODE", "GNA_SW_FP32"}, {"GNA_COMPACT_MODE", "NO"}});
}

TEST_P(LSTMCellTestBase, GNA_I16_single_lstm_test) {
    DISABLE_TEST_ON_GNA2
    runSingleLSTMTest( {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_COMPACT_MODE", "NO"},
        {"GNA_PRECISION", "I16"},
        {"GNA_SCALE_FACTOR_0", "1024"},
        {"GNA_SCALE_FACTOR_1", "1024"},
        {"GNA_SCALE_FACTOR_2", "1024"}
    }, 0.099);
}

TEST_P(LSTMCellTestBase, GNA_I8_single_lstm_test) {
    DISABLE_TEST_ON_GNA2
    runSingleLSTMTest({
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_COMPACT_MODE", "NO"},
        {"GNA_PRECISION", "I8"},
        {"GNA_SCALE_FACTOR_0", "1024"},
        {"GNA_SCALE_FACTOR_1", "1024"},
        {"GNA_SCALE_FACTOR_2", "1024"}
    }, 0.011);
}

static const lstm_cell_param gna_workload[] = {{1, StateSize, DataSize}, {1, 16, 16}};

RUN_CASE_P_WITH_SUFFIX(GNA, _smoke, LSTMCellTestBase, gna_workload);
