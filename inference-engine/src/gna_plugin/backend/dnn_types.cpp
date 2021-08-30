// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnn_types.h"

const char *intel_dnn_activation_name[kActNumType] = {
        "kActNone",
        "kActSigmoid",
        "kActTanh",
        "kActRelu",
        "kActLeakyRelu",
        "kActIdentity",
        "kActKaldiLstmClipping",
        "kActCustom",
        "kActExp",
        "kActLog",
        "kActSign",
        "kActAbs",
        "kActNegLog",
        "kActNegHalfLog",
        "kActSoftSign",
        "kActPow",
        "kActFakeQuantize"
};

const char *intel_dnn_softmax_name[kSoftmaxNumType] = {
        "kSoftmaxNone",
        "kSoftmaxKaldiSumGroup",
        "kSoftmaxKaldiApplyLog",
        "kSoftmaxGoogle"
};

const char* intel_dnn_operation_name[kDnnNumOp] = {
        "kDnnNullOp",
        "kDnnAffineOp",
        "kDnnDiagonalOp",
        "kDnnConvolutional1dOp",
        "kDnnConvolutional2dOp",
        "kDnnPiecewiselinearOp",
        "kDnnMaxPoolOp",
        "kDnnRecurrentOp",
        "kDnnInterleaveOp",
        "kDnnDeinterleaveOp",
        "kDnnCopyOp"
};

const char *intel_dnn_macro_operation_name[kDnnNumMacroOp] = {
        "kDnnMacroOpNone",
        "kDnnMacroOpLstm",
        "kDnnMacroOpBiLstm"
};

const char *intel_dnn_number_type_name[kDnnNumNumberType] = {
        "kDnnFloat",
        "kDnnInt"
};
