// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
            //TODO: Issue: 34748
            R"(.*(ComparisonLayerTest).*)",
            // TODO: Issue: 39014
            R"(.*CoreThreadingTestsWithIterations.*smoke_LoadNetwork.*)",
            // TODO: Issue: 39612
            R"(.*Interpolate.*cubic.*tf_half_pixel_for_nn.*FP16.*)",
            // Expected behavior
            R"(.*EltwiseLayerTest.*eltwiseOpType=Pow.*netPRC=I64.*)",
            R"(.*EltwiseLayerTest.*IS=\(.*\..*\..*\..*\..*\).*eltwiseOpType=Pow.*secondaryInputType=CONSTANT.*)",
            // TODO: Issue: 43794
            R"(.*(PreprocessTest).*(SetScalePreProcess).*)",
            R"(.*(PreprocessTest).*(ReverseInputChannelsPreProcess).*)",
            // TODO: Issue: 41467 -- "unsupported element type f16 op Convert"
            R"(.*(ConvertLayerTest).*targetPRC=FP16.*)",
            // TODO: Issue: 41462
            R"(.*(SoftMaxLayerTest).*axis=0.*)",
            // TODO: Issue: 41461
            R"(.*TopKLayerTest.*k=10.*mode=min.*sort=index.*)",
            R"(.*TopKLayerTest.*k=5.*sort=(none|index).*)",
            // TODO: Issue: 43511
            R"(.*EltwiseLayerTest.*IS=\(1.4.3.2.1.3\).*)",
            R"(.*EltwiseLayerTest.*IS=\(2\).*OpType=Mod.*opType=VECTOR.*)",
            R"(.*EltwiseLayerTest.*OpType=FloorMod.*netPRC=I64.*)",
            // TODO: Issue: 46841
            R"(.*(QuantGroupConvBackpropData3D).*)",

            // These tests might fail due to accuracy loss a bit bigger than threshold
            R"(.*(GRUCellTest).*)",
            R"(.*(RNNSequenceTest).*)",
            R"(.*(GRUSequenceTest).*)",
            // These test cases might fail due to FP16 overflow
            R"(.*(LSTM).*activations=\(relu.*netPRC=FP16.*)",

            // Need to update activation primitive to support any broadcastable constant to enable these cases.
            R"(.*ActivationParamLayerTest.*)",
            // Unknown issues
            R"(.*(LSTMSequence).*mode=CONVERT_TO_TI_RAND_SEQ_LEN.*)",
            R"(.*(smoke_DetectionOutput3In).*)",
            R"(.*(smoke_DetectionOutput5In).*)",
            R"(.*(ScatterUpdateLayerTest).*)",
    };
}
