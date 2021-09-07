// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "functional_test_utils/skip_tests_config.hpp"

std::vector<std::string> disabledTestPatterns() {
    return {
            //TODO: Issue: 34748
            R"(.*(ComparisonLayerTest).*)",
            // TODO: Issue: 39612
            R"(.*Interpolate.*cubic.*tf_half_pixel_for_nn.*FP16.*)",
            // Expected behavior
            R"(.*EltwiseLayerTest.*eltwiseOpType=Pow.*netPRC=I64.*)",
            R"(.*EltwiseLayerTest.*IS=\(.*\..*\..*\..*\..*\).*eltwiseOpType=Pow.*secondaryInputType=CONSTANT.*)",
            // TODO: Issue: 43794
            R"(.*(PreprocessTest).*(SetScalePreProcessSetBlob).*)",
            R"(.*(PreprocessTest).*(SetScalePreProcessGetBlob).*)",
            R"(.*(PreprocessTest).*(SetMeanValuePreProcessSetBlob).*)",
            R"(.*(PreprocessTest).*(SetMeanImagePreProcessSetBlob).*)",
            R"(.*(PreprocessTest).*(ReverseInputChannelsPreProcessGetBlob).*)",
            R"(.*(InferRequestPreprocessDynamicallyInSetBlobTest).*)",
            // TODO: Issue: 51764
            ".*InferRequestPreprocessConversionTest.*",
            // TODO: Issue: 41462
            R"(.*(SoftMaxLayerTest).*axis=0.*)",
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
            R"(.*(LSTMSequence).*mode=.*_RAND_SEQ_LEN_CONST.*)",
            R"(.*(smoke_DetectionOutput5In).*)",
            // TODO: Issue: 47773
            R"(.*(ProposalLayerTest).*)",
            // INT8 StridedSlice not supported
            R"(.*(LPT/StridedSliceTransformation).*)",
            // TODO: Issue: 48106
            R"(.*ConstantResultSubgraphTest.*inPrc=I16.*)",
            // TODO: Issue: 54194
            R"(.*ActivationLayerTest.*SoftPlus.*)",
            // need to implement Export / Import
            R"(.*IEClassImportExportTestP.*)",
            R"(.*Behavior.*InferRequestSetBlobByType.*Device=HETERO.*)",
            // TODO: Issue: 59586, NormalizeL2 output mismatch for empty axes case
            R"(.*NormalizeL2LayerTest.*axes=\(\).*)",

            // Not allowed dynamic loop tests on GPU
            R"(.*smoke_StaticShapeLoop_dynamic_exit.*)",
            // CVS-58963: Not implemented yet
            R"(.*Behavior.*InferRequest.*OutOfFirstOutIsInputForSecondNetwork.*)",
            // Not expected behavior
            R"(.*Behavior.*InferRequestIOBBlobSetLayoutTest.*layout=(95|OIHW).*)",
            R"(.*Behavior.*InferRequestIOBBlobSetLayoutTest.*CanSetInBlobWithDifferentLayouts.*layout=NHWC.*)",
            R"(.*Behavior.*InferRequestIOBBlobSetLayoutTest.*CanSetOutBlobWithDifferentLayouts.*layout=(CN|HW).*)",
            R"(.*Behavior_Multi.*InferRequestSetBlobByType.*Batched.*)",
            R"(.*(Multi|Auto).*Behavior.*InferRequestIOBBlobTest.*canProcessDeallocatedOutputBlobAfterGetAndSetBlob.*)",
            // TODO: until issue is xxx-59670 is resolved
            R"(.*Gather8LayerTest.*)"
    };
}
