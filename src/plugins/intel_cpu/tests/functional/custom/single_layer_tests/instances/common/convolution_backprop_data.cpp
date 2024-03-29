// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/convolution_backprop_data.hpp"
#include "shared_test_classes/single_op/convolution_backprop_data.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

TEST_P(DeconvolutionLayerCPUTest, CompareWithRefs) {
    if (!fusedOps.empty()) {
        bool isSupportedParams = stride[stride.size() - 1] <= kernel[kernel.size() - 1];
        if (stride.size() > 1)
            isSupportedParams &= stride[stride.size() - 2] <= kernel[kernel.size() - 2];
        if (stride.size() > 2)
            isSupportedParams &= stride[stride.size() - 3] <= kernel[kernel.size() - 3];
        if (!isSupportedParams) {
            GTEST_SKIP() << "Fusing with strides more than kernel size was disabled, because oneDNN deconvolution "
                            "doesn't support it"
                         << std::endl;
        }
    }

    run();
    CheckPluginRelatedResults(compiledModel, "Deconvolution");
}
