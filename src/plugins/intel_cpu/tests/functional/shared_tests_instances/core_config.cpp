// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

void core_configuration(ov::test::SubgraphBaseTest* test) {
    #if defined(OV_CPU_ARM_ENABLE_FP16) || defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
        //force fp32 inference precision if it is not configured specially
        if (!test->configuration.count(ov::hint::inference_precision.name())) {
            test->configuration.insert({ov::hint::inference_precision.name(), ov::element::f32.to_string()});
        }
    #endif
        // todo: issue: 123320
        test->convert_precisions.insert({ov::element::bf16, ov::element::f32});
        test->convert_precisions.insert({ov::element::f16, ov::element::f32});

        // Enable CPU pinning in CPU funtional tests to save validation time of Intel CPU plugin func tests (parallel)
        // on Windows
        test->configuration.insert({ov::hint::enable_cpu_pinning.name(), true});
}

} // namespace test
} // namespace ov
