// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/core_config.hpp"
#include "common_test_utils/file_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "openvino/util/file_util.hpp"

#include "conformance.hpp"

// todo: remove as LayerTestBase will be removed
void CoreConfiguration(LayerTestsUtils::LayerTestsCommon* test) {}

namespace ov {
namespace test {

void core_configuration(ov::test::SubgraphBaseTest* test) {
    ov::element::Type hint = ov::element::f32;
    for (auto& param : test->function->get_parameters()) {
        if (param->get_output_element_type(0) == ov::element::f16) {
            hint = ov::element::f16;
            break;
        }
    }

    // Set inference_precision hint to run fp32 model in fp32 runtime precision as default plugin execution precision
    // may vary
    if (test->targetDevice == ov::test::utils::DEVICE_GPU)  {
        test->core->set_property(ov::test::utils::DEVICE_GPU, {{ov::hint::inference_precision.name(), hint.get_type_name()}});
    } else if (test->targetDevice == ov::test::utils::DEVICE_CPU) {
        test->convert_precisions.insert({ov::element::bf16, ov::element::f32});
        test->convert_precisions.insert({ov::element::f16, ov::element::f32});
    }
}

} // namespace test
} // namespace ov