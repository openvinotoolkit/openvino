// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ov_models/builders.hpp>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "ie_common.h"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

class InputNoReorderEltwiseBF16 : virtual public SubgraphBaseStaticTest, public CPUTestsBase {
protected:
    void SetUp() override {
        auto netPrecision = inType = ov::element::f32;
        outType = ov::element::bf16;
        targetDevice = ov::test::utils::DEVICE_CPU;
        ov::AnyMap additional_config{ov::hint::inference_precision(ov::element::bf16)};
        configuration.insert(additional_config.begin(), additional_config.end());

        ov::Shape inputShape{2, 4, 4, 1};
        auto eltwiseType = ov::test::utils::EltwiseTypes::ADD;

        ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(netPrecision, inputShape)};

        auto tensor = ov::test::utils::create_and_fill_tensor(netPrecision, inputShape);
        auto secondaryInput = std::make_shared<ov::op::v0::Constant>(tensor);

        auto eltwise = ngraph::builder::makeEltwise(input[0], secondaryInput, eltwiseType);

        function = makeNgraphFunction(netPrecision, input, eltwise, "Eltwise");
    }
};

/* FP32 network with enforced BF16 precision.
 * Test that no Reorder (or Convert) is inserted after Input.
 * Eltwise performs the conversion by itself.

    Input[FP32]        Constant[FP32]
          \                 /
           \               /
            X  No Reorder X
             \           /
           Eltwise[FP32->BF16] (or Subgraph[FP32->BF16])
                  |
                  |
             Output[BF16]
*/
TEST_F(InputNoReorderEltwiseBF16, smoke_CompareWithRefs) {
    run();

    CheckNumberOfNodesWithType(compiledModel, "Reorder", 0);
    CheckNumberOfNodesWithType(compiledModel, "Convert", 0);
    CheckNumberOfNodesWithTypes(compiledModel, {"Eltwise", "Subgraph"}, 1);
}
}  // namespace test
}  // namespace ov
