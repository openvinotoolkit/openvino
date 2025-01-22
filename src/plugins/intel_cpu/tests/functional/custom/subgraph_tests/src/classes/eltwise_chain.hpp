// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <string>
#include <vector>
#include <memory>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

typedef std::tuple<std::vector<InputShape>, // Input shapes
        ov::test::utils::InputLayerType,    // Secondary input type
        std::vector<ElementType>,           // Input precisions
        std::vector<ov::test::utils::EltwiseTypes>,  // Eltwise operations
        bool,                               // With quantization
        ov::element::Type,                  // Conversion type
        std::string                         // Device name
        >
EltwiseChainTuple;

class EltwiseChainTest : public testing::WithParamInterface<EltwiseChainTuple>,
                         virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<EltwiseChainTuple> &obj);
    ov::Tensor generate_eltwise_input(const ov::element::Type& type, const ov::Shape& shape);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;

protected:
    void SetUp() override;
};

namespace eltwise_chain {
std::vector<std::vector<ov::Shape>> inputShapes();
std::vector<std::vector<ElementType>> inputPrecisions();
std::vector<std::vector<ov::test::utils::EltwiseTypes>> eltwiseOps();
std::vector<std::vector<ov::Shape>> inputShapesConvert();
std::vector<std::vector<ov::test::utils::EltwiseTypes>> eltwiseOpsConvert();
std::vector<std::vector<ElementType>> inputPrecisionsConvert();
}  // namespace eltwise_chain
}  // namespace test
}  // namespace ov
