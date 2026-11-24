// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/node_builders/activation.hpp"
#include "common_test_utils/node_builders/comparison.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/eltwise.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

enum class PostNode : uint8_t { EMPTY, QUANTIZE, SOFTSIGN, LOGICAL_NOT };

inline std::ostream& operator<<(std::ostream& os, PostNode node) {
    switch (node) {
        case PostNode::EMPTY:    os << "EMPTY"; break;
        case PostNode::QUANTIZE: os << "QUANTIZE"; break;
        case PostNode::LOGICAL_NOT:
            os << "LOGICAL_NOT";
            break;
        case PostNode::SOFTSIGN: os << "SOFTSIGN"; break;
        default:                 os << "UNKNOWN"; break;
    }
    return os;
}

// An operation of the chain. A comparison or activation can only be the last one.
using EltwiseChainOp = std::variant<ov::test::utils::EltwiseTypes,
                                    ov::test::utils::ComparisonTypes,
                                    ov::test::utils::ActivationTypes>;

inline std::ostream& operator<<(std::ostream& os, const EltwiseChainOp& op) {
    std::visit([&os](auto type) { os << type; }, op);
    return os;
}

typedef std::tuple<std::vector<InputShape>,           // Input shapes
                   ov::test::utils::InputLayerType,   // Secondary input type
                   std::vector<ElementType>,          // Input precisions
                   std::vector<EltwiseChainOp>,       // Eltwise operations
                   PostNode,                          // { QA | SoftSign | LogicalNot | Nothing }
                   ov::element::Type,                 // Conversion type
                   std::string                        // Device name
                   >
    EltwiseChainTuple;

class EltwiseChainTestBase : virtual public SubgraphBaseTest {
protected:
    static std::string get_chain_case_name(const EltwiseChainTuple& params);
    void setup_chain(const EltwiseChainTuple& params);
    ov::Tensor generate_eltwise_input(const ov::element::Type& type, const ov::Shape& shape, int32_t seed);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};

class EltwiseChainTest : public testing::WithParamInterface<EltwiseChainTuple>, public EltwiseChainTestBase {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<EltwiseChainTuple>& obj) {
        return get_chain_case_name(obj.param);
    }

protected:
    void SetUp() override {
        setup_chain(GetParam());
    }
};

namespace eltwise_chain {
std::vector<std::vector<ov::Shape>> inputShapes();
std::vector<std::vector<ElementType>> inputPrecisions();
std::vector<std::vector<EltwiseChainOp>> eltwiseOps();
std::vector<std::vector<ov::Shape>> inputShapesConvert();
std::vector<std::vector<EltwiseChainOp>> eltwiseOpsConvert();
std::vector<std::vector<ElementType>> inputPrecisionsConvert();
}  // namespace eltwise_chain
}  // namespace test
}  // namespace ov
