// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/three_inputs_eltwise.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {
namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, ThreeInputsEltwise,
                     ::testing::Combine(
                             ::testing::Values(ov::Shape {1, 64, 10, 10}),
                             ::testing::Values(ov::Shape {1, 64, 10,  1}),
                             ::testing::Values(ov::Shape {1, 1, 1,  10}),
                             ::testing::Values(2), // eltwises fuse only for non-broadcasted shapes
                             ::testing::Values(0), // SnippetsMarkSkipped disables tokenization for eltwise chains after inputs
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         ThreeInputsEltwise::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, ThreeInputsEltwiseSinh,
        ::testing::Combine(
        ::testing::Values(ov::Shape {1, 64, 10, 10}),
        ::testing::Values(ov::Shape {1, 64, 10,  1}),
        ::testing::Values(ov::Shape {1, 1, 1,  10}),
        ::testing::Values(4), // Subgraph + 3 converts after inputs
        ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         ThreeInputsEltwiseSinh::getTestCaseName);

namespace snippets_dynamic_1 {
// test when all inputs are dynamic
InputShape inShapesDynamic1 = {{16, 3, ngraph::Dimension(1, 512)}, {{16, 3, 63}}};
std::vector<InputShape> inShapesDynamic2 = {{{16, 3, ngraph::Dimension(1, 512)}, {{16, 3, 63}}},
                                            {{16, 3, ngraph::Dimension(1, 512)}, {{16, 3, 1}}},
                                            {{16, 1, ngraph::Dimension(1, 512)}, {{16, 1, 1}}},
                                            {{16, 1, ngraph::Dimension(1, 512)}, {{16, 1, 63}}}};
INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_Eltwise,
    ThreeInputsEltwiseSinhDynamic,
    ::testing::Combine(::testing::Values(inShapesDynamic1),
                       ::testing::ValuesIn(inShapesDynamic2),
                       ::testing::ValuesIn(inShapesDynamic2),
                       ::testing::Values(4),  // Subgraph + 3 converts after inputs
                       ::testing::Values(1),  // Subgraph is created, since the inputs are followed by converts
                       ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    ThreeInputsEltwiseSinhDynamic::getTestCaseName);
} //namespace snippets_dynamic_1

namespace snippets_dynamic_2 {
// test dynamic + static shapes combination: vector inner Tile (multiple iterations)
InputShape inShapesDynamic1 = {{16, 3, ngraph::Dimension(1, 512)}, {{16, 3, 32}}};
std::vector<InputShape> inShapesDynamic2 = {{{16, 3, ngraph::Dimension(1, 512)}, {{16, 3, 32}}},
                                            {{16, 3, ngraph::Dimension(1, 512)}, {{16, 3, 1}}}};

std::vector<InputShape> inShapesDynamic3 = {{{16, 3, 32}, {{16, 3, 32}}},
                                            {{16, 3, 1}, {{16, 3, 1}}}};
INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_Eltwise,
    ThreeInputsEltwiseSinhDynamic,
    ::testing::Combine(::testing::Values(inShapesDynamic1),
                       ::testing::ValuesIn(inShapesDynamic2),
                       ::testing::ValuesIn(inShapesDynamic3),
                       ::testing::Values(4),  // Subgraph + 3 converts after inputs
                       ::testing::Values(1),  // Subgraph is created, since the inputs are followed by converts
                       ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    ThreeInputsEltwiseSinhDynamic::getTestCaseName);
} //namespace snippets_dynamic_2

namespace snippets_dynamic_3 {
// test dynamic + static shapes combination: scalar inner Tile (multiple iterations)
InputShape inShapesDynamic1 = {{16, 29, ngraph::Dimension(1, 512)}, {{16, 29, 7}}};
std::vector<InputShape> inShapesDynamic2 = {{{16, 29, ngraph::Dimension(1, 512)}, {{16, 29, 7}}},
                                            {{16, 29, ngraph::Dimension(1, 512)}, {{16, 29, 1}}}};

std::vector<InputShape> inShapesDynamic3 = {{{16, 29, 1}, {{16, 29, 1}}},
                                            {{16, 29, 7}, {{16, 29, 7}}}};
INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_Eltwise,
    ThreeInputsEltwiseSinhDynamic,
    ::testing::Combine(::testing::Values(inShapesDynamic1),
                       ::testing::ValuesIn(inShapesDynamic2),
                       ::testing::ValuesIn(inShapesDynamic3),
                       ::testing::Values(4),  // Subgraph + 3 converts after inputs
                       ::testing::Values(1),  // Subgraph is created, since the inputs are followed by converts
                       ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    ThreeInputsEltwiseSinhDynamic::getTestCaseName);
} //namespace snippets_dynamic_3

namespace snippets_dynamic_4 {
// test dynamic + static shapes combination: vector + scalar inner Tiles
InputShape inShapesDynamic1 = {{16, 12, ngraph::Dimension(1, 512)}, {{16, 12, 63}}};
std::vector<InputShape> inShapesDynamic2 = {{{16, 1, ngraph::Dimension(1, 512)}, {{16, 1, 63}}},
                                            {{16, 12, ngraph::Dimension(1, 512)}, {{16, 12, 63}}},
                                            {{16, 1, ngraph::Dimension(1, 512)}, {{16, 1, 1}}}};

std::vector<InputShape> inShapesDynamic3 = {{{16, 1, 1}, {{16, 1, 1}}},
                                            {{16, 12, 1}, {{16, 12, 1}}},
                                            {{16, 12, 63}, {{16, 12, 63}}},
                                            {{16, 1, 63}, {{16, 1, 63}}}};
INSTANTIATE_TEST_SUITE_P(
    smoke_Snippets_Eltwise,
    ThreeInputsEltwiseSinhDynamic,
    ::testing::Combine(::testing::Values(inShapesDynamic1),
                       ::testing::ValuesIn(inShapesDynamic2),
                       ::testing::ValuesIn(inShapesDynamic3),
                       ::testing::Values(4),  // Subgraph + 3 converts after inputs
                       ::testing::Values(1),  // Subgraph is created, since the inputs are followed by converts
                       ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    ThreeInputsEltwiseSinhDynamic::getTestCaseName);
} //namespace snippets_dynamic_4

}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov