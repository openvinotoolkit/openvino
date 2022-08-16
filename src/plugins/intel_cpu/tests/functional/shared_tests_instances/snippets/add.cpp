// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/add.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace snippets {


namespace {

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, Add,
                     ::testing::Combine(
                             ::testing::Values(ov::Shape {1, 42, 16, 64}),
                             ::testing::Values(ov::Shape {1, 42, 16,  1}),
                             ::testing::Values(1), // one node - Add
                             ::testing::Values(0), // SnippetsMarkSkipped disables tokenization for eltwise chains after inputs
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                     Add::getTestCaseName);


namespace snippets_static_1 {
// These  inputs are needed to test static TileScheduler optimizations (emit the whole tile, body with increments, set WA etc)
std::vector<ov::Shape> inShapesStatic1{{1, 16, 29,  1}, {1, 16, 29,  7}, {1, 16, 29,  8}, {1, 16, 29,  15}, {1, 16, 29,  16}, {1, 16, 29,  31}};
INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, AddSinh,
                         ::testing::Combine(
                             ::testing::ValuesIn(inShapesStatic1),
                             ::testing::Values(ov::Shape {1, 16, 29, 1}),
                             ::testing::Values(3), // Add + 2 converts after inputs
                             ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         AddSinh::getTestCaseName);

} // namespace snippets_static_1

INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, AddSinhConst,
                     ::testing::Combine(
                             ::testing::Values(ov::Shape {1, 42, 16, 64}),
                             ::testing::Values(2), // Add + 2 converts after inputs
                             ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
                             ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                     AddSinhConst::getTestCaseName);

namespace snippets_dynamic_1 {
    InputShape inShapesDynamic1 = {{16, 6, ngraph::Dimension(1, 512)}, {{16, 6, 16}}};
    std::vector<InputShape> inShapesDynamic2 = {{{16, 6, ngraph::Dimension(1, 512)}, {{16, 6, 16}}},
                                                {{16, 6, ngraph::Dimension(1, 512)}, {{16, 6, 1}}}};

    INSTANTIATE_TEST_SUITE_P(
        smoke_Snippets_Eltwise,
        AddSinhDynamic,
        ::testing::Combine(::testing::Values(inShapesDynamic1),
                           ::testing::ValuesIn(inShapesDynamic2),
                           ::testing::Values(3),  // Add + 2 converts after inputs
                           ::testing::Values(1),  // Subgraph is created, since the inputs are followed by converts
                           ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        AddSinhDynamic::getTestCaseName);
} // namespace snippets_dynamic_1

namespace snippets_dynamic_7 {
    // the main purpose here is to tests different dimension collapsing strategies and how they might interfere in a single dynamic shape
    PartialShape dynamic_4D(std::vector<Dimension>(4, ngraph::Dimension(0, -1)));
    InputShape inShapesDynamic = {dynamic_4D, {{6, 12, 4, 25}, {2, 32, 1024, 1}, {1, 2, 6, 1024}}};

    INSTANTIATE_TEST_SUITE_P(
            smoke_Snippets_Eltwise,
            AddSinhDynamic,
            ::testing::Combine(::testing::Values(inShapesDynamic),
                               ::testing::Values(inShapesDynamic),
                               ::testing::Values(3),  // Add + 2 converts after inputs
                               ::testing::Values(1),  // Subgraph is created, since the inputs are followed by converts
                               ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            AddSinhDynamic::getTestCaseName);
} // namespace snippets_dynamic_7
}  // namespace
} // namespace snippets
} // namespace test
} // namespace ov