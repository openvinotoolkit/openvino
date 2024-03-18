// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_fq.hpp"
#include "common_test_utils/data_utils.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include <snippets/op/subgraph.hpp>

namespace ov {
namespace test {
namespace snippets {

std::shared_ptr<ov::Model> ThreeFQFunction::initOriginal() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);
    auto fq0_data = ov::builder::subgraph::FakeQuantizeOnDataWithConstant(256,
                                                                              std::vector<ov::Shape>{{1, 3, 1, 1}, {1, 3, 1, 1}, {1}, {1}},
                                                                              std::vector<float>{-397898.40625, -435929.78125, -476583.9375},
                                                                              std::vector<float>{394789.84375, 432524.09375, 472860.625},
                                                                              std::vector<float>{0},
                                                                              std::vector<float>{255},
                                                                              ov::element::u8);
    auto fq0 = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(data0, ov::element::f32, fq0_data);
    auto fq1_data = ov::builder::subgraph::FakeQuantizeOnDataWithConstant(256,
                                                                              std::vector<ov::Shape>{{1, 3, 1, 1}, {1, 3, 1, 1}, {1}, {1}},
                                                                              std::vector<float>{5.390228, 9.7712707, 9.333160},
                                                                              std::vector<float>{250.524688, 254.905731, 254.467620},
                                                                              std::vector<float>{-128},
                                                                              std::vector<float>{127},
                                                                              ov::element::i8);
    auto fq1 = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(fq0, ov::element::f32, fq1_data);
    auto fq2_data = ov::builder::subgraph::FakeQuantizeOnDataWithConstant(256,
                                                                              std::vector<ov::Shape>{{1, 3, 1, 1}, {1, 3, 1, 1}, {1}, {1}},
                                                                              std::vector<float>{-98.16883, -142.82466, -155.0642700},
                                                                              std::vector<float>{97.334106, 141.599884, 153.8145446},
                                                                              std::vector<float>{0},
                                                                              std::vector<float>{255},
                                                                              ov::element::u8);
    auto fq2 = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(fq1, ov::element::f32, fq2_data);
    return std::make_shared<ov::Model>(NodeVector{fq2}, ParameterVector{data0});
}
std::shared_ptr<ov::Model> ThreeFQFunction::initReference() const {
    auto data0 = std::make_shared<op::v0::Parameter>(precision, input_shapes[0]);

    auto indata0 = std::make_shared<op::v0::Parameter>(precision, data0->get_shape());
    auto fq0_data = ov::builder::subgraph::FakeQuantizeOnDataWithConstant(256,
                                                                              std::vector<ov::Shape>{{1, 3, 1, 1}, {1, 3, 1, 1}, {1}, {1}},
                                                                              std::vector<float>{-397898.40625, -435929.78125, -476583.9375},
                                                                              std::vector<float>{394789.84375, 432524.09375, 472860.625},
                                                                              std::vector<float>{0},
                                                                              std::vector<float>{255},
                                                                              ov::element::u8);
    auto fq0 = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(indata0, ov::element::f32, fq0_data);
    auto fq1_data = ov::builder::subgraph::FakeQuantizeOnDataWithConstant(256,
                                                                              std::vector<ov::Shape>{{1, 3, 1, 1}, {1, 3, 1, 1}, {1}, {1}},
                                                                              std::vector<float>{5.390228, 9.7712707, 9.333160},
                                                                              std::vector<float>{250.524688, 254.905731, 254.467620},
                                                                              std::vector<float>{-128},
                                                                              std::vector<float>{127},
                                                                              ov::element::i8);
    auto fq1 = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(fq0, ov::element::f32, fq1_data);
    auto fq2_data = ov::builder::subgraph::FakeQuantizeOnDataWithConstant(256,
                                                                              std::vector<ov::Shape>{{1, 3, 1, 1}, {1, 3, 1, 1}, {1}, {1}},
                                                                              std::vector<float>{-98.16883, -142.82466, -155.0642700},
                                                                              std::vector<float>{97.334106, 141.599884, 153.8145446},
                                                                              std::vector<float>{0},
                                                                              std::vector<float>{255},
                                                                              ov::element::u8);
    auto fq2 = ov::builder::subgraph::makeFakeQuantizeTypeRelaxed(fq1, ov::element::f32, fq2_data);
    auto subgraph1 = std::make_shared<ov::snippets::op::Subgraph>(NodeVector{data0},
                                      std::make_shared<ov::Model>(NodeVector{fq2}, ParameterVector{indata0}));

    return std::make_shared<ov::Model>(NodeVector{subgraph1}, ParameterVector{data0});
}
}  // namespace snippets
}  // namespace test
}  // namespace ov