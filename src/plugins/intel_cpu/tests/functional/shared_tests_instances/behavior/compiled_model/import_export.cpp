// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/test_constants.hpp"
#include "behavior/compiled_model/import_export.hpp"

namespace {

using namespace ov::test::behavior;

const std::vector<ov::element::Type_t> netPrecisions = {
    ov::element::i8,
    ov::element::i16,
    ov::element::i32,
    ov::element::i64,
    ov::element::u8,
    ov::element::u16,
    ov::element::u32,
    ov::element::u64,
    ov::element::f16,
    ov::element::f32,
};
const ov::AnyMap empty_property = {};

INSTANTIATE_TEST_SUITE_P(smoke_serialization,
                         OVCompiledGraphImportExportTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU),
                                            ::testing::Values(empty_property)),
                         OVCompiledGraphImportExportTest::getTestCaseName);

TEST_P(OVCompiledModelGraphUniqueNodeNamesTest, CheckUniqueNodeNames) {
    std::shared_ptr<ov::Core> core = ov::test::utils::PluginCache::get().core();
    auto compiled_model = core->compile_model(model, target_device);
    auto exec_graph = compiled_model.get_runtime_model();

    int numReorders = 0;
    int expectedReorders = 2;
    std::unordered_set<std::string> names;
    ASSERT_NE(exec_graph, nullptr);

    for (const auto& op : exec_graph->get_ops()) {
        const auto& rtInfo = op->get_rt_info();
        auto it = rtInfo.find(ov::exec_model_info::LAYER_TYPE);
        ASSERT_NE(rtInfo.end(), it);
        auto opType = it->second.as<std::string>();

        if (opType == "Reorder") {
            numReorders++;
        }
    }

    ASSERT_EQ(numReorders, expectedReorders)
        << "Expected reorders: " << expectedReorders << ", actual reorders: " << numReorders;
};

const std::vector<ov::element::Type> netPrc = {
    ov::element::f32,
};

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape,
                         OVCompiledModelGraphUniqueNodeNamesTest,
                         ::testing::Combine(::testing::ValuesIn(netPrc),
                                            ::testing::Values(ov::Shape{1, 2, 5, 5}),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         OVCompiledModelGraphUniqueNodeNamesTest::getTestCaseName);

}  // namespace

