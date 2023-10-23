// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/core.hpp"
#include <common_test_utils/test_common.hpp>
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/file_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "ov_models/subgraph_builders.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

using namespace ::testing;
using namespace ov::test;

using OVDynamicBatchParams = std::tuple<
    std::vector<InputShape>,                                           // dynamic and static case sizes
    ElementType,                                                       // Network precision
    std::string,                                                       // Device name
    ov::AnyMap                                                         // Config
>;

class OVDynamicBatchShape_Tests : public WithParamInterface<OVDynamicBatchParams>,
    virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(TestParamInfo<OVDynamicBatchParams> obj) {
        std::vector<InputShape> inputShapes;
        ElementType netPrecision;
        std::string targetDevice;
        ov::AnyMap configuration;
        std::tie(inputShapes, netPrecision, targetDevice, configuration) = obj.param;

        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : inputShapes) {
            result << ov::test::utils::partialShape2str({ shape.first }) << "_";
        }
        result << "TS=";
        for (const auto& shape : inputShapes) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << ov::test::utils::vec2str(itr);
                }
            }
            result << ")_";
        }
        result << "netPRC=" << netPrecision << "_";
        result << "targetDevice=" << targetDevice;
        if (!configuration.empty()) {
            for (auto& configItem : configuration) {
                result << "configItem=" << configItem.first << "_";
                configItem.second.print(result);
                result << "_";
            }
        }
        return result.str();
    }

protected:
    void SetUp() override {
        if (core)
            core.reset();
        std::tie(inputShape, netPrecision, targetDevice, configuration) = this->GetParam();

        init_input_shapes(inputShape);
        //TODO: think how we can switch between several input topologies in the future
        //  function = ngraph::builder::subgraph::makeSplitConvConcat(inputShape.front().first.get_min_shape(), netPrecision);
        function = ngraph::builder::subgraph::makeSplitMultiConvConcat(inputShape.front().first.get_min_shape(), netPrecision);

        //  make topology dynamic
        std::map<std::string, ov::PartialShape> dynShape;
        dynShape["input_tensor"] = inputShape.front().first;
        function->reshape(dynShape);
    }
    std::shared_ptr<ov::Model> src_func;
    // std::map<std::string, std::string> configuration;
    std::vector<InputShape> inputShape;
    ElementType netPrecision;
};

TEST_P(OVDynamicBatchShape_Tests, InferDynamicBatchBound) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    core = std::make_shared<ov::Core>();
    run();
}

TEST_P(OVDynamicBatchShape_Tests, InferDynamicBatchBound_cached) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::string cacheFolderName;
    {
        std::stringstream ss;
        ss << "InferDynamicBatchBound_cached_" << netPrecision << "_" << targetDevice;
        cacheFolderName = ss.str();

        ov::test::utils::removeFilesWithExt(cacheFolderName, "blob");
        ov::test::utils::removeFilesWithExt(cacheFolderName, "cl_cache");
        ov::test::utils::removeDir(cacheFolderName);

        core = std::make_shared<ov::Core>();
        core->set_property(ov::cache_dir(cacheFolderName));
        run();
    }
    {
        core = std::make_shared<ov::Core>();
        core->set_property(ov::cache_dir(cacheFolderName));
        run();

        ov::test::utils::removeFilesWithExt(cacheFolderName, "blob");
        ov::test::utils::removeFilesWithExt(cacheFolderName, "cl_cache");
        ov::test::utils::removeDir(cacheFolderName);
    }
}

namespace {
auto config = []() {
    return ov::AnyMap{};
};

auto hetero_config = []() {
    return ov::AnyMap{{"TARGET_FALLBACK", ov::test::utils::DEVICE_GPU}};
};

const std::vector<InputShape> inputShapes = {
    { { {1, 19}, 4, 20, 20}, { {1, 4, 20, 20}, {7, 4, 20, 20}, {17, 4, 20, 20} } }
};

const std::vector<ElementType> netPrecisions = {
    ElementType::f16,
    ElementType::f32
};

INSTANTIATE_TEST_SUITE_P(smoke_GPU_DynBatch, OVDynamicBatchShape_Tests,
    ::testing::Combine(
        ::testing::Values(inputShapes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU),
        ::testing::Values(config())),
    OVDynamicBatchShape_Tests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GPU_DynBatchHetero, OVDynamicBatchShape_Tests,
    ::testing::Combine(
        ::testing::Values(inputShapes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_HETERO),
        ::testing::Values(hetero_config())),
    OVDynamicBatchShape_Tests::getTestCaseName);
}  // namespace
