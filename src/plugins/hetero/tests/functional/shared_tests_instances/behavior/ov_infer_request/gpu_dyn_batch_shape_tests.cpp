// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/file_utils.hpp"
#include "common_test_utils/ov_plugin_cache.hpp"
#include "common_test_utils/subgraph_builders/split_multi_conv_concat.hpp"
#include "common_test_utils/test_common.hpp"
#include "openvino/runtime/core.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using OVDynamicBatchParams = std::tuple<std::vector<InputShape>,  // dynamic and static case sizes
                                        ov::element::Type,        // Model type
                                        std::string,              // Device name
                                        ov::AnyMap                // Config
                                        >;

class OVDynamicBatchShape_Tests : public ::testing::WithParamInterface<OVDynamicBatchParams>,
                                  virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(::testing::TestParamInfo<OVDynamicBatchParams> obj) {
        std::vector<InputShape> input_shapes;
        ov::element::Type model_type;
        std::string target_device;
        ov::AnyMap configuration;
        std::tie(input_shapes, model_type, target_device, configuration) = obj.param;

        std::ostringstream result;
        result << "IS=";
        for (const auto& shape : input_shapes) {
            result << ov::test::utils::partialShape2str({shape.first}) << "_";
        }
        result << "TS=";
        for (const auto& shape : input_shapes) {
            result << "(";
            if (!shape.second.empty()) {
                for (const auto& itr : shape.second) {
                    result << ov::test::utils::vec2str(itr);
                }
            }
            result << ")_";
        }
        result << "netPRC=" << model_type << "_";
        result << "targetDevice=" << target_device;
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
        if (core) {
            core.reset();
            core = ov::test::utils::PluginCache::get().core();
        }
        std::vector<InputShape> input_shape;

        std::tie(input_shape, model_type, targetDevice, configuration) = this->GetParam();

        init_input_shapes(input_shape);
        // TODO: think how we can switch between several input topologies in the future
        //   function = ov::test::utils::make_split_conv_concat(input_shape.front().first.get_min_shape(), model_type);
        function = ov::test::utils::make_split_multi_conv_concat(input_shape.front().first.get_min_shape(), model_type);

        //  make topology dynamic
        std::map<std::string, ov::PartialShape> dynShape;
        dynShape["input_tensor"] = input_shape.front().first;
        function->reshape(dynShape);
    }

    ov::element::Type model_type;
};

TEST_P(OVDynamicBatchShape_Tests, InferDynamicBatchBound) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
}

TEST_P(OVDynamicBatchShape_Tests, InferDynamicBatchBound_cached) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    std::string cacheFolderName;
    {
        std::stringstream ss;
        ss << "InferDynamicBatchBound_cached_" << model_type << "_" << targetDevice;
        cacheFolderName = ss.str();

        ov::test::utils::removeFilesWithExt(cacheFolderName, "blob");
        ov::test::utils::removeFilesWithExt(cacheFolderName, "cl_cache");
        ov::test::utils::removeDir(cacheFolderName);

        core->set_property(ov::cache_dir(cacheFolderName));
        run();
    }
    {
        core.reset();
        core = ov::test::utils::PluginCache::get().core();
        core->set_property(ov::cache_dir(cacheFolderName));
        run();

        ov::test::utils::removeFilesWithExt(cacheFolderName, "blob");
        ov::test::utils::removeFilesWithExt(cacheFolderName, "cl_cache");
        ov::test::utils::removeDir(cacheFolderName);
    }
}

auto hetero_config = ov::AnyMap{{ov::device::priorities.name(), ov::test::utils::DEVICE_GPU}};

const std::vector<InputShape> input_shapes = {
    {{{1, 19}, 4, 20, 20}, {{1, 4, 20, 20}, {7, 4, 20, 20}, {17, 4, 20, 20}}}};

const std::vector<ov::element::Type> model_types = {ov::element::f16, ov::element::f32};

INSTANTIATE_TEST_SUITE_P(nightly_GPU_DynBatchHetero,
                         OVDynamicBatchShape_Tests,
                         ::testing::Combine(::testing::Values(input_shapes),
                                            ::testing::ValuesIn(model_types),
                                            ::testing::Values(ov::test::utils::DEVICE_HETERO),
                                            ::testing::Values(hetero_config)),
                         OVDynamicBatchShape_Tests::getTestCaseName);
}  // namespace test
}  // namespace ov
