// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/function.hpp"

#include "common_test_utils/test_common.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils/summary.hpp"

namespace ov {
namespace test {

using InputShape = std::pair<ov::PartialShape, std::vector<ov::Shape>>;
using InputShapes = std::pair<std::vector<ov::PartialShape>, std::vector<std::vector<ov::Shape>>>;
using ElementType = ov::element::Type_t;
using Config = std::map<std::string, std::string>;
using TargetDevice = std::string;

class SubgraphBaseTest : public CommonTestUtils::TestsCommon {
public:
    virtual void run();
    virtual void serialize();
    virtual void query_model();

    void TearDown() override {
        if (!configuration.empty()) {
            ov::test::utils::PluginCache::get().core().reset();
        }
    }

protected:
    void compare(const std::vector<ov::runtime::Tensor> &expected,
                 const std::vector<ov::runtime::Tensor> &actual);

    virtual void configure_model();
    virtual void compile_model();
    virtual void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes);
    virtual void infer();
    virtual void validate();

    void init_input_shapes(const InputShapes& shapes);
    void init_input_shapes(const InputShape& shapes);

    std::shared_ptr<ov::runtime::Core> core = ov::test::utils::PluginCache::get().core();
    std::string targetDevice;
    Config configuration;

    std::shared_ptr<ov::Function> function, functionRefs = nullptr;
    std::map<std::string, ov::runtime::Tensor> inputs;
    std::vector<ngraph::PartialShape> inputDynamicShapes;
    std::vector<std::vector<ngraph::Shape>> targetStaticShapes;

    ov::runtime::ExecutableNetwork executableNetwork;
    ov::runtime::InferRequest inferRequest;

    constexpr static const double disable_threshold = std::numeric_limits<double>::max();
    double abs_threshold = disable_threshold, rel_threshold = disable_threshold;

    // TODO: iefode: change namespace names a bit later
    LayerTestsUtils::Summary& summary = LayerTestsUtils::Summary::getInstance();;

private:
    std::vector<ov::runtime::Tensor> calculate_refs();
    std::vector<ov::runtime::Tensor> get_plugin_outputs();
};

inline std::vector<InputShape> static_shapes_to_test_representation(const std::vector<ov::Shape>& staticShapes) {
    std::vector<InputShape> result;
    for (const auto& staticShape : staticShapes) {
        result.push_back({{}, {staticShape}});
    }
    return result;
}

inline std::vector<InputShapes> static_shapes_to_test_representation(const std::vector<std::vector<ov::Shape>>& staticShapes) {
    std::vector<InputShapes> result;
    for (const auto& staticShape : staticShapes) {
        result.push_back({{}, {staticShape}});
    }
    return result;
}
}  // namespace test
}  // namespace ov