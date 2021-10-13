// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <typeindex>
#include <string>
#include <vector>
#include <memory>
#include <tuple>

#include <gtest/gtest.h>

#include <openvino/core/node.hpp>
#include <openvino/core/function.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/core/type/bfloat16.hpp>

#include <ie_plugin_config.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"
#include "common_test_utils/test_constants.hpp"

#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/ov_plugin_cache.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/layer_test_utils/summary.hpp"
#include "functional_test_utils/layer_test_utils/environment.hpp"

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"

namespace ov {
namespace test {

class SubgraphBaseTest : public CommonTestUtils::TestsCommon {
public:
    virtual void run();
    virtual void serialize();
    virtual void query_model();

    void TearDown() override {
        if (!configuration.empty()) {
            ov::test::PluginCache::get().core().reset();
        }
    }

protected:
    void compare(const std::vector<ov::runtime::Tensor> &expected,
                 const std::vector<ov::runtime::Tensor> &actual);

//    virtual void compare_desc(const ov::runtime::Tensor &expected, const ov::runtime::Tensor &actual);

//    std::shared_ptr<ov::Function> get_function();

//    std::map<std::string, std::string> &get_configuration();

//    std::string get_runtime_precision(const std::string &layerName);

//    std::string get_runtime_precision_by_type(const std::string &layerType);

//#ifndef NDEBUG
//
//    void show_runtime_precision();
//
//#endif

//    virtual void configure_model();

    virtual void compile_model();
    virtual void generate_inputs(const std::vector<ngraph::Shape>& targetInputStaticShapes);
    virtual void infer();
    virtual void validate();

    void init_input_shapes(const std::pair<std::vector<ov::PartialShape>, std::vector<std::vector<ov::Shape>>>& shapes);
    void init_input_shapes(const std::pair<ov::PartialShape, std::vector<ov::Shape>>& shapes);

    //    virtual std::vector<ov::runtime::Tensor> get_outputs();

    std::shared_ptr<ov::runtime::Core> core = ov::test::PluginCache::get().core();
    std::string targetDevice;
    std::map<std::string, std::string> configuration;

    std::shared_ptr<ngraph::Function> function;
    std::shared_ptr<ngraph::Function> functionRefs;
    ov::element::Type inPrc;
    ov::element::Type outPrc;
    std::map<std::string, ov::runtime::Tensor> inputs;
    std::vector<ngraph::PartialShape> inputDynamicShapes;
    std::vector<std::vector<ngraph::Shape>> targetStaticShapes;

    ov::runtime::ExecutableNetwork executableNetwork;
    ov::runtime::InferRequest inferRequest;

    constexpr static const auto disable_treshold = std::numeric_limits<double>::max();
    double abs_threshold = disable_treshold;
    double rel_threshold = disable_treshold;

    LayerTestsUtils::Summary& summary = LayerTestsUtils::Summary::getInstance();;

private:
    void resize_ngraph_function(const std::vector<ngraph::Shape>& targetInputStaticShapes);
    std::vector<ov::runtime::Tensor> calculate_refs();
    std::vector<ov::runtime::Tensor> get_outputs();
};
}  // namespace test
}  // namespace ov