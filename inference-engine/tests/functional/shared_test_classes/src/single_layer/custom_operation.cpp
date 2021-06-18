// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/custom_operation.hpp"
#include <ngraph/ngraph.hpp>
#include <ie_core.hpp>
#include <file_utils.h>

using namespace LayerTestsDefinitions;

static std::string get_extension_path() {
    return FileUtils::makePluginLibraryName<char>({}, std::string("template_extension") + IE_BUILD_POSTFIX);
}

static const InferenceEngine::IExtensionPtr& get_extension(InferenceEngine::Core* core = nullptr) {
    static InferenceEngine::IExtensionPtr extension;
    if (!extension) {
        // Core is created from the cache, so create a singleton extension
        extension = std::make_shared<InferenceEngine::Extension>(get_extension_path());
        if (core) {
            core->AddExtension(extension);
        }
    }
    return extension;
}

CustomOpLayerTest::CustomOpLayerTest(): LayerTestsUtils::LayerTestsCommon() {
    get_extension(core.get());
}

std::string CustomOpLayerTest::getTestCaseName(const testing::TestParamInfo<CustomOpLayerParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

InferenceEngine::Blob::Ptr CustomOpLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
    return FuncTestUtils::createAndFillBlobConsistently(info.getTensorDesc(), 3, 0, 1);
}

void CustomOpLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::tie(netPrecision, inputShapes, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, {inputShapes});

    std::shared_ptr<ngraph::Node> customOp;
    ASSERT_NO_THROW(customOp.reset(get_extension()->getOpSets()["custom_opset"].create("Template")));
    customOp->set_argument(0, params[0]);
    customOp->validate_and_infer_types();

    ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(customOp)};
    function = std::make_shared<ngraph::Function>(results, params, "CustomOpInference");
}
