// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <typeindex>
#include <string>
#include <vector>
#include <memory>
#include <tuple>
#include <gtest/gtest.h>
#include <ngraph/node.hpp>
#include <ngraph/function.hpp>
#include <ie_plugin_config.hpp>
#include <ngraph/function.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"

#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"
#include "ng_test_utils.hpp"

namespace LayerTestsUtils {

using TargetDevice = std::string;

typedef std::tuple<
        InferenceEngine::Precision,  // Network Precision
        InferenceEngine::SizeVector, // Input Shape
        TargetDevice                 // Target Device
> basicParams;

class LayerTestsCommon :  public FuncTestUtils::ComparableNGTestCommon {
public:
    virtual InferenceEngine::Blob::Ptr generateInput(const InferenceEngine::InputInfo &info) const;

    virtual void compare(const std::vector<std::vector<std::uint8_t>> &expectedOutputs,
                         const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs);

protected:
    LayerTestsCommon();

    ~LayerTestsCommon() override;

    std::shared_ptr<InferenceEngine::Core> getCore() {
        return core;
    }

    void configurePlugin();

    void configureNetwork() const;

    void loadNetwork();

    void getActualResults() override;

    virtual void infer();

    void setInput() override;

    void validate() override;

    TargetDevice targetDevice;
    std::map<std::string, std::string> configuration;
    // Non default values of layouts/precisions will be set to CNNNetwork
    InferenceEngine::Layout inLayout = InferenceEngine::Layout::ANY;
    InferenceEngine::Layout outLayout = InferenceEngine::Layout::ANY;
    InferenceEngine::Precision inPrc = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::Precision outPrc = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::ExecutableNetwork executableNetwork;
    std::vector<InferenceEngine::Blob::Ptr> inputs;
    InferenceEngine::CNNNetwork cnnNetwork;
    InferenceEngine::InferRequest inferRequest;
    std::vector<InferenceEngine::Blob::Ptr> getOutputs();
    std::shared_ptr<InferenceEngine::Core> core;
};

}  // namespace LayerTestsUtils
