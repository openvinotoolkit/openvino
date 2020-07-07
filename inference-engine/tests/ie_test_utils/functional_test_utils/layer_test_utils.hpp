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

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"

#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"


namespace LayerTestsUtils {

using TargetDevice = std::string;

typedef std::tuple<
        InferenceEngine::Precision,  // Network Precision
        InferenceEngine::SizeVector, // Input Shape
        TargetDevice                 // Target Device
> basicParams;

enum RefMode {
    INTERPRETER,
    CONSTANT_FOLDING,
    IE
};

class LayerTestsCommon : public CommonTestUtils::TestsCommon {
public:
    virtual InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const;

    virtual void Run();

    virtual void Compare(const std::vector<std::vector<std::uint8_t>> &expectedOutputs,
                         const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs);

    virtual void Compare(const std::vector<std::uint8_t> &expected, const InferenceEngine::Blob::Ptr &actual);

    virtual void SetRefMode(RefMode mode);

protected:
    LayerTestsCommon();

    ~LayerTestsCommon() override;

    template<class T>
    void Compare(const T *expected, const T *actual, std::size_t size, T threshold) {
        std::cout << std::endl;
        for (std::size_t i = 0; i < size; ++i) {
            const auto &ref = expected[i];
            const auto &res = actual[i];
            const auto absoluteDifference = std::abs(res - ref);
            if (absoluteDifference <= threshold) {
                continue;
            }

            const auto max = std::max(std::abs(res), std::abs(ref));
            ASSERT_TRUE(max != 0 && ((absoluteDifference / max) <= threshold))
                                        << "Relative comparison of values expected: " << ref << " and actual: " << res
                                        << " at index " << i << " with threshold " << threshold
                                        << " failed";
        }
    }

    RefMode GetRefMode() {
        return refMode;
    }

    void ConfigurePlugin() const;

    void LoadNetwork();

    void Infer();

    TargetDevice targetDevice;
    std::shared_ptr<ngraph::Function> function;
    std::map<std::string, std::string> configuration;
    // Non default values of layouts/precisions will be set to CNNNetwork
    InferenceEngine::Layout inLayout = InferenceEngine::Layout::ANY;
    InferenceEngine::Layout outLayout = InferenceEngine::Layout::ANY;
    InferenceEngine::Precision inPrc = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::Precision outPrc = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::ExecutableNetwork executableNetwork;
    std::vector<InferenceEngine::Blob::Ptr> inputs;
    float threshold;
    InferenceEngine::CNNNetwork cnnNetwork;

    virtual void Validate();

    virtual std::vector<std::vector<std::uint8_t>> CalculateRefs();

    InferenceEngine::InferRequest inferRequest;

private:
    void ConfigureNetwork() const;

    std::vector<InferenceEngine::Blob::Ptr> GetOutputs();
    std::shared_ptr<InferenceEngine::Core> core;
    RefMode refMode = RefMode::INTERPRETER;
};

}  // namespace LayerTestsUtils
