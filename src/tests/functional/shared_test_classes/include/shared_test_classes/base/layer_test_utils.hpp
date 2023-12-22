// Copyright (C) 2018-2023 Intel Corporation
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
#include <ngraph/type/bfloat16.hpp>
#include <ngraph/pass/serialize.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"

#include "functional_test_utils/crash_handler.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/summary/op_summary.hpp"
#include "functional_test_utils/summary/environment.hpp"

#include "ov_models/utils/ov_helpers.hpp"
#include "ov_models/pass/convert_prc.hpp"

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

class LayerTestsCommon : public ov::test::TestsCommon {
public:
    virtual InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &inputInfo) const;

    virtual void Run();

    virtual void Serialize(ngraph::pass::Serialize::Version ir_version = ngraph::pass::Serialize::Version::UNSPECIFIED);

    virtual void QueryNetwork();

    static void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expected,
                        const std::vector<InferenceEngine::Blob::Ptr> &actual,
                        float threshold,
                        float abs_threshold = -1.f);

    static void Compare(const std::pair<ngraph::element::Type, std::vector<std::uint8_t>> &expected,
                        const InferenceEngine::Blob::Ptr &actual,
                        float threshold,
                        float abs_threshold = -1.f);

    virtual void Compare(const std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> &expectedOutputs,
                         const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs);

    virtual void Compare(const std::pair<ngraph::element::Type, std::vector<std::uint8_t>> &expected, const InferenceEngine::Blob::Ptr &actual);

    virtual void Compare(const InferenceEngine::Blob::Ptr &expected, const InferenceEngine::Blob::Ptr &actual);

    virtual void Compare(const InferenceEngine::TensorDesc &actualDesc, const InferenceEngine::TensorDesc &expectedDesc);

    virtual void SetRefMode(RefMode mode);

    std::shared_ptr<ngraph::Function> GetFunction();

    std::map<std::string, std::string>& GetConfiguration();

    // get runtime precision by operation friendly name
    std::string getRuntimePrecision(const std::string& layerName);

    // get runtime precision by operation type
    std::string getRuntimePrecisionByType(const std::string& layerType);

    // get runtime precision by operation friendly name which can be fused
    std::string getRuntimePrecisionByFusedName(const std::string& layerName);

    std::map<std::string, ngraph::Node::RTMap> getRuntimeInfo();

#ifndef NDEBUG
    void showRuntimePrecisions();
#endif

    template<class T_IE, class T_NGRAPH>
    static void Compare(const T_NGRAPH *expected, const T_IE *actual, std::size_t size, float threshold, float abs_threshold = -1.f) {
        for (std::size_t i = 0; i < size; ++i) {
            const T_NGRAPH &ref = expected[i];
            const auto &res = actual[i];
            const auto absoluteDifference = ov::test::utils::ie_abs(res - ref);
            if (abs_threshold > 0.f && absoluteDifference > abs_threshold) {
                IE_THROW() << "Absolute comparison of values expected: " << std::to_string(ref) << " and actual: " << std::to_string(res)
                           << " at index " << i << " with absolute threshold " << abs_threshold
                           << " failed";
            }
            if (absoluteDifference <= threshold) {
                continue;
            }
            double max;
            if (sizeof(T_IE) < sizeof(T_NGRAPH)) {
                max = static_cast<double>(std::max(ov::test::utils::ie_abs(T_NGRAPH(res)), ov::test::utils::ie_abs(ref)));
            } else {
                max = static_cast<double>(std::max(ov::test::utils::ie_abs(res), ov::test::utils::ie_abs(T_IE(ref))));
            }
            double diff = static_cast<float>(absoluteDifference) / max;
            if (max == 0 || (diff > static_cast<float>(threshold)) ||
                (std::isnan(static_cast<float>(res)) ^ std::isnan(static_cast<float>(ref)))) {
                IE_THROW() << "Relative comparison of values expected: " << std::to_string(ref) << " and actual: " << std::to_string(res)
                           << " at index " << i << " with threshold " << threshold
                           << " failed";
            }
        }
    }

protected:
    LayerTestsCommon();

    RefMode GetRefMode() {
        return refMode;
    }

    std::shared_ptr<InferenceEngine::Core> getCore() {
        return core;
    }

    virtual void ConfigureNetwork();

    virtual void LoadNetwork();

    virtual void ExpectLoadNetworkToThrow(const std::string& msg);

    virtual void GenerateInputs();

    virtual void ConfigureInferRequest();

    virtual void Infer();

    TargetDevice targetDevice;
    std::shared_ptr<ngraph::Function> function;
    std::shared_ptr<ngraph::Function> functionRefs;
    std::map<std::string, std::string> configuration;
    // Non default values of layouts/precisions will be set to CNNNetwork
    InferenceEngine::Layout inLayout = InferenceEngine::Layout::ANY;
    InferenceEngine::Layout outLayout = InferenceEngine::Layout::ANY;
    InferenceEngine::Precision inPrc = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::Precision outPrc = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::ExecutableNetwork executableNetwork;
    std::vector<InferenceEngine::Blob::Ptr> inputs;
    float threshold;
    float abs_threshold;
    InferenceEngine::CNNNetwork cnnNetwork;
    std::shared_ptr<InferenceEngine::Core> core;

    virtual void Validate();

    virtual std::vector<std::pair<ngraph::element::Type, std::vector<std::uint8_t>>> CalculateRefs();

    /// default method to convert parameters for reference operation. Used before reference implementation execution
    /// can be overridden by specific operation test
    virtual void ConvertRefsParams();

    virtual std::vector<InferenceEngine::Blob::Ptr> GetOutputs();

    InferenceEngine::InferRequest inferRequest;

private:
    RefMode refMode = RefMode::INTERPRETER;
};

}  // namespace LayerTestsUtils
