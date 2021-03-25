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
#include <ngraph/type/bfloat16.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_common.hpp"

#include "functional_test_utils/skip_tests_config.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"

#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"

namespace LayerTestsUtils {

// filename length limitation due to Windows constraints (max 256 characters)
constexpr std::size_t maxFileNameLength = 140;

class Summary;

class SummaryDestroyer {
private:
    Summary *p_instance;
public:
    ~SummaryDestroyer();

    void initialize(Summary *p);
};

class TestEnvironment;

class LayerTestsCommon;

struct PassRate {
    enum Statuses {
        PASSED,
        FAILED,
        SKIPPED
    };
    unsigned long passed = 0;
    unsigned long failed = 0;
    unsigned long skipped = 0;

    PassRate() = default;

    PassRate(unsigned long p, unsigned long f, unsigned long s) {
        passed = p;
        failed = f;
        skipped = s;
    }

    float getPassrate() const {
        if (passed + failed == 0) {
            return 0.f;
        } else {
            return passed * 100.f / (passed + failed + skipped);
        }
    }
};

class Summary {
private:
    static Summary *p_instance;
    static SummaryDestroyer destroyer;
    std::map<ngraph::NodeTypeInfo, PassRate> opsStats = {};
    std::string deviceName;

protected:
    Summary() = default;

    Summary(const Summary &);

    Summary &operator=(Summary &);

    ~Summary() = default;

    void updateOPsStats(ngraph::NodeTypeInfo op, PassRate::Statuses status);

    std::map<ngraph::NodeTypeInfo, PassRate> getOPsStats() { return opsStats; }

    std::string getDeviceName() const { return deviceName; }

    void setDeviceName(std::string device) { deviceName = device; }

    friend class SummaryDestroyer;

    friend class TestEnvironment;

    friend class LayerTestsCommon;

public:
    static Summary &getInstance();
};

class TestEnvironment : public ::testing::Environment {
public:
    void TearDown() override;

private:
    std::string reportFileName = "report.xml";
};

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
    class PerformanceItem {
    public:
        PerformanceItem() {}

        PerformanceItem(
            std::string execOrder,
            std::string execTimeMcs,
            std::string layerType,
            std::string originalLayersNames,
            std::string outputLayouts,
            std::string outputPrecisions,
            std::string primitiveType,
            std::string runtimePrecision) :
            execOrder(execOrder),
            execTimeMcs(execTimeMcs),
            layerType(layerType),
            originalLayersNames(originalLayersNames),
            outputLayouts(outputLayouts),
            outputPrecisions(outputPrecisions),
            primitiveType(primitiveType),
            runtimePrecision(runtimePrecision) {}

        std::string execOrder;
        std::string execTimeMcs;
        std::string layerType;
        std::string originalLayersNames;
        std::string outputLayouts;
        std::string outputPrecisions;
        std::string primitiveType;
        std::string runtimePrecision;
    };

    virtual InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const;

    virtual void Run();

    virtual void Serialize();

    virtual void Compare(const std::vector<std::vector<std::uint8_t>> &expectedOutputs,
                         const std::vector<InferenceEngine::Blob::Ptr> &actualOutputs);

    virtual void Compare(const std::vector<std::uint8_t> &expected, const InferenceEngine::Blob::Ptr &actual);

    virtual void Compare(const InferenceEngine::Blob::Ptr &expected, const InferenceEngine::Blob::Ptr &actual);

    virtual void SetRefMode(RefMode mode);

    std::shared_ptr<ngraph::Function> GetFunction();

    std::map<std::string, std::string>& GetConfiguration();

    std::string getRuntimePrecision(const std::string& layerName);

    PerformanceItem getPerformanceItem(const std::string& layerName);

protected:
    LayerTestsCommon();

    template<class T>
    static void Compare(const T *expected, const T *actual, std::size_t size, T threshold) {
        for (std::size_t i = 0; i < size; ++i) {
            const auto &ref = expected[i];
            const auto &res = actual[i];
            const auto absoluteDifference = CommonTestUtils::ie_abs(res - ref);
            if (absoluteDifference <= threshold) {
                continue;
            }

            const auto max = std::max(CommonTestUtils::ie_abs(res), CommonTestUtils::ie_abs(ref));
            float diff = static_cast<float>(absoluteDifference) / static_cast<float>(max);
            ASSERT_TRUE(max != 0 && (diff <= static_cast<float>(threshold)))
                                        << "Relative comparison of values expected: " << ref << " and actual: " << res
                                        << " at index " << i << " with threshold " << threshold
                                        << " failed";
        }
    }

    RefMode GetRefMode() {
        return refMode;
    }

    std::shared_ptr<InferenceEngine::Core> getCore() {
        return core;
    }

    virtual void ConfigureNetwork();

    void LoadNetwork();

    virtual void Infer();

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
    std::shared_ptr<InferenceEngine::Core> core;

    virtual void Validate();

    virtual std::vector<std::vector<std::uint8_t>> CalculateRefs();

    std::vector<InferenceEngine::Blob::Ptr> GetOutputs();

    InferenceEngine::InferRequest inferRequest;

private:
    RefMode refMode = RefMode::INTERPRETER;
};

}  // namespace LayerTestsUtils
