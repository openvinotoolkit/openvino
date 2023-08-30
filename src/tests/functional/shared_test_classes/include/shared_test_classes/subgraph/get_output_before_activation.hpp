// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/layer_test_utils.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace SubgraphTestsDefinitions {
enum class midOutputType {
    Sum,
    Sub,
    Mul,
};

typedef std::tuple<
    std::string,                        // Target device name
    InferenceEngine::Precision,         // Network precision
    size_t,                             // Input size
    midOutputType,                      // Type of layer that will be an output
    std::map<std::string, std::string>  // Configuration
> OutputBeforeActivationLegacyParams;

std::ostream& operator<< (std::ostream& os, const midOutputType& oType);

class OutputBeforeActivationLegacy : virtual public LayerTestsUtils::LayerTestsCommon,
    public testing::WithParamInterface<OutputBeforeActivationLegacyParams> {
protected:
    void SetUp() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<OutputBeforeActivationLegacyParams> &obj);
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo &info) const override;
};
} // namespace SubgraphTestsDefinitions

namespace ov {
namespace test {
enum class midOutputType {
    Sum,
    Sub,
    Mul,
};

typedef std::tuple<
    std::string,                        // Target device name
    ov::element::Type,                  // Network precision
    size_t,                             // Input size
    midOutputType,                      // Type of layer that will be an output
    std::map<std::string, std::string>  // Configuration
> OutputBeforeActivationLegacyParams;

std::ostream& operator<< (std::ostream& os, const midOutputType& oType);

class OutputBeforeActivation : public testing::WithParamInterface<OutputBeforeActivationLegacyParams>,
                               virtual public ov::test::SubgraphBaseTest{
protected:
    void SetUp() override;
public:
    static std::string getTestCaseName(const testing::TestParamInfo<OutputBeforeActivationLegacyParams> &obj);
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
};
} //  namespace test
} //  namespace ov
