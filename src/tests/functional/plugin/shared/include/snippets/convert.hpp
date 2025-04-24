// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/base/snippets_test_utils.hpp"

namespace ov {
namespace test {
namespace snippets {

typedef std::tuple<
        std::vector<InputShape>,                                                    // InputShapes
        std::pair<std::vector<ov::element::Type>, std::vector<ov::element::Type>>,  // Input and Output data types for Converts
        size_t,                                                                     // Expected num nodes
        size_t,                                                                     // Expected num subgraphs
        std::string                                                                 // Target Device
> ConvertParams;

using parameters = std::vector<std::tuple<int32_t, int32_t, int32_t>>;

class Convert : public testing::WithParamInterface<ov::test::snippets::ConvertParams>,
                virtual public SnippetsTestsCommon {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ov::test::snippets::ConvertParams> obj);

protected:
    void SetUp() override;

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    virtual parameters generate_params_random() const;

    ov::element::Type output_type = ov::element::f32;
};

class ConvertInput : public Convert {
protected:
    void SetUp() override;

    parameters generate_params_random() const override;
};

class ConvertOutput : public ConvertInput {
protected:
    void SetUp() override;
};

class ConvertStub : public ConvertInput {
protected:
    void SetUp() override;
};

class ConvertPartialInputsAndResults : public ConvertInput {
protected:
    void SetUp() override;
};

class ConvertManyOnInputs : public ConvertInput {
protected:
    void SetUp() override;
};

class ConvertManyOnOutputs : public ConvertInput {
protected:
    void SetUp() override;
};

class ConvertManyOnInputOutput : public ConvertInput {
protected:
    void SetUp() override;
};

} // namespace snippets
} // namespace test
} // namespace ov