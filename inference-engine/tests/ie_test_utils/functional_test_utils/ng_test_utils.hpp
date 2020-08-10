// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
//
#include <common_test_utils/test_common.hpp>
#include "blob_factory.hpp"
#include "functional_test_utils/skip_tests_config.hpp"

namespace FuncTestUtils {
enum RefMode {
    INTERPRETER,
    INTERPRETER_TRANSFORMATIONS,
    CONSTANT_FOLDING,
    IE
};
class IComparableNGTestCommon : public CommonTestUtils::TestsCommon {
protected:
    std::shared_ptr<ngraph::Function> function;
    std::vector<std::vector<std::uint8_t>> byteInputData;
    std::vector<std::vector<std::uint8_t>> actualByteOutput;
    std::vector<std::vector<std::uint8_t>> expectedByteOutput;
    virtual void setInput() {}
    virtual void preproc() {}
    virtual void getActualResults() {}
    virtual void getExpectedResults() {}
    virtual void postproc() {}
    virtual void validate() {}
    void Run() final {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()
        setInput();
        preproc();
        getActualResults();
        getExpectedResults();
        postproc();
        validate();
    }
};
class ComparableNGTestCommon : public IComparableNGTestCommon {
protected:
    float threshold;
    FuncTestUtils::RefMode refMode = FuncTestUtils::RefMode::INTERPRETER;

    void setRefMode(FuncTestUtils::RefMode mode) {
        refMode = mode;
    }

    FuncTestUtils::RefMode getRefMode() {
        return refMode;
    }

    template<class T>
    void compareTypedVectors(const std::vector<std::vector<T>>& v1, const std::vector<std::vector<T>>& v2);
    void compareBytes(const std::vector<std::vector<std::uint8_t>>& expectedVector, const std::vector<std::vector<std::uint8_t>>& actualVector,
                      const InferenceEngine::Precision precision);
    template<class T>
    void compareValues(const T *expected, const T *actual, std::size_t size, T thr);

    void compareValues(const void *expected, const void *actual, std::size_t size, const InferenceEngine::Precision precision);

    virtual std::vector<std::vector<std::uint8_t>> CalculateRefs(std::shared_ptr<ngraph::Function> _function, std::vector<std::vector<std::uint8_t>> _inputs);
};

}  // namespace FuncTestUtils