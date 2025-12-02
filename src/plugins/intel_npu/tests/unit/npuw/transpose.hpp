// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>
#include <immintrin.h>

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>

#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "test_utils.hpp"
#include "util.hpp"

namespace {

using TransposeTestsParams = std::tuple<ov::element::Type_t,  // Precision
                                        ShapesInitializer     // input_shape
                                        >;

class TransposeTestsBase {
protected:
    ov::element::Type type;
    ov::Tensor inTensor;
    ov::Tensor outTensor;

    std::vector<int8_t> input;
    std::vector<int8_t> output;
    std::vector<int8_t> ref_output;
    ov::Shape input_shape;

    void make_input();

public:
    void SetUp(const TransposeTestsParams& getParam);
    std::string ToString() const;
    virtual bool isNegative() const;
    virtual void make_ref_output();
};

template <class T>
class TransposeTestsTmpl : public ::testing::Test,
                           public T,
                           public ::testing::WithParamInterface<TransposeTestsParams> {
protected:
    void SetUp() override {
        T::SetUp(GetParam());
    }

public:
    static std::string getTestCaseName(const testing::TestParamInfo<TransposeTestsParams>& obj) {
        T _bt;
        _bt.SetUp(obj.param);
        return _bt.ToString();
    }
};

using TransposeTests = TransposeTestsTmpl<TransposeTestsBase>;
class TransposeTestsRef : public TransposeTests {};

}  // anonymous namespace
