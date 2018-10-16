// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <list>
#include <inference_engine/shape_infer/built-in/ie_built_in_holder.hpp>
#include <shape_infer/mock_ishape_infer_impl.hpp>
#include <inference_engine/shape_infer/built-in/ie_equal_shape_infer.hpp>

using namespace InferenceEngine;
using namespace ShapeInfer;

class ShapeInferHolderTest : public ::testing::Test {
protected:
    StatusCode sts = GENERAL_ERROR;
    ResponseDesc resp;
    std::vector<InferenceEngine::SizeVector> outShapes;
    std::map<std::string, std::string> params;
    std::map<std::string, Blob::Ptr> blobs;

    std::list<std::string> _expectedTypes = {
            "Power",
            "Convolution",
            "Deconvolution",
            "Pooling",
            "LRN",
            "Norm",
            "SoftMax",
            "ReLU",
            "Clamp",
            "Split",
            "Slice",
            "Concat",
            "Eltwise",
            "ScaleShift",
            "PReLU",
            "Crop",
            "Reshape",
            "Tile",
            "BatchNormalization",
            "Input",
            "Memory",
            "Const"
    };

    void TearDown() override {
    }

    void SetUp() override {
    }

public:

};

TEST_F(ShapeInferHolderTest, canCreateHolder) {
    ASSERT_NO_THROW(BuiltInShapeInferHolder());
}

TEST_F(ShapeInferHolderTest, DISABLED_allRegistered) {
    auto holder = std::make_shared<BuiltInShapeInferHolder>();
    char** types = nullptr;
    unsigned int size = 0;
    ASSERT_NO_THROW(sts = holder->getPrimitiveTypes(types, size, &resp));
    std::list<std::string> actualTypes;
    for (int i = 0; i < size; i++) {
        actualTypes.emplace_back(types[i], strlen(types[i]));
    }

    _expectedTypes.sort();
    actualTypes.sort();

    std::vector<std::string> different_words;
    std::set_difference(actualTypes.begin(), actualTypes.end(),
                        _expectedTypes.begin(), _expectedTypes.end(),
                        std::back_inserter(different_words));
    // TODO: update expectedTypes!
    ASSERT_EQ(19, different_words.size());
}


TEST_F(ShapeInferHolderTest, returnNullForNotKnown) {
    IShapeInferImpl::Ptr impl;

    sts = BuiltInShapeInferHolder().getShapeInferImpl(impl, "NOT_KNOWN_TYPE", &resp);
    ASSERT_FALSE(impl) << resp.msg;
    ASSERT_EQ(NOT_FOUND, sts);
}

class ShapeInferNotSupportedTest
        : public ShapeInferHolderTest, public testing::WithParamInterface<std::string> {
};

TEST_P(ShapeInferNotSupportedTest, returnNotFoundOnNotSupported) {
    std::string type = GetParam();
    IShapeInferImpl::Ptr impl;

    sts = BuiltInShapeInferHolder().getShapeInferImpl(impl, type.c_str(), &resp);
    ASSERT_FALSE(impl) << resp.msg;
    ASSERT_EQ(NOT_FOUND, sts) << resp.msg;
}

// TODO: list all not supported later
INSTANTIATE_TEST_CASE_P(
        NotSupported, ShapeInferNotSupportedTest, ::testing::Values("NOT_SUPPORTED"));
