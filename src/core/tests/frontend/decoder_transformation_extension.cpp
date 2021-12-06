// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common/extensions/decoder_transformation_extension.hpp>
#include <openvino/pass/manager.hpp>
#include "gtest/gtest.h"

using namespace ov::frontend;

TEST(DecoderTransformation, MatcherPass) {
    bool flag = false;
    DecoderTransformationExtension decoder_ext([&](ov::pass::MatcherPass* matcher){
       flag = true;
    });

    ov::pass::Manager manager;
    decoder_ext.register_pass(manager);
    manager.run_passes(std::make_shared<ov::Function>(ov::ResultVector{}, ov::ParameterVector{}));
    EXPECT_EQ(flag, true);
}

TEST(DecoderTransformation, FunctionPass) {
    bool flag = false;
    DecoderTransformationExtension decoder_ext([&](const std::shared_ptr<ov::Function>&){
        flag = true;
        return flag;
    });

    ov::pass::Manager manager;
    decoder_ext.register_pass(manager);
    manager.run_passes(std::make_shared<ov::Function>(ov::ResultVector{}, ov::ParameterVector{}));
    EXPECT_EQ(flag, true);
}

TEST(DecoderTransformation, TestPass) {
    class TestPass :  public ov::pass::FunctionPass {
    public:
        OPENVINO_RTTI("ov::pass::TestPass");
        TestPass() = default;
        TestPass(const TestPass& tp) = default;
        bool run_on_function(std::shared_ptr<ov::Function>) override {
            *m_flag = true;
            return *m_flag;
        }
        std::shared_ptr<bool> m_flag = std::make_shared<bool>(false);
    } test_pass;
    DecoderTransformationExtension decoder_ext(test_pass);

    ov::pass::Manager manager;
    decoder_ext.register_pass(manager);
    manager.run_passes(std::make_shared<ov::Function>(ov::ResultVector{}, ov::ParameterVector{}));
    EXPECT_EQ(*test_pass.m_flag, true);
}
