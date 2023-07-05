// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pad_test_utils.hpp"

using PadFactoryPtr = std::shared_ptr<IPadFactory>;
using TestModelFactoryPtr = std::shared_ptr<ITestModelFactory>;
using TestParams = std::tuple<PadFactoryPtr, TestModelFactoryPtr>;

TEST_P(PadTestFixture, CompareFunctions) {
    PadFactoryPtr pad_factory;
    TestModelFactoryPtr model_factory;
    std::tie(pad_factory, model_factory) = this->GetParam();

    model_factory->setup(pad_factory, manager);
    model = model_factory->function;
    model_ref = model_factory->function_ref;
    if (!model_ref)
        model_ref = model->clone();
}

std::string PadTestFixture::get_test_name(const ::testing::TestParamInfo<TestParams>& obj) {
    PadFactoryPtr pad_factory;
    TestModelFactoryPtr model_factory;
    std::tie(pad_factory, model_factory) = obj.param;

    std::ostringstream test_name;
    test_name << "pad_factory=" << pad_factory->getTypeName() << "/";
    test_name << "model_factory=" << model_factory->test_name;

    return test_name.str();
}
