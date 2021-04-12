// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <frontend_manager/frontend_manager.hpp>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ngraph;
using namespace ngraph::frontend;

class FrontEndMock: public FrontEnd {
public:
    FrontEndMock() = default;
    ~FrontEndMock() = default;

    MOCK_CONST_METHOD1(loadFromFile, InputModel::Ptr(const std::string&));
    MOCK_CONST_METHOD1(convert, std::shared_ptr<ngraph::Function>(InputModel::Ptr model));
};

TEST(FrontEndManagerTest, testAvailableFrontEnds)
{
    FrontEndManager fem;
    ASSERT_NO_THROW(fem.registerFrontEnd("mock", [](FrontEndCapabilities fec) {
        return std::make_shared<FrontEndMock>();
    }));
    auto frontends = fem.availableFrontEnds();
    ASSERT_NE(std::find(frontends.begin(), frontends.end(), "mock"), frontends.end());
}