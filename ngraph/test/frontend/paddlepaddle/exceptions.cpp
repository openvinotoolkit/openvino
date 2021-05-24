// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "gtest/gtest.h"
#include <frontend_manager/frontend_exceptions.hpp>

TEST(PDPD_Exceptions, pdpd_check_no_throw)
{
    EXPECT_NO_THROW(FRONT_END_GENERAL_CHECK(true));
}

TEST(PDPD_Exceptions, pdpd_check_no_throw_info)
{
    EXPECT_NO_THROW(FRONT_END_GENERAL_CHECK( true, "msg example"));
}

TEST(PDPD_Exceptions, pdpd_check_throw_no_info)
{
    EXPECT_THROW(FRONT_END_GENERAL_CHECK( false), ngraph::frontend::GeneralFailure);
}

TEST(PDPD_Exceptions, pdpd_check_throw_info)
{
    EXPECT_THROW(FRONT_END_THROW("msg example"), ngraph::frontend::GeneralFailure);
}

TEST(PDPD_Exceptions, pdpd_check_throw_check_info)
{
    std::string msg("msg example");
    try {
        FRONT_END_THROW(msg);
    } catch (const ngraph::frontend::GeneralFailure& ex) {
        std::string caught_msg(ex.what());
        EXPECT_NE(caught_msg.find(msg), std::string::npos);
    }
}
