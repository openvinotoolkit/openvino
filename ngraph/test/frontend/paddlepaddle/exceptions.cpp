// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "gtest/gtest.h"

// todo(itikhono): do not use relative paths
#include <paddlepaddle_frontend/exceptions.hpp>

TEST(PDPD_Exceptions, pdpd_check_no_throw)
{
    EXPECT_NO_THROW(PDPD_CHECK(true));
}

TEST(PDPD_Exceptions, pdpd_check_no_throw_info)
{
    EXPECT_NO_THROW(PDPD_CHECK(true, "msg example"));
}

TEST(PDPD_Exceptions, pdpd_check_throw_no_info)
{
    EXPECT_THROW(PDPD_CHECK(false), ngraph::frontend::pdpd::CheckFailurePDPD);
}

TEST(PDPD_Exceptions, pdpd_check_throw_info)
{
    EXPECT_THROW(PDPD_CHECK(false, "msg example"), ngraph::frontend::pdpd::CheckFailurePDPD);
}

TEST(PDPD_Exceptions, pdpd_check_throw_check_info)
{
    std::string msg("msg example");
    try {
        PDPD_CHECK(false, msg) ;
    } catch (const ngraph::frontend::pdpd::CheckFailurePDPD& ex) {
        std::string caught_msg(ex.what());
        EXPECT_NE(caught_msg.find(msg), std::string::npos);
    }
}
