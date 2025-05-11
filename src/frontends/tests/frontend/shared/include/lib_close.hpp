// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>

#include <tuple>

/**
 * \brief Front end library close test parameters.
 *
 * \param frontend    Frontend name.
 * \param model_path  Model file path.
 * \param exp_name    Tensor name in model used to get place.
 */
using FrontendLibCloseParams = std::tuple<std::string,  // frontend name
                                          std::string,  // model file path
                                          std::string   // tensor name to get place
                                          >;

/** \brief Frontend library close test fixture with parameters \ref FrontendLibCloseParams. */
class FrontendLibCloseTest : public testing::TestWithParam<FrontendLibCloseParams> {
public:
    static std::string get_test_case_name(const testing::TestParamInfo<FrontendLibCloseParams>& obj);

protected:
    std::string frontend;
    std::string model_path;
    std::string exp_name;

    void SetUp() override;
};
