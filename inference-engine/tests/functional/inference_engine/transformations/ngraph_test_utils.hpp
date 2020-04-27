// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/function.hpp>

#include "common_test_utils/test_common.hpp"

using TransformationTests = CommonTestUtils::TestsCommon;

std::pair<bool, std::string> compare_functions(const std::shared_ptr<ngraph::Function> & f1, const std::shared_ptr<ngraph::Function> & f2);

void check_rt_info(const std::shared_ptr<ngraph::Function> & f);