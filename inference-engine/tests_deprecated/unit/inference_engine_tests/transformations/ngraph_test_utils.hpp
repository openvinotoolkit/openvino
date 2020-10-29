// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <ngraph/function.hpp>

std::pair<bool, std::string> compare_functions(const std::shared_ptr<ngraph::Function> & f1, const std::shared_ptr<ngraph::Function> & f2);