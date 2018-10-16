// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef USE_BOOST_RE
#include <regex>
#define REPLACE_WITH_STR(SRC, PATTERN, STR) SRC = std::regex_replace(SRC, std::regex(PATTERN), STR)
#define FIND_STR(SRC, PATTERN) std::regex_search(SRC, std::regex(PATTERN))
#else
#include <boost/regex.hpp>
#define REPLACE_WITH_STR(SRC, PATTERN, STR) SRC = boost::regex_replace(SRC, boost::regex(PATTERN), STR)
#define FIND_STR(SRC, PATTERN) boost::regex_search(SRC, boost::regex(PATTERN))
#endif

#define REPLACE_WITH_NUM(SRC, PATTERN, NUM) REPLACE_WITH_STR(SRC, PATTERN, std::to_string(NUM))
#define REMOVE_LINE(SRC, PATTERN) REPLACE_WITH_STR(SRC, PATTERN, "")
