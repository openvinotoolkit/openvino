// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "common_test_utils/file_utils.hpp"

#if defined(__GNUC__) && (__GNUC__ <= 4) && (__GNUC_MINOR__ < 9) && !defined(__clang__)
# define IE_GCC_4_8
#endif

#ifndef IE_GCC_4_8
# include <regex>
# define REPLACE_WITH_STR(SRC, PATTERN, STR) SRC = std::regex_replace(SRC, std::regex(PATTERN), STR)
# define FIND_STR(SRC, PATTERN) std::regex_search(SRC, std::regex(PATTERN))
#elif defined USE_BOOST_RE
# include <boost/regex.hpp>
# define REPLACE_WITH_STR(SRC, PATTERN, STR) SRC = boost::regex_replace(SRC, boost::regex(PATTERN), STR)
# define FIND_STR(SRC, PATTERN) boost::regex_search(SRC, boost::regex(PATTERN))
#else
# error "Cannot implement regex"
# define REPLACE_WITH_STR(SRC, PATTERN, STR)
# define FIND_STR(SRC, PATTERN)
#endif

#define REPLACE_WITH_NUM(SRC, PATTERN, NUM) REPLACE_WITH_STR(SRC, PATTERN, CommonTestUtils::to_string_c_locale(NUM))
