// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <iostream>
#include <regex>
#include <sstream>

#include "openvino/core/log_util.hpp"
#include "openvino/util/log.hpp"

namespace ov::util::test {

using LogEntries = std::tuple<const char*, int, const char*>;

class LogHelperTo : public testing::TestWithParam<LogEntries> {
    std::ostream* const out_stream = &std::cout;

protected:
    void SetUp() override {
        reset_log_handler();
        out_stream->flush();
        out_buf = out_stream->rdbuf();
        out_stream->rdbuf(str_stream.rdbuf());
    }

    void TearDown() override {
        out_stream->rdbuf(out_buf);
        reset_log_handler();
    }

    std::streambuf* out_buf;
    std::stringstream str_stream;

    auto make_regex(const char* path, int line_no, const char* message) {
        std::stringstream log_regex;
        log_regex << path << ".+" << line_no << ".+" << message << R"(\n$)";
        return std::regex{log_regex.str()};
    }
};

TEST_P(LogHelperTo, std_cout) {
    const char *path, *message;
    int line_no;
    std::tie(path, line_no, message) = GetParam();

    { LogHelper{LOG_TYPE::_LOG_TYPE_INFO, path, line_no, get_log_handler()}.stream() << message; }

    EXPECT_TRUE(std::regex_search(str_stream.str(), make_regex(path, line_no, message)));
}

INSTANTIATE_TEST_SUITE_P(Logging,
                         LogHelperTo,
                         ::testing::ValuesIn(std::vector<LogEntries>{{"the_path", 42, "tEst-mEssagE"},
                                                                     {"in the middle", 0.f, "the nowhere"}}));

}  // namespace ov::util::test
