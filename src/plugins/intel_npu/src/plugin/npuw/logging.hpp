// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>

#include "openvino/openvino.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {

namespace npuw {

enum class LogLevel { None = 0, Error = 1, Warning = 2, Info = 3, Verbose = 4, Debug = 5 };

LogLevel get_log_level();

class __logging_indent__ {
    static thread_local int this_indent;

public:
    __logging_indent__();
    ~__logging_indent__();
    static int __level__();
};

void dump_tensor(const ov::SoPtr<ov::ITensor>& tensor, const std::string& base_path);

void dump_input_list(const std::string base_name, const std::vector<std::string>& base_input_names);

void dump_output_list(const std::string base_name, const std::vector<std::string>& base_output_names);

void dump_failure(const std::shared_ptr<ov::Model>& model, const std::string& device, const char* extra);
}  // namespace npuw
}  // namespace ov

#define LOG_IMPL(str, level, levelstr)                                        \
    do {                                                                      \
        if (ov::npuw::get_log_level() >= ov::npuw::LogLevel::level) {         \
            std::cout << "[ NPUW:" levelstr " ] ";                            \
            const int this_level = ov::npuw::__logging_indent__::__level__(); \
            for (int i = 0; i < this_level; i++)                              \
                std::cout << "    ";                                          \
            std::cout << str << std::endl;                                    \
        }                                                                     \
    } while (0)

#define LOG_INFO(str)  LOG_IMPL(str, Info, "INFO")
#define LOG_WARN(str)  LOG_IMPL(str, Warning, "WARN")
#define LOG_ERROR(str) LOG_IMPL(str, Error, " ERR")
#define LOG_DEBUG(str) LOG_IMPL(str, Debug, " DBG")
#define LOG_VERB(str)  LOG_IMPL(str, Verbose, "VERB")

#define LOG_BLOCK() ov::npuw::__logging_indent__ object_you_should_never_use__##__LINE__

#define NPUW_ASSERT(expr)                                       \
    do {                                                        \
        if (!(expr)) {                                          \
            OPENVINO_THROW("NPUW: Assertion " #expr " failed"); \
        }                                                       \
    } while (0)

#ifdef _MSC_VER
#    define __PRETTY_FUNCTION__ __FUNCSIG__
#endif
