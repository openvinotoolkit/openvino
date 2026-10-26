//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/logger.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <streambuf>
#include <string>
#include <string_view>

namespace {

class ScopedEnvVar final {
public:
    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;
    ScopedEnvVar(ScopedEnvVar&&) = delete;
    ScopedEnvVar& operator=(ScopedEnvVar&&) = delete;

    ScopedEnvVar(const char* name, const char* value): _name(name) {
        const auto* current = std::getenv(_name.c_str());
        if (current != nullptr) {
            _hadValue = true;
            _oldValue = current;
        }

        if (value == nullptr) {
            unset();
        } else {
            set(value);
        }
    }

    ~ScopedEnvVar() {
        if (_hadValue) {
            set(_oldValue.c_str());
            return;
        }
        unset();
    }

private:
    void set(const char* value) {
#if defined(_WIN32)
        _putenv_s(_name.c_str(), value);
#else
        setenv(_name.c_str(), value, 1);
#endif
    }

    void unset() {
#if defined(_WIN32)
        _putenv_s(_name.c_str(), "");
#else
        unsetenv(_name.c_str());
#endif
    }

private:
    std::string _name;
    bool _hadValue = false;
    std::string _oldValue;
};

class VirtualMachineLoggerTest : public testing::Test {
public:
    static std::string captureLogOutput(const std::function<void()>& emitLog) {
        std::stringstream buffer;
        std::streambuf* old = std::cerr.rdbuf(buffer.rdbuf());
        emitLog();
        std::cerr.flush();
        std::cerr.rdbuf(old);
        return buffer.str();
    }

protected:
    void SetUp() override {
        _envLock = std::unique_lock<std::mutex>(envMutex());
#if defined(VPUX_DEVELOPER_BUILD)
        // Reset logging cache between test cases to ensure test isolation when env vars are changed
        ::intel_npu::vm::detail::resetLoggingCache();
#endif
    }

    static std::mutex& envMutex() {
        static std::mutex mutex;
        return mutex;
    }

private:
    std::unique_lock<std::mutex> _envLock;
};

void expectLogContains(std::string_view output, std::string_view expectedText) {
    EXPECT_THAT(output, testing::HasSubstr(std::string(expectedText)));
}

void expectLogNotContains(std::string_view output, std::string_view unexpectedText) {
    EXPECT_THAT(output, testing::Not(testing::HasSubstr(std::string(unexpectedText))));
}

TEST_F(VirtualMachineLoggerTest, FormatsDifferentArgumentTypes) {
#if defined(VPUX_DEVELOPER_BUILD)
    const ScopedEnvVar logLevel{"NPU_VM_LOG_LEVEL", "LOG_INFO"};
    const ScopedEnvVar logColors{"NPU_VM_LOG_COLORS", "0"};
#endif

    const std::string output = captureLogOutput([] {
        NPU_VM_LOG_INFO("format test {} {} {}", 42, std::string{"abc"}, std::string_view{"xyz"});
    });

#if BUILD_LOG_LEVEL >= 4  // 4 => Info, Warn, Error, Fatal are enabled
    expectLogContains(output, "[INFO]");
    expectLogContains(output, "logger.cpp:");
    expectLogContains(output, "format test 42 abc xyz");
#else
    EXPECT_TRUE(output.empty());
#endif
}

TEST_F(VirtualMachineLoggerTest, HandlesMissingAndExtraPlaceholders) {
#if defined(VPUX_DEVELOPER_BUILD)
    const ScopedEnvVar logLevel{"NPU_VM_LOG_LEVEL", "LOG_INFO"};
    const ScopedEnvVar logColors{"NPU_VM_LOG_COLORS", "0"};
#endif

    const std::string output = captureLogOutput([] {
        NPU_VM_LOG_INFO("missing={} extra={}", 1, 2, 3, "tail");
    });

#if BUILD_LOG_LEVEL >= 4  // 4 => Info, Warn, Error, Fatal are enabled
    expectLogContains(output, "missing=1 extra=2 3 tail");
#else
    EXPECT_TRUE(output.empty());
#endif
}

TEST_F(VirtualMachineLoggerTest, SerializesNullCStringAsEmptyText) {
#if defined(VPUX_DEVELOPER_BUILD)
    const ScopedEnvVar logLevel{"NPU_VM_LOG_LEVEL", "LOG_INFO"};
    const ScopedEnvVar logColors{"NPU_VM_LOG_COLORS", "0"};
#endif

    const std::string output = captureLogOutput([] {
        NPU_VM_LOG_INFO("nullable={}suffix", static_cast<const char*>(nullptr));
    });

#if BUILD_LOG_LEVEL >= 4  // 4 => Info, Warn, Error, Fatal are enabled
    expectLogContains(output, "nullable=suffix");
#else
    EXPECT_TRUE(output.empty());
#endif
}

TEST_F(VirtualMachineLoggerTest, HonorsBuildLogLevelThresholds) {
#if defined(VPUX_DEVELOPER_BUILD)
    const ScopedEnvVar logLevel{"NPU_VM_LOG_LEVEL", "LOG_TRACE"};
    const ScopedEnvVar logColors{"NPU_VM_LOG_COLORS", "0"};
#endif

    const std::string output = captureLogOutput([] {
        NPU_VM_LOG_TRACE("trace message");
        NPU_VM_LOG_DEBUG("debug message");
        NPU_VM_LOG_INFO("info message");
        NPU_VM_LOG_WARN("warn message");
        NPU_VM_LOG_ERROR("error message");
        NPU_VM_LOG_FATAL("fatal message");
    });

#if BUILD_LOG_LEVEL == 6  // 6 => Trace is enabled (all levels enabled)
#if defined(VPUX_DEVELOPER_BUILD)
    expectLogContains(output, "trace message");
#else
    expectLogNotContains(output, "trace message");
#endif
#else
    expectLogNotContains(output, "trace message");
#endif

#if BUILD_LOG_LEVEL >= 5  // 5 => Debug and above are enabled
#if defined(VPUX_DEVELOPER_BUILD)
    expectLogContains(output, "debug message");
#else
    expectLogNotContains(output, "debug message");
#endif
#else
    expectLogNotContains(output, "debug message");
#endif

#if BUILD_LOG_LEVEL >= 4  // 4 => Info, Warn, Error, Fatal are enabled
    expectLogContains(output, "info message");
#else
    expectLogNotContains(output, "info message");
#endif

#if BUILD_LOG_LEVEL >= 3  // 3 => Warn, Error, Fatal are enabled
    expectLogContains(output, "warn message");
#else
    expectLogNotContains(output, "warn message");
#endif

#if BUILD_LOG_LEVEL >= 2  // 2 => Error and Fatal are enabled
    expectLogContains(output, "error message");
#else
    expectLogNotContains(output, "error message");
#endif

#if BUILD_LOG_LEVEL >= 1  // 1 => Fatal is enabled
    expectLogContains(output, "fatal message");
#else
    expectLogNotContains(output, "fatal message");
#endif
}

#if defined(VPUX_DEVELOPER_BUILD)

TEST_F(VirtualMachineLoggerTest, FormatsTimestampLevelSourceAndMessage) {
    const ScopedEnvVar logLevel{"NPU_VM_LOG_LEVEL", "LOG_INFO"};
    const ScopedEnvVar logColors{"NPU_VM_LOG_COLORS", "0"};
    const ScopedEnvVar fullPath{"NPU_VM_LOG_FULL_PATH", "0"};

    const std::string output = captureLogOutput([] {
        NPU_VM_LOG_INFO("format components message");
    });

#if BUILD_LOG_LEVEL >= 4  // 4 => Info, Warn, Error, Fatal are enabled
    const std::regex logFormatRegex(
            R"(^\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3}\] \[INFO\] logger\.cpp:[0-9]+ format components message\n$)");
    EXPECT_TRUE(std::regex_match(output, logFormatRegex)) << "Actual log: " << output;
#else
    EXPECT_TRUE(output.empty());
#endif
}

TEST_F(VirtualMachineLoggerTest, DeveloperBuildUsesInfoAsDefaultLevelWithoutEnvironmentVariable) {
    const ScopedEnvVar clearNpuVmLevel{"NPU_VM_LOG_LEVEL", nullptr};
    const ScopedEnvVar clearOvLevel{"OV_NPU_LOG_LEVEL", nullptr};
    const ScopedEnvVar logColors{"NPU_VM_LOG_COLORS", "0"};

    const std::string output = captureLogOutput([] {
        NPU_VM_LOG_INFO("info without explicit level");
        NPU_VM_LOG_WARN("warn without explicit level");
        NPU_VM_LOG_ERROR("error without explicit level");
        NPU_VM_LOG_FATAL("fatal without explicit level");
        NPU_VM_LOG_DEBUG("debug without explicit level");
        NPU_VM_LOG_TRACE("trace without explicit level");
    });

#if BUILD_LOG_LEVEL >= 4  // 4 => Info, Warn, Error, Fatal are enabled
    expectLogContains(output, "info without explicit level");
#else
    expectLogNotContains(output, "info without explicit level");
#endif

#if BUILD_LOG_LEVEL >= 3  // 3 => Warn, Error, Fatal are enabled
    expectLogContains(output, "warn without explicit level");
#else
    expectLogNotContains(output, "warn without explicit level");
#endif

#if BUILD_LOG_LEVEL >= 2  // 2 => Error and Fatal are enabled
    expectLogContains(output, "error without explicit level");
#else
    expectLogNotContains(output, "error without explicit level");
#endif

#if BUILD_LOG_LEVEL >= 1  // 1 => Fatal is enabled
    expectLogContains(output, "fatal without explicit level");
#else
    expectLogNotContains(output, "fatal without explicit level");
#endif

    expectLogNotContains(output, "debug without explicit level");
    expectLogNotContains(output, "trace without explicit level");
}

TEST_F(VirtualMachineLoggerTest, DeveloperBuildPrioritizesNpuVmLogLevelOverOvNpuLogLevel) {
    const ScopedEnvVar npuVmLevel{"NPU_VM_LOG_LEVEL", "LOG_FATAL"};
    const ScopedEnvVar ovLevel{"OV_NPU_LOG_LEVEL", "LOG_TRACE"};
    const ScopedEnvVar logColors{"NPU_VM_LOG_COLORS", "0"};

    const std::string output = captureLogOutput([] {
        NPU_VM_LOG_INFO("info message hidden by precedence");
        NPU_VM_LOG_FATAL("fatal message visible by precedence");
    });

    expectLogNotContains(output, "info message hidden by precedence");
    expectLogContains(output, "fatal message visible by precedence");
}

TEST_F(VirtualMachineLoggerTest, DeveloperBuildPrioritizesNpuVmFilterOverIeFilter) {
    const ScopedEnvVar npuVmLevel{"NPU_VM_LOG_LEVEL", "LOG_INFO"};
    const ScopedEnvVar npuVmFilter{"NPU_VM_LOG_FILTER", "does_not_match_logger_file"};
    const ScopedEnvVar ieFilter{"IE_NPU_LOG_FILTER", "logger\\.cpp"};
    const ScopedEnvVar logColors{"NPU_VM_LOG_COLORS", "0"};

    const std::string output = captureLogOutput([] {
        NPU_VM_LOG_INFO("filter precedence message");
    });

    EXPECT_TRUE(output.empty());
}

TEST_F(VirtualMachineLoggerTest, DeveloperBuildPrintsNothingOnInvalidFilter) {
    const ScopedEnvVar npuVmLevel{"NPU_VM_LOG_LEVEL", "LOG_INFO"};
    const ScopedEnvVar invalidNpuVmFilter{"NPU_VM_LOG_FILTER", "("};
    const ScopedEnvVar clearIeFilter{"IE_NPU_LOG_FILTER", nullptr};
    const ScopedEnvVar logColors{"NPU_VM_LOG_COLORS", "0"};

    const std::string output = captureLogOutput([] {
        NPU_VM_LOG_INFO("invalid filter message");
    });
    EXPECT_TRUE(output.empty());
}

TEST_F(VirtualMachineLoggerTest, DeveloperBuildUsesIeFilterWhenNpuVmFilterMissing) {
    const ScopedEnvVar npuVmLevel{"NPU_VM_LOG_LEVEL", "LOG_INFO"};
    const ScopedEnvVar clearNpuVmFilter{"NPU_VM_LOG_FILTER", nullptr};
    const ScopedEnvVar ieFilter{"IE_NPU_LOG_FILTER", "logger\\.cpp"};
    const ScopedEnvVar logColors{"NPU_VM_LOG_COLORS", "0"};

    const std::string output = captureLogOutput([] {
        NPU_VM_LOG_INFO("ie filter fallback message");
    });

#if BUILD_LOG_LEVEL >= 4  // 4 => Info, Warn, Error, Fatal are enabled
    expectLogContains(output, "ie filter fallback message");
#else
    EXPECT_TRUE(output.empty());
#endif
}

TEST_F(VirtualMachineLoggerTest, DeveloperBuildCanEnableColors) {
    const ScopedEnvVar npuVmLevel{"NPU_VM_LOG_LEVEL", "LOG_INFO"};
    const ScopedEnvVar logColors{"NPU_VM_LOG_COLORS", "1"};

    const std::string output = captureLogOutput([] {
        NPU_VM_LOG_INFO("message with ansi colors");
    });

#if BUILD_LOG_LEVEL >= 4  // 4 => Info, Warn, Error, Fatal are enabled
    expectLogContains(output, "\x1b[");
#else
    EXPECT_TRUE(output.empty());
#endif
}

TEST_F(VirtualMachineLoggerTest, DeveloperBuildCanDisableColors) {
    const ScopedEnvVar npuVmLevel{"NPU_VM_LOG_LEVEL", "LOG_INFO"};
    const ScopedEnvVar logColors{"NPU_VM_LOG_COLORS", "0"};

    const std::string output = captureLogOutput([] {
        NPU_VM_LOG_INFO("message without ansi colors");
    });

#if BUILD_LOG_LEVEL >= 4  // 4 => Info, Warn, Error, Fatal are enabled
    expectLogNotContains(output, "\x1b[");
    expectLogNotContains(output, "\033[");
#else
    EXPECT_TRUE(output.empty());
#endif
}

TEST_F(VirtualMachineLoggerTest, DeveloperBuildCanEnableFullSourcePathInOutput) {
    const ScopedEnvVar npuVmLevel{"NPU_VM_LOG_LEVEL", "LOG_INFO"};
    const ScopedEnvVar logColors{"NPU_VM_LOG_COLORS", "0"};
    const ScopedEnvVar fullPath{"NPU_VM_LOG_FULL_PATH", "1"};

    const std::string output = captureLogOutput([] {
        NPU_VM_LOG_INFO("message with full path");
    });

#if BUILD_LOG_LEVEL >= 4  // 4 => Info, Warn, Error, Fatal are enabled
#if defined(_WIN32)
    expectLogContains(output, "virtual_machine\\logger.cpp:");
#else
    expectLogContains(output, "virtual_machine/logger.cpp:");
#endif
#else
    EXPECT_TRUE(output.empty());
#endif
}

#endif  // defined(VPUX_DEVELOPER_BUILD)

}  // namespace
