// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <iosfwd>
#include <string>
#include <utility>

#include <vpu/utils/extra.hpp>
#include <vpu/utils/enums.hpp>
#include <vpu/utils/auto_scope.hpp>
#include <vpu/utils/io.hpp>

namespace vpu {

//
// OutputStream
//

class OutputStream {
public:
    using Ptr = std::shared_ptr<OutputStream>;

    virtual ~OutputStream() = default;

    virtual std::ostream& get() = 0;

    virtual bool supportColors() const = 0;

    virtual void lock() = 0;
    virtual void unlock() = 0;
};

OutputStream::Ptr consoleOutput();
OutputStream::Ptr fileOutput(const std::string& fileName);

//
// Logger
//

VPU_DECLARE_ENUM(LogLevel,
    None,
    Error,
    Warning,
    Info,
    Debug
)

class Logger final {
public:
    using Ptr = std::shared_ptr<Logger>;

    class Section final {
    public:
        explicit Section(const Logger::Ptr& log) : _log(log) {
            IE_ASSERT(_log != nullptr);
            ++_log->_ident;
        }

        ~Section() {
            --_log->_ident;
        }

    private:
        Logger::Ptr _log;
    };

public:
    Logger(const std::string& name, LogLevel lvl, const OutputStream::Ptr& out) :
            _name(name), _logLevel(lvl), _out(out) {
        IE_ASSERT(_out != nullptr);
    }

    LogLevel level() const { return _logLevel; }

    template <typename... Args>
    void error(const char* format, const Args&... args) const noexcept {
        addEntry(LogLevel::Error, format, args...);
    }

    template <typename... Args>
    void warning(const char* format, const Args&... args) const noexcept {
        addEntry(LogLevel::Warning, format, args...);
    }

    template <typename... Args>
    void info(const char* format, const Args&... args) const noexcept {
        addEntry(LogLevel::Info, format, args...);
    }

    template <typename... Args>
    void debug(const char* format, const Args&... args) const  noexcept {
        addEntry(LogLevel::Debug, format, args...);
    }

private:
    template <typename... Args>
    void addEntry(LogLevel msgLevel, const char* format, const Args&... args) const noexcept {
        if (static_cast<int>(msgLevel) > static_cast<int>(_logLevel)) {
            return;
        }

        _out->lock();
        AutoScope scope([this] { _out->unlock(); });

        printHeader(msgLevel);
        formatPrint(_out->get(), format, args...);
        printFooter();
    }

    void printHeader(LogLevel msgLevel) const noexcept;
    void printFooter() const noexcept;

private:
    std::string _name;
    LogLevel _logLevel = LogLevel::None;
    OutputStream::Ptr _out;

    size_t _ident = 0;

    friend class Section;
};

#define VPU_LOGGER_SECTION(log) vpu::Logger::Section VPU_COMBINE(logSec, __LINE__) (log)

}  // namespace vpu
