// Copyright (C) 2018-2022 Intel Corporation
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
#include <vpu/utils/log_level.hpp>

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

OutputStream::Ptr defaultOutput(const std::string& fileName = std::string());

class Logger final {
public:
    using Ptr = std::shared_ptr<Logger>;

    static constexpr const int IDENT_SIZE = 4;

    class Section final {
    public:
        inline explicit Section(const Logger::Ptr& log) : _log(log) {
            IE_ASSERT(_log != nullptr);
            ++_log->_ident;
        }

        inline ~Section() {
            --_log->_ident;
        }

    private:
        Logger::Ptr _log;
    };

public:
    inline Logger(std::string name, LogLevel lvl, OutputStream::Ptr out) :
            _name(std::move(name)), _logLevel(lvl), _out(std::move(out)) {
        IE_ASSERT(_out != nullptr);
    }

    inline LogLevel level() const {
        return _logLevel;
    }
    inline bool isActive(LogLevel msgLevel) const {
        return static_cast<int>(msgLevel) <= static_cast<int>(_logLevel);
    }
    void setLevel(LogLevel lvl) {
        _logLevel = lvl;
    }

    template <typename... Args>
    inline void fatal(const char* format, const Args&... args) const {
        addEntry(LogLevel::Fatal, format, args...);
    }

    template <typename... Args>
    inline void error(const char* format, const Args&... args) const {
        addEntry(LogLevel::Error, format, args...);
    }

    template <typename... Args>
    inline void warning(const char* format, const Args&... args) const {
        addEntry(LogLevel::Warning, format, args...);
    }

    template <typename... Args>
    inline void info(const char* format, const Args&... args) const {
        addEntry(LogLevel::Info, format, args...);
    }

    template <typename... Args>
    inline void debug(const char* format, const Args&... args) const {
        addEntry(LogLevel::Debug, format, args...);
    }

    template <typename... Args>
    inline void trace(const char* format, const Args&... args) const {
        addEntry(LogLevel::Trace, format, args...);
    }

private:
    template <typename... Args>
    void addEntry(LogLevel msgLevel, const char* format, const Args&... args) const {
        if (!isActive(msgLevel)) {
            return;
        }

        _out->lock();
        AutoScope scope([this] { _out->unlock(); });

        printHeader(msgLevel);
        formatPrint(_out->get(), format, args...);
        printFooter();

        _out->get().flush();
    }

    void printHeader(LogLevel msgLevel) const;
    void printFooter() const;

private:
    std::string _name;
    LogLevel _logLevel = LogLevel::None;
    OutputStream::Ptr _out;

    size_t _ident = 0;

    friend class Section;
};

#define VPU_LOGGER_SECTION(log) vpu::Logger::Section VPU_COMBINE(logSec, __LINE__) (log)

}  // namespace vpu
