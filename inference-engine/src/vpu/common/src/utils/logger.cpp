// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/logger.hpp>

#include <ctime>

#include <mutex>
#include <string>
#include <fstream>
#include <iomanip>
#include <memory>
#include <chrono>
#include <unordered_map>

namespace vpu {

//
// OutputStream
//

namespace {

class OutputStreamBase : public OutputStream {
public:
    void lock() override {
        _mtx.lock();
    }
    void unlock() override {
        _mtx.unlock();
    }

private:
    std::mutex _mtx;
};

class ConsoleOutput final : public OutputStreamBase {
public:
    std::ostream& get() override {
        return std::cout;
    }

    bool supportColors() const override {
#ifdef _WIN32
        // TODO: check if Windows supports colors in terminal
        return false;
#else
        return true;
#endif
    }
};

class FileOutput final : public OutputStreamBase {
public:
    explicit FileOutput(const std::string& fileName) : _file(fileName) {
        if (!_file.is_open()) {
            std::cerr << "Failed to open LOG file\n";
            std::abort();
        }
    }

    std::ostream& get() override {
        return _file;
    }

    bool supportColors() const override {
        return false;
    }

private:
    std::ofstream _file;
};

}  // namespace

OutputStream::Ptr consoleOutput() {
    static auto obj = std::make_shared<ConsoleOutput>();
    return obj;
}

OutputStream::Ptr fileOutput(const std::string& fileName) {
    static std::unordered_map<std::string, std::weak_ptr<FileOutput>> openFiles;
    static std::mutex mtx;

    std::lock_guard<std::mutex> lock(mtx);

    const auto it = openFiles.find(fileName);
    if (it != openFiles.end()) {
        if (const auto stream = it->second.lock()) {
            return stream;
        }
    }

    const auto stream = std::make_shared<FileOutput>(fileName);
    openFiles.emplace(fileName, stream);

    return stream;
}

OutputStream::Ptr defaultOutput(const std::string& fileName) {
    return fileName.empty() ? consoleOutput() : fileOutput(fileName);
}

//
// Logger
//

namespace {

const auto COLOR_BLACK = "\033[1;30m";
const auto COLOR_RED = "\033[1;31m";
const auto COLOR_GREEN = "\033[1;32m";
const auto COLOR_YELLOW = "\033[1;33m";
const auto COLOR_BLUE = "\033[1;34m";
const auto COLOR_PURPLE = "\033[1;35m";
const auto COLOR_RESET = "\033[0m";

}  // namespace

void Logger::printHeader(LogLevel msgLevel) const noexcept {
    try {
        if (_out->supportColors()) {
            static const EnumMap<LogLevel, const char*> levelColors{
                {LogLevel::Fatal,   COLOR_BLACK},
                {LogLevel::Error,   COLOR_RED},
                {LogLevel::Warning, COLOR_YELLOW},
                {LogLevel::Info,    COLOR_GREEN},
                {LogLevel::Debug,   COLOR_BLUE},
                {LogLevel::Trace,   COLOR_PURPLE},
            };

            _out->get() << levelColors.at(msgLevel);
        }

        _out->get() << "[" << std::setw(7) << std::left << msgLevel << "]";
        _out->get() << "[VPU]";
        _out->get() << "[" << _name << "] ";

        static std::string singleIdent(IDENT_SIZE, ' ');

        for (size_t i = 0; i < _ident; ++i) {
            _out->get() << singleIdent;
        }
    } catch (...) {
        std::cerr << "[VPU] Cannot print header\n";
        std::abort();
    }
}

void Logger::printFooter() const noexcept {
    try {
        if (_out->supportColors()) {
            _out->get() << COLOR_RESET;
        }

        _out->get() << std::endl;
    } catch (...) {
        std::cerr << "[VPU] Cannot print footer\n";
        std::abort();
    }
}

}  // namespace vpu
