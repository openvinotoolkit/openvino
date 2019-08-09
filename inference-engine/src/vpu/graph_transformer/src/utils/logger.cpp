// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/utils/logger.hpp>

#include <mutex>
#include <string>
#include <fstream>
#include <iomanip>
#include <memory>

namespace vpu {

//
// OutputStream
//

namespace {

class ConsoleOutput final : public OutputStream {
public:
    std::ostream& get() override { return std::cout; }

    bool supportColors() const override {
#ifdef _WIN32
        // TODO: check if Windows supports colors in terminal
        return false;
#else
        return true;
#endif
    }

    void lock() override { _mtx.lock(); }
    void unlock() override { _mtx.unlock(); }

private:
    std::mutex _mtx;
};

class FileOutput final : public OutputStream {
public:
    explicit FileOutput(const std::string& fileName) : _file(fileName) {
        if (!_file.is_open()) {
            std::cerr << "Failed to open LOG file\n";
            std::abort();
        }
    }

    std::ostream& get() override { return _file; }

    bool supportColors() const override { return false; }

    void lock() override { _mtx.lock(); }

    void unlock() override { _mtx.unlock(); }

private:
    std::ofstream _file;
    std::mutex _mtx;
};

}  // namespace

OutputStream::Ptr consoleOutput() {
    static auto obj = std::make_shared<ConsoleOutput>();
    return obj;
}

OutputStream::Ptr fileOutput(const std::string& fileName) {
    return std::make_shared<FileOutput>(fileName);
}

//
// Logger
//

namespace {

const auto COLOR_RED = "\033[1;31m";
const auto COLOR_GRN = "\033[1;32m";
const auto COLOR_YEL = "\033[1;33m";
const auto COLOR_BLU = "\033[1;34m";
const auto COLOR_END = "\033[0m";

}  // namespace

void Logger::printHeader(LogLevel msgLevel) const noexcept {
    try {
        if (_out->supportColors()) {
            static const EnumMap<LogLevel, const char *> levelColors{
                    {LogLevel::Error,   COLOR_RED},
                    {LogLevel::Warning, COLOR_YEL},
                    {LogLevel::Info,    COLOR_GRN},
                    {LogLevel::Debug,   COLOR_BLU},
            };

            _out->get() << levelColors.at(msgLevel);
        }

        _out->get() << "[" << std::setw(7) << std::left << msgLevel << "]";
        _out->get() << "[VPU]";
        _out->get() << "[" << _name << "] ";

        for (size_t i = 0; i < _ident; ++i) {
            for (int j = 0; j < IDENT_SIZE; ++j) {
                _out->get() << ' ';
            }
        }
    } catch (...) {
        std::cerr << "[VPU] Cannot print header\n";
        std::abort();
    }
}

void Logger::printFooter() const noexcept {
    try {
        if (_out->supportColors()) {
            _out->get() << COLOR_END;
        }
        _out->get() << std::endl;
    } catch (...) {
        std::cerr << "[VPU] Cannot print footer\n";
        std::abort();
    }
}

}  // namespace vpu
