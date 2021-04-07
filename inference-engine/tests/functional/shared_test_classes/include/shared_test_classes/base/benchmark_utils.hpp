// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace perforemace_tests {

// TODO make this name configurable
static inline std::string fileToSave() {
    return "vino_benchmark.log";
}

template <typename Duration>
static inline int64_t microseconds(Duration d) {
    return std::chrono::duration_cast<std::chrono::microseconds>(d).count();
}

class Clock {
public:
    using SourceClock = std::chrono::system_clock;
    using Duration = SourceClock::duration;
    using TimePoint = SourceClock::time_point;

    static TimePoint now() {
        return SourceClock::now();
    }
};

class Timer {
public:
    void start() {
        beg = Clock::now();
    }

    void stop() {
        end = Clock::now();
    }

    Clock::Duration duration() const {
        return end - beg;
    }

private:
    Clock::TimePoint beg;
    Clock::TimePoint end;
};

class ExecutionTimeWriter {
public:
    virtual ~ExecutionTimeWriter() = default;

    virtual void testName(const std::string& name) = 0;
    virtual void testDurations(const std::vector<Clock::Duration>& durations) = 0;
    virtual void flush() {}
};

class ExecutionTime {
    ExecutionTime() = default;

public:
    ExecutionTime(ExecutionTime&&) = delete;
    ExecutionTime(const ExecutionTime&) = delete;
    ExecutionTime& operator=(ExecutionTime&&) = delete;
    ExecutionTime& operator=(const ExecutionTime&) = delete;

    static ExecutionTime& instance() {
        // QUESTION: do we need thread safety for access to this instance
        static ExecutionTime obj;
        return obj;
    }

    using TestName = std::string;
    using Executions = std::vector<Clock::Duration>;
    using Records = std::unordered_map<TestName, Executions>;

    const Executions& getExecutions(const TestName& name) const {
        static const Executions empty;
        const auto& found = records.find(name);
        if (found != end(records)) {
            return found->second;
        }
        return empty;
    }

    const Records& getRecords() const {
        return records;
    }

    void addRecord(const TestName& name, const Clock::Duration& d) {
        records[name].push_back(d);
    }

    void writeAll(ExecutionTimeWriter& w) const {
        for (const auto& r : records) {
            writeRecord(r, w);
        }
    }

    void write(const std::string& test_name, ExecutionTimeWriter& w) const {
        const auto& r = records.find(test_name);
        if (r == end(records)) {
            return;
        }
        writeRecord(*r, w);
    }

    void drop() {
        records = {};
    }

private:
    void writeRecord(Records::const_reference record, ExecutionTimeWriter& w) const {
        w.testName(record.first);
        w.testDurations(record.second);
    }

    Records records;
};

class FileWriter : public ExecutionTimeWriter {
public:
    FileWriter(const std::string& filename, std::ios_base::openmode mode = {}) : file(filename, mode) {}
    ~FileWriter() override {
        try {
            flush();
        } catch (...) {
        }
    }

    void testName(const std::string& name) override {
        file << "Test: [" << name << "]"
             << " : ";
    }

    void testDurations(const std::vector<Clock::Duration>& durations) override {
        file << "(exec no: " << durations.size() << ") [ ";
        const char* glue = "";
        for (const auto& d : durations) {
            file << glue << microseconds(d) << "us";
            glue = ", ";
        }
        file << " ]\n";
    }

    void flush() override {
        file.flush();
    }

private:
    std::ofstream file;
};

class CsvWriter : public ExecutionTimeWriter {
public:
    CsvWriter(const std::string& filename, std::ios_base::openmode mode = {}) : file(filename, mode) {
        if (file.tellp() == 0) {
            writeHeader();
        }
    }
    ~CsvWriter() override {
        try {
            flush();
        } catch (...) {
        }
    }

    void testName(const std::string& name) override {
        file << name << ",";
    }

    void testDurations(const std::vector<Clock::Duration>& durations) override {
        file << durations.size();
        for (const auto& d : durations) {
            file << "," << microseconds(d);
        }
        file << "\n";
    }

    void flush() override {
        file.flush();
    }

private:
    void writeHeader() {
        file << "test_name,execution_number,execution_time_-_microseconds\n";
    }
    std::ofstream file;
};

// TODO better name for this
class SingleTestSuppervisior {
public:
    void addTestDuration(Clock::Duration d) {
        ++exec_counter;
        aggregate_duration += d;
    }

    bool shouldRunNextTest() const {
        if (exec_counter >= max_exec_number) {
            return false;
        }
        return exec_counter < required_exec_number || aggregate_duration < min_exec_duration;
    }

private:
    int exec_counter{0};
    Clock::Duration aggregate_duration{};
    static constexpr int required_exec_number{5};
    static constexpr int max_exec_number{100};
    static constexpr Clock::Duration min_exec_duration{std::chrono::milliseconds{100}};
};

class Benchmark {
public:
    Benchmark(std::string name) : bench_name(std::move(name)) {}

    void endRound() {
        timer.stop();
        const auto duration = timer.duration();
        test_suppervisior.addTestDuration(duration);
        ExecutionTime::instance().addRecord(bench_name, duration);
    }

    bool nextRound() {
        if (test_suppervisior.shouldRunNextTest()) {
            timer.start();
            return true;
        }
        return false;
    }

private:
    std::string bench_name;
    Timer timer;
    SingleTestSuppervisior test_suppervisior;
};

};  // namespace perforemace_tests
