#pragma once

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <exception>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "mem_usage.hpp"
#include "openvino/openvino.hpp"

namespace ov::test::behavior::host_compile_profile {

class Stage {
public:
    Stage(const char* name, const void* object) : name(name), object(object) {
        static std::atomic<size_t> nextId{0};
        id = nextId.fetch_add(1, std::memory_order_relaxed);
        peakBefore = ::intel_npu::get_peak_memory_usage();
        exceptionsBefore = std::uncaught_exceptions();
        write("begin", 0.0, peakBefore);
        started = std::chrono::steady_clock::now();
    }

    ~Stage() noexcept {
        const auto finished = std::chrono::steady_clock::now();
        try {
            write(std::uncaught_exceptions() > exceptionsBefore ? "exception" : "end",
                  std::chrono::duration<double, std::milli>(finished - started).count(),
                  ::intel_npu::get_peak_memory_usage());
        } catch (const std::exception& error) {
            std::fprintf(stderr, "[HOST_COMPILE_PROFILE] id=%zu measurement_error=%s\n", id, error.what());
        }
    }

    Stage(const Stage&) = delete;
    Stage& operator=(const Stage&) = delete;

private:
    void write(const char* event, double elapsedMs, int64_t peak) const {
#ifdef _WIN32
        constexpr auto metric = "process_peak_working_set_KiB";
#else
        constexpr auto metric = "process_VmPeak_KiB";
#endif
        const auto* test = testing::UnitTest::GetInstance()->current_test_info();
        std::ostringstream line;
        line << "[HOST_COMPILE_PROFILE] id=" << id << " event=" << event << " stage=\"" << name
             << "\" thread=" << std::this_thread::get_id() << " object=" << object
             << " test=" << (test ? test->test_suite_name() : "unknown") << '.' << (test ? test->name() : "unknown")
             << " elapsed_ms=" << elapsedMs << ' ' << metric << '=' << peak
             << " process_peak_growth_KiB=" << peak - peakBefore << '\n';
        static std::mutex outputMutex;
        const auto text = line.str();
        std::lock_guard<std::mutex> lock(outputMutex);
        std::fputs(text.c_str(), stderr);
    }

    const char* name;
    const void* object;
    size_t id = 0;
    int64_t peakBefore = 0;
    int exceptionsBefore = 0;
    std::chrono::steady_clock::time_point started;
};

inline ov::CompiledModel compile(ov::Core& core,
                                 const std::shared_ptr<ov::Model>& model,
                                 const std::string& device,
                                 const ov::AnyMap& config = {}) {
    Stage stage(device == "TEMPLATE" ? "TEMPLATE compile" : "NPU compile", model.get());
    return core.compile_model(model, device, config);
}

inline ov::InferRequest createRequest(ov::CompiledModel& model, const char* device) {
    Stage stage(device, &model);
    return model.create_infer_request();
}

inline void infer(ov::InferRequest& request, const char* device) {
    Stage stage(device, &request);
    request.infer();
}

inline void startAsync(ov::InferRequest& request) {
    Stage stage("start_async", &request);
    request.start_async();
}

inline void wait(ov::InferRequest& request) {
    Stage stage("wait", &request);
    request.wait();
}

template <typename... Arguments>
void compare(Arguments&&... arguments) {
    Stage stage("compare", nullptr);
    ov::test::utils::compare(std::forward<Arguments>(arguments)...);
}

}  // namespace ov::test::behavior::host_compile_profile