// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_pipelines.h"

#include <math.h>

#include <algorithm>
#include <array>
#include <inference_engine.hpp>
#include <string>

using namespace InferenceEngine;

// Number of pipeline runs before it starts measuring
#define WARMUP_STEPS 30
// Number memory peaks ignored. LibC memory manager can produce peaks with
// overall flat consumption
#define MAX_OUTLIERS 5
// Maximum number of measuring pipeline restarts
#define MAX_RETRY 3
// Maximum values to compute an average for reference
#define MAX_AVERAGE 100
// Size of log line string to pre-allocate
#define LOG_LINE_RESERVE 1024
// A threshold for which memory growth will be considered an error
#define THRESHOLD 0.1

// Measure values
enum MeasureValue { VMRSS = 0, VMHWM, VMSIZE, VMPEAK, THREADS, MeasureValueMax };

namespace util {
template <typename In, typename Out, typename Func>
void transform(const In& in, Out& out, const Func& func) {
    std::transform(std::begin(in), std::end(in), std::begin(out), func);
}

template <typename In1, typename In2, typename Out, typename Func>
void transform(const In1& in1, const In2& in2, Out& out, const Func& func) {
    std::transform(std::begin(in1), std::end(in1), std::begin(in2), std::begin(out), func);
}
}  // namespace util

TestResult common_test_pipeline(const std::function<void()>& test_pipeline, const int& n) {
    int retry_count = 0;
    float mem_threshold = THRESHOLD;
    std::array<long, MeasureValueMax> cur = {0};           // measured for current iteration
    std::array<long, MeasureValueMax> ref = {0};           // recorded reference
    std::array<long, MeasureValueMax> diff = {0};          // difference between current and reference
    std::array<bool, MeasureValueMax> outlier = {0};       // flag if current does not fit threshold
    std::array<int, MeasureValueMax> outlier_count = {0};  // counter for how many times current does not fit threshold
    std::array<float, MeasureValueMax> threshold = {0};    // ref * THRESHOLD
    std::vector<std::array<long, MeasureValueMax>> past;   // past measures
    std::string progress_str;

    progress_str.reserve(LOG_LINE_RESERVE);
    past.resize(std::min(n / 2, MAX_AVERAGE));

    log_info("Warming up for " << WARMUP_STEPS << " iterations");
    log_info("i\tVMRSS\tVMHWM\tVMSIZE\tVMPEAK\tTHREADS");
    int measure_count = n;
    for (size_t iteration = 0; measure_count > 0; iteration++) {
        // Warm up to take reference values
        test_pipeline();
        getVmValues(cur[VMSIZE], cur[VMPEAK], cur[VMRSS], cur[VMHWM]);
        cur[THREADS] = getThreadsNum();
        past[iteration % past.size()] = cur;
        progress_str = std::to_string(iteration + 1) + "\t" + std::to_string(cur[VMRSS]) + "\t" +
                       std::to_string(cur[VMHWM]) + "\t" + std::to_string(cur[VMSIZE]) + "\t" +
                       std::to_string(cur[VMPEAK]) + "\t" + std::to_string(cur[THREADS]);

        // measure
        if (iteration >= WARMUP_STEPS) {
            // set reference
            if (WARMUP_STEPS == iteration || (retry_count < MAX_RETRY && (outlier_count[VMRSS] > MAX_OUTLIERS ||
                                                                          outlier_count[VMHWM] > MAX_OUTLIERS))) {
                if (0 != retry_count) log_info("Retrying " << retry_count + 1 << " of " << MAX_RETRY);
                retry_count++;
                measure_count = n;
                outlier_count = {0};
                // set reference as an average of `past` elements
                ref = {0};
                size_t past_size = std::min(iteration + 1, past.size());  // count number of past elements
                for (size_t i = 0; i < past_size; i++) {
                    // ref = ref + past
                    util::transform(ref, past[i], ref, [](long ref_val, long past_val) -> long {
                        return ref_val + past_val;
                    });
                }
                // ref = ref / past_size
                util::transform(ref, ref, [&past_size](long ref_val) -> float {
                    return ref_val / past_size;
                });
                // threshold = THRESHOLD * ref
                util::transform(ref, threshold, [](long ref_val) -> float {
                    return THRESHOLD * ref_val;
                });
                log_info("Setting thresholds to average of "
                         << past_size << " past elements:"
                         << " VMRSS=" << ref[VMRSS] << "(+-" << static_cast<int>(threshold[VMRSS]) << "),"
                         << " VMHWM=" << ref[VMHWM] << "(+-" << static_cast<int>(threshold[VMHWM]) << ")");
            }
            measure_count--;
            // diff = cur - ref
            util::transform(cur, ref, diff, [](long cur_val, long ref_val) -> long {
                // no labs() here - ignore cur smaller than ref
                return cur_val - ref_val;
            });
            // outlier = diff > threshold
            util::transform(diff, threshold, outlier, [](long diff_val, float threshold_val) -> bool {
                return diff_val > threshold_val;
            });
            // outlier_count = outlier_count + (outlier ? 1 : 0)
            util::transform(outlier, outlier_count, outlier_count,
                            [](bool outlier_val, long outlier_count_val) -> long {
                                return outlier_count_val + (outlier_val ? 1 : 0);
                            });

            if (outlier[VMRSS]) {
                progress_str += "\t<-VMRSS outlier";
            }
            if (outlier[VMHWM]) {
                progress_str += "\t<-VMHWM outlier";
            }
        }

        log_info(progress_str);
    }

    if (outlier_count[VMRSS] > MAX_OUTLIERS)
        return TestResult(TestStatus::TEST_FAILED, "Test failed: RSS virtual memory consumption grown too much.");

    if (outlier_count[VMHWM] > MAX_OUTLIERS)
        return TestResult(TestStatus::TEST_FAILED, "Test failed: HWM virtual memory consumption grown too much.");

    return TestResult(TestStatus::TEST_OK, "");
}

TestResult test_load_unload_plugin(const std::string& target_device, const int& n) {
    log_info("Load/unload plugin for device: " << target_device << " for " << n << " times");
    return common_test_pipeline(load_unload_plugin(target_device), n);
}

TestResult test_read_network(const std::string& model, const int& n) {
    log_info("Read network: \"" << model << "\" for " << n << " times");
    return common_test_pipeline(create_cnnnetwork(model), n);
}

TestResult test_create_cnnnetwork(const std::string& model, const int& n) {
    log_info("Create CNNNetwork from network: \"" << model << "\" for " << n << " times");
    return common_test_pipeline(create_cnnnetwork(model), n);
}

TestResult test_cnnnetwork_reshape_batch_x2(const std::string& model, const int& n) {
    log_info("Reshape to batch*=2 of CNNNetwork created from network: \"" << model << "\" for " << n << " times");
    return common_test_pipeline(cnnnetwork_reshape_batch_x2(model), n);
}

TestResult test_set_input_params(const std::string& model, const int& n) {
    log_info("Apply preprocessing for CNNNetwork from network: \"" << model << "\" for " << n << " times");
    return common_test_pipeline(set_input_params(model), n);
}

TestResult test_create_exenetwork(const std::string& model, const std::string& target_device, const int& n) {
    log_info("Create ExecutableNetwork from network: \"" << model << "\" for device: \"" << target_device << "\" for "
                                                         << n << " times");
    return common_test_pipeline(create_exenetwork(model, target_device), n);
}

TestResult test_recreate_exenetwork(InferenceEngine::Core& ie, const std::string& model,
                                    const std::string& target_device, const int& n) {
    log_info("Recreate ExecutableNetwork from network within existing InferenceEngine::Core: \""
             << model << "\" for device: \"" << target_device << "\" for " << n << " times");
    return common_test_pipeline(recreate_exenetwork(ie, model, target_device), n);
}

TestResult test_create_infer_request(const std::string& model, const std::string& target_device, const int& n) {
    log_info("Create InferRequest from network: \"" << model << "\" for device: \"" << target_device << "\" for " << n
                                                    << " times");
    return common_test_pipeline(create_infer_request(model, target_device), n);
}

TestResult test_recreate_infer_request(ExecutableNetwork& network, const std::string& model,
                                       const std::string& target_device, const int& n) {
    log_info("Create InferRequest from network: \"" << model << "\" for device: \"" << target_device << "\" for " << n
                                                    << " times");
    return common_test_pipeline(recreate_infer_request(network), n);
}

TestResult test_infer_request_inference(const std::string& model, const std::string& target_device, const int& n) {
    log_info("Inference of InferRequest from network: \"" << model << "\" for device: \"" << target_device << "\" for "
                                                          << n << " times");
    return common_test_pipeline(infer_request_inference(model, target_device), n);
}

TestResult test_reinfer_request_inference(InferenceEngine::InferRequest& infer_request,
                                          InferenceEngine::CNNNetwork& cnnNetwork, const std::string& model,
                                          const std::string& target_device, const int& n) {
    log_info("Inference of InferRequest from network: \"" << model << "\" for device: \"" << target_device << "\" for "
                                                          << n << " times");
    return common_test_pipeline(reinfer_request_inference(infer_request, cnnNetwork), n);
}
