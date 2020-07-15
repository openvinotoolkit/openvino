// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_pipelines.h"

#include <string>
#include <math.h>

#define REPORTING_THRESHOLD 1.3
// delimiter used for measurements print. Should be compatible with script parses tests logs
#define MEMCHECK_DELIMITER "\t\t"


MemCheckPipeline::MemCheckPipeline() {
    start_measures[VMRSS] = (long) getVmRSSInKB();
    start_measures[VMHWM] = start_measures[VMRSS];
    start_measures[VMSIZE] = (long) getVmSizeInKB();
    start_measures[VMPEAK] = start_measures[VMSIZE];
    start_measures[THREADS] = (long) getThreadsNum();

    std::copy(start_measures.begin(), start_measures.end(), measures.begin());
}

MemCheckPipeline::~MemCheckPipeline() {
    // message required for DB data upload
    log_info("Current values of virtual memory consumption:");
    print_measures();
}

void MemCheckPipeline::do_measures() {
    measures[VMRSS] = (long) getVmRSSInKB();
    measures[VMHWM] = (long) getVmHWMInKB();
    measures[VMSIZE] = (long) getVmSizeInKB();
    measures[VMPEAK] = (long) getVmPeakInKB();
    measures[THREADS] = (long) getThreadsNum();     // TODO: resolve *-32295
}

std::array<long, MeasureValueMax> MemCheckPipeline::get_measures() {
    std::array<long, MeasureValueMax> temp;
    std::transform(std::begin(measures), std::end(measures), std::begin(start_measures), std::begin(temp),
                   [](long measure, long start_measure) -> long {
                       return measure - start_measure;
                   });
    return temp;
}

 std::string MemCheckPipeline::get_measures_as_str() {
    std::array<long, MeasureValueMax> temp = get_measures();
    std::string str = std::to_string(*temp.begin());
    for (auto it = temp.begin() + 1; it != temp.end(); it++)
        str += MEMCHECK_DELIMITER + std::to_string(*it);
    return str;
}

void MemCheckPipeline::print_measures() {
    log_info(util::get_measure_values_headers(MEMCHECK_DELIMITER));
    log_info(get_measures_as_str());
}

void MemCheckPipeline::print_actual_measures() {
    do_measures();
    print_measures();
}

std::string MemCheckPipeline::get_reference_record_for_test(std::string test_name, std::string model_name,
                                              std::string target_device) {
    std::stringstream ss;
    ss << "Record to update reference config: "
       << "<model path=\"" << model_name << "\"" <<
       " test=\"" << test_name << "\" device=\"" << target_device <<
       "\" vmsize=\"" << (int) (measures[VMSIZE] * REPORTING_THRESHOLD) <<
       "\" vmpeak=\"" << (int) (measures[VMPEAK] * REPORTING_THRESHOLD) <<
       "\" vmrss=\"" << (int) (measures[VMRSS] * REPORTING_THRESHOLD) <<
       "\" vmhwm=\"" << (int) (measures[VMHWM] * REPORTING_THRESHOLD) << "\" />";
    return ss.str();
}

TestResult common_test_pipeline(const std::function<std::array<long, MeasureValueMax>()>& test_pipeline,
                                const std::array<long, MeasureValueMax> &references) {
    log_info("Reference values of virtual memory consumption:");
    log_info(util::get_measure_values_headers(MEMCHECK_DELIMITER));
    log_info(util::get_measure_values_as_str(references, MEMCHECK_DELIMITER));

    std::array<long, MeasureValueMax> measures = test_pipeline();

    if ((!Environment::Instance().getCollectResultsOnly()) && (measures[VMRSS] > references[VMRSS]))
        return TestResult(TestStatus::TEST_FAILED,
                          "Test failed: RSS virtual memory consumption became greater than reference.\n"
                          "Reference RSS memory consumption: " + std::to_string(references[VMRSS]) + " KB.\n" +
                          "Current RSS memory consumption: " + std::to_string(measures[VMRSS]) + " KB.\n");

    if ((!Environment::Instance().getCollectResultsOnly()) && (measures[VMHWM] > references[VMHWM]))
        return TestResult(TestStatus::TEST_FAILED,
                          "Test failed: HWM (peak of RSS) virtual memory consumption is greater than reference.\n"
                          "Reference HWM of memory consumption: " + std::to_string(references[VMHWM]) + " KB.\n" +
                          "Current HWM of memory consumption: " + std::to_string(measures[VMHWM]) + " KB.\n");

    return TestResult(TestStatus::TEST_OK, "");
}
