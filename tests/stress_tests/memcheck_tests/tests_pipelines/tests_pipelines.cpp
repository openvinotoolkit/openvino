// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tests_pipelines.h"

#include <string>
#include <math.h>
#include <chrono>

#include <inference_engine.hpp>

#define REPORTING_THRESHOLD 1.3
// delimiter used for measurements print. Should be compatible with script parses tests logs
#define MEMCHECK_DELIMITER "\t\t"

using namespace InferenceEngine;


/**
 * @brief Class response for encapsulating measure and measurements printing
 *
 * Current class measures only in scope of it's lifetime. In this case need
 * to note that deletion of objects created before class creation may lead
 * to negative values because of alignment on starting values.
 * Also deletion  of objects created in scope of class lifetime may decrease
 * values computed on previous measure.
 */
class MemCheckPipeline {
private:
    std::array<long, MeasureValueMax> measures;            // current measures
    std::array<long, MeasureValueMax> start_measures;      // measures before run (will be used as baseline)
public:
    /**
     * @brief Constructs MemCheckPipeline object and
     *        measure values to use as baseline
     */
    MemCheckPipeline() {
        start_measures[VMRSS] = (long) getVmRSSInKB();
        start_measures[VMHWM] = start_measures[VMRSS];
        start_measures[VMSIZE] = (long) getVmSizeInKB();
        start_measures[VMPEAK] = start_measures[VMSIZE];
        start_measures[THREADS] = (long) getThreadsNum();

        std::copy(start_measures.begin(), start_measures.end(), measures.begin());
    }

    /**
     * @brief Destructs MemCheckPipeline object and prints the latest measurements
     */
    ~MemCheckPipeline() {
        // message required for DB data upload
        log_info("Current values of virtual memory consumption:");
        print_measures();
    }

    /**
     * @brief Measures values at the current point of time
     */
    void do_measures() {
        measures[VMRSS] = (long) getVmRSSInKB();
        measures[VMHWM] = (long) getVmHWMInKB();
        measures[VMSIZE] = (long) getVmSizeInKB();
        measures[VMPEAK] = (long) getVmPeakInKB();
        measures[THREADS] = (long) getThreadsNum();     // TODO: resolve *-32295
    }

    /**
     * @brief Returns measurements aligned on a baseline
     */
    std::array<long, MeasureValueMax> get_measures() {
        std::array<long, MeasureValueMax> temp;
        std::transform(std::begin(measures), std::end(measures), std::begin(start_measures), std::begin(temp),
                       [](long measure, long start_measure) -> long {
                           return measure - start_measure;
                       });
        return temp;
    }

    /**
     * @brief Returns measurements as string separated within hardcoded delimiter
     */
    std::string get_measures_as_str() {
        std::array<long, MeasureValueMax> temp = get_measures();
        std::string str = std::to_string(*temp.begin());
        for (auto it = temp.begin() + 1; it != temp.end(); it++)
            str += MEMCHECK_DELIMITER + std::to_string(*it);
        return str;
    }

    /**
     * @brief Prints headers and corresponding measurements using hardcoded delimiter
     */
    void print_measures() {
        log_info(util::get_measure_values_headers(MEMCHECK_DELIMITER));
        log_info(get_measures_as_str());
    }

    /**
     * @brief Prepares string used for fast generation of file with references
     */
    std::string get_reference_record_for_test(std::string test_name, std::string model_name,
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
};

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


TestResult
test_create_exenetwork(const std::string &model_name, const std::string &model_path, const std::string &target_device,
                       const std::array<long, MeasureValueMax> &references) {
    log_info("Create ExecutableNetwork from network: \"" << model_path
                                                         << "\" for device: \"" << target_device << "\"");

    auto test_pipeline = [&]{
        MemCheckPipeline memCheckPipeline;

        log_info("Memory consumption before run:");
        memCheckPipeline.do_measures();
        memCheckPipeline.print_measures();

        Core ie;
        log_info("Memory consumption after Core creation:");
        memCheckPipeline.do_measures();
        memCheckPipeline.print_measures();

        ie.GetVersions(target_device);
        log_info("Memory consumption after GetCPPPluginByName (via GetVersions):");
        memCheckPipeline.do_measures();
        memCheckPipeline.print_measures();

        CNNNetwork cnnNetwork = ie.ReadNetwork(model_path);
        log_info("Memory consumption after ReadNetwork:");
        memCheckPipeline.do_measures();
        memCheckPipeline.print_measures();

        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
        log_info("Memory consumption after LoadNetwork:");
        memCheckPipeline.do_measures();
        memCheckPipeline.print_measures();

        log_debug(memCheckPipeline.get_reference_record_for_test("create_exenetwork", model_name, target_device));
        return memCheckPipeline.get_measures();
    };

    return common_test_pipeline(test_pipeline, references);
}

TestResult
test_infer_request_inference(const std::string &model_name, const std::string &model_path,
                             const std::string &target_device, const std::array<long, MeasureValueMax> &references) {
    log_info("Inference of InferRequest from network: \"" << model_path
                                                          << "\" for device: \"" << target_device << "\"");

    auto test_pipeline = [&]{
        MemCheckPipeline memCheckPipeline;

        log_info("Memory consumption before run:");
        memCheckPipeline.do_measures();
        memCheckPipeline.print_measures();

        Core ie;
        log_info("Memory consumption after Core creation:");
        memCheckPipeline.do_measures();
        memCheckPipeline.print_measures();

        ie.GetVersions(target_device);
        log_info("Memory consumption after GetCPPPluginByName (via GetVersions):");
        memCheckPipeline.do_measures();
        memCheckPipeline.print_measures();

        CNNNetwork cnnNetwork = ie.ReadNetwork(model_path);
        log_info("Memory consumption after ReadNetwork:");
        memCheckPipeline.do_measures();
        memCheckPipeline.print_measures();

        ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
        log_info("Memory consumption after LoadNetwork:");
        memCheckPipeline.do_measures();
        memCheckPipeline.print_measures();

        InferRequest inferRequest = exeNetwork.CreateInferRequest();
        log_info("Memory consumption after CreateInferRequest:");
        memCheckPipeline.do_measures();
        memCheckPipeline.print_measures();

        inferRequest.Infer();
        OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
        for (auto &output : output_info)
            Blob::Ptr outputBlob = inferRequest.GetBlob(output.first);
        log_info("Memory consumption after Inference:");
        memCheckPipeline.do_measures();
        memCheckPipeline.print_measures();

        log_debug(memCheckPipeline.get_reference_record_for_test("infer_request_inference", model_name, target_device));
        return memCheckPipeline.get_measures();
    };

    return common_test_pipeline(test_pipeline, references);
}
