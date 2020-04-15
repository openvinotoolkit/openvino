#include "tests_pipelines.h"

#include <string>
#include <math.h>
#include <chrono>

#include <inference_engine.hpp>

#define REPORTING_THRESHOLD 1.1

using namespace InferenceEngine;

#define getAlignedVmValues(vmsize, vmpeak, vmrss, vmhwm, vmsize_to_align, vmrss_to_align)   \
        getVmValues(test_cur_vmsize, test_cur_vmpeak, test_cur_vmrss, test_cur_vmhwm);      \
        test_cur_vmsize -= vmsize_before_test;                                              \
        test_cur_vmpeak -= vmsize_before_test;                                              \
        test_cur_vmrss -= vmrss_before_test;                                                \
        test_cur_vmhwm -= vmrss_before_test;

#define log_debug_ref_record_for_test(test_name)                                                            \
        log_debug("Record to update reference config: "                                                           \
                  << "<model path=\"" + model_name + "\"" + " test=\"" + test_name + "\" device=\"" +       \
                  target_device +                                                                           \
                  "\" vmsize=\"" + std::to_string((int) (test_cur_vmsize * REPORTING_THRESHOLD)) +          \
                  "\" vmpeak=\"" + std::to_string((int) (test_cur_vmpeak * REPORTING_THRESHOLD)) +          \
                  "\" vmrss=\"" + std::to_string((int) (test_cur_vmrss * REPORTING_THRESHOLD)) +            \
                  "\" vmhwm=\"" + std::to_string((int) (test_cur_vmhwm * REPORTING_THRESHOLD)) + "\" />");

#define log_info_ref_mem_usage()                                                                \
        log_info("Reference values of virtual memory consumption:");                            \
        log_info("VMRSS\t\tVMHWM\t\tVMSIZE\t\tVMPEAK");                                               \
        log_info(ref_vmrss << "\t\t" << ref_vmhwm << "\t\t" << ref_vmsize << "\t\t" << ref_vmpeak);

#define log_info_cur_mem_usage()                                                                                    \
        log_info("Current values of virtual memory consumption:");                                                  \
        log_info("VMRSS\t\tVMHWM\t\tVMSIZE\t\tVMPEAK");                                                                   \
        log_info(test_cur_vmrss << "\t\t" << test_cur_vmhwm << "\t\t" << test_cur_vmsize << "\t\t" << test_cur_vmpeak);

TestResult
test_create_exenetwork(const std::string &model_name, const std::string &model_path, const std::string &target_device,
                       const long &ref_vmsize, const long &ref_vmpeak, const long &ref_vmrss, const long &ref_vmhwm) {
    log_info("Create ExecutableNetwork from network: \"" << model_path
                                                         << "\" for device: \"" << target_device << "\"");
    long vmsize_before_test = 0, vmrss_before_test = 0,
            test_cur_vmsize = 0, test_cur_vmpeak = 0,
            test_cur_vmrss = 0, test_cur_vmhwm = 0;

    vmsize_before_test = (long) getVmSizeInKB();
    vmrss_before_test = (long) getVmRSSInKB();

    create_exenetwork(model_path, target_device)();

    getAlignedVmValues(test_cur_vmsize, test_cur_vmpeak, test_cur_vmrss, test_cur_vmhwm,
                       vmsize_before_test, vmrss_before_test);

    log_debug_ref_record_for_test("create_exenetwork");
    log_info_ref_mem_usage();
    log_info_cur_mem_usage();

    if (test_cur_vmhwm > ref_vmhwm)
        return TestResult(TestStatus::TEST_FAILED,
                          "Test failed: HWM (peak of RSS) virtual memory consumption is greater than reference.\n"
                          "Reference HWM of memory consumption: " + std::to_string(ref_vmhwm) + " KB.\n" +
                          "Current HWM of memory consumption: " + std::to_string(test_cur_vmhwm) + " KB.\n");

    return TestResult(TestStatus::TEST_OK, "");
}

TestResult
test_infer_request_inference(const std::string &model_name, const std::string &model_path,
                             const std::string &target_device,
                             const long &ref_vmsize, const long &ref_vmpeak, const long &ref_vmrss,
                             const long &ref_vmhwm) {
    log_info("Inference of InferRequest from network: \"" << model_path
                                                          << "\" for device: \"" << target_device << "\"");
    long vmsize_before_test = 0, vmrss_before_test = 0,
            test_cur_vmsize = 0, test_cur_vmpeak = 0,
            test_cur_vmrss = 0, test_cur_vmhwm = 0;
    std::chrono::system_clock::time_point t_start, t_end;
    std::chrono::duration<double> t_diff;

    vmsize_before_test = (long) getVmSizeInKB();
    vmrss_before_test = (long) getVmRSSInKB();

    Core ie;
    CNNNetwork cnnNetwork = ie.ReadNetwork(model_path);
    ExecutableNetwork exeNetwork = ie.LoadNetwork(cnnNetwork, target_device);
    InferRequest infer_request = exeNetwork.CreateInferRequest();

    log_info_ref_mem_usage();

    t_start = std::chrono::system_clock::now();
    int seconds = 1;
    do {
        infer_request.Infer();
        OutputsDataMap output_info(cnnNetwork.getOutputsInfo());
        for (auto &output : output_info)
            Blob::Ptr outputBlob = infer_request.GetBlob(output.first);
        t_end = std::chrono::system_clock::now();
        t_diff = t_end - t_start;

        getAlignedVmValues(test_cur_vmsize, test_cur_vmpeak, test_cur_vmrss, test_cur_vmhwm,
                           vmsize_before_test, vmrss_before_test);

        if (test_cur_vmrss > ref_vmrss) {
            log_debug_ref_record_for_test("infer_request_inference");
            return TestResult(TestStatus::TEST_FAILED,
                              "Test failed: RSS virtual memory consumption became greater than reference "
                              "after " + std::to_string(t_diff.count()) + " sec of inference.\n"
                              "Reference RSS memory consumption: " + std::to_string(ref_vmrss) + " KB.\n" +
                              "Current RSS memory consumption: " + std::to_string(test_cur_vmrss) + " KB.\n");
        }

        if (t_diff.count() > (double) (seconds)) {
            log_info("Current values of virtual memory consumption after " << seconds << " seconds:");
            log_info("VMRSS\t\tVMHWM\t\tVMSIZE\t\tVMPEAK");
            log_info(test_cur_vmrss << "\t\t" << test_cur_vmhwm << "\t\t" << test_cur_vmsize << "\t\t" << test_cur_vmpeak);
            seconds++;
        }
    } while (t_diff.count() < 5);
    log_debug_ref_record_for_test("infer_request_inference");

    return TestResult(TestStatus::TEST_OK, "");
}
