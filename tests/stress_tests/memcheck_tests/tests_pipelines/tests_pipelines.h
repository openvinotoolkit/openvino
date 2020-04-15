#pragma once

#include "../../common/tests_utils.h"
#include "../../common/utils.h"
#include "../../common/ie_pipelines/pipelines.h"

#include <string>

// tests_pipelines/tests_pipelines.cpp
TestResult test_create_exenetwork(const std::string &model_name, const std::string &model_path, const std::string &target_device,
                                  const long &ref_vmsize, const long &ref_vmpeak, const long &ref_vmrss, const long &ref_vmhwm);
TestResult test_infer_request_inference(const std::string &model_name, const std::string &model_path, const std::string &target_device,
                                        const long &ref_vmsize, const long &ref_vmpeak, const long &ref_vmrss, const long &ref_vmhwm);
// tests_pipelines/tests_pipelines.cpp
