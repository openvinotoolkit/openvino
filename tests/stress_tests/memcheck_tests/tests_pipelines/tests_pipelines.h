// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../tests_utils.h"
#include "../../common/tests_utils.h"
#include "../../common/utils.h"

#include <string>

// tests_pipelines/tests_pipelines.cpp
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
    MemCheckPipeline();

    /**
     * @brief Measures values at the current point of time
     */
    void do_measures();

    /**
     * @brief Returns measurements aligned on a baseline
     */
    std::array<long, MeasureValueMax> get_measures();

    /**
     * @brief Returns measurements as string separated within hardcoded delimiter
     */
    std::string get_measures_as_str();

    /**
     * @brief Prints headers and corresponding collected measurements using hardcoded delimiter
     */
    void print_measures();

    /**
     * @brief Upload to DataBase headers and corresponding collected measurements using hardcoded delimiter
     */
    void upload_measures(const std::string & step_name);

    /**
     * @brief Measures values at the current point of time and prints immediately
     */
    void print_actual_measures();

    /**
     * @brief Measures values at the current point of time and upload to DataBase immediately
     */
    void upload_actual_measures(const std::string & step_name);

    /**
     * @brief Prepares string used for fast generation of file with references
     */
    std::string get_reference_record_for_test(std::string test_name, std::string model_name,
                                              std::string target_device);
};

TestResult common_test_pipeline(const std::function<std::array<long, MeasureValueMax>()>& test_pipeline,
                                const std::array<long, MeasureValueMax> &references);
// tests_pipelines/tests_pipelines.cpp
