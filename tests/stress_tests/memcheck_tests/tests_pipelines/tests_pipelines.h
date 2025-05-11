// Copyright (C) 2018-2025 Intel Corporation
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
 * Also deletion of objects created in scope of class lifetime may decrease
 * values computed on previous measure.
 */
class MemCheckPipeline {
private:
    std::array<long, MeasureValueMax> start_measures;      // measures before run (will be used as baseline)

    /**
     * @brief Measures values at the current point of time
     */
    std::array<long, MeasureValueMax> _measure();

public:
    /**
     * @brief Constructs MemCheckPipeline object and
     *        measure values to use as baseline
     */
    MemCheckPipeline();

    /**
     * @brief Measures values at the current point of time and
     *        returns measurements aligned on a baseline
     */
    std::array<long, MeasureValueMax> measure();

    /**
     * @brief Measures values and records aligned measurements using provided identifier
     *        provided identifier
     */
    void record_measures(const std::string &id);

    /**
     * @brief Prepares string used for fast generation of file with references
     */
    std::string get_reference_record_for_test(std::string test_name, std::string model_name,
                                              std::string precision, std::string target_device);
};

TestResult common_test_pipeline(const std::function<std::array<long, MeasureValueMax>()> &test_pipeline,
                                const std::array<long, MeasureValueMax> &references);
// tests_pipelines/tests_pipelines.cpp
