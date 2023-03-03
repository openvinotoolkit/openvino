// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file cpu_streams_calculation.hpp
 * @brief A header file for CPU streams calulation implementation.
 */

#pragma once

#include <vector>

namespace ov {
namespace intel_cpu {

/**
 * @brief Enum to define possible processor type hints for CPU inference
 * @ingroup ov_runtime_cpu_prop_cpp_api
 */
enum class ProcessorType {
    UNDEFINED = -1,  //!<  Undefined value, default setting may vary by platform and performance hints
    ALL = 1,  //!<  All processors can be used. If hyper threading is enabled, both processors of oneperformance-core
              //!<  will be used.
    PHY_CORE_ONLY = 2,    //!<  Only one processor can be used per CPU core even with hyper threading enabled.
    P_CORE_ONLY = 3,      //!<  Only processors of performance-cores can be used. If hyper threading is enabled, both
                          //!<  processors of one performance-core will be used.
    E_CORE_ONLY = 4,      //!<  Only processors of efficient-cores can be used.
    PHY_P_CORE_ONLY = 5,  //!<  Only one processor can be used per performance-core even with hyper threading enabled.
};

/**
 * @brief      Limit available CPU resource in processors type table according to processor type property
 * @param[in]  input_type input value of processor type property.
 * @param[in]  proc_type_table candidate processors available at this time
 * @return     updated proc_type_table which removed unmatched processors
 */
std::vector<std::vector<int>> apply_processor_type(const ProcessorType input_type,
                                                   const std::vector<std::vector<int>> proc_type_table);

}  // namespace intel_cpu
}  // namespace ov