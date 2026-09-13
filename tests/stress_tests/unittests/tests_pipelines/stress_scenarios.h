// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

void stress_load_unload(const std::string& model, const std::string& device, int iterations,
                        int threads, const std::string& compilation_config = "");
void stress_parallel_infer(const std::string& model, const std::string& device, int iterations,
                           int threads, const std::string& compilation_config = "");
void stress_concurrent_load_infer(const std::string& model, const std::string& device,
                                  int iterations, int threads, const std::string& compilation_config = "");
void stress_import_export(const std::string& model, const std::string& device, int iterations,
                          int threads, const std::string& compilation_config = "");
void stress_mid_flight_cancel(const std::string& model, const std::string& device, int iterations,
                              int threads, const std::string& compilation_config = "");
void stress_memory_pressure(const std::string& model, const std::string& device, int iterations,
                            int threads, const std::string& compilation_config = "");
void stress_destroy_compiled_model(const std::string& model, const std::string& device,
                                   int iterations, int threads, const std::string& compilation_config = "");
void stress_multiple_cores(const std::string& model, const std::string& device, int iterations,
                           int threads, const std::string& compilation_config = "");
void stress_heterogeneous_concurrent_infer(const std::string& model_heavy,
                                           const std::string& model_light,
                                           const std::string& device,
                                           int iterations,
                                           int threads,
                                           const std::string& compilation_config_heavy = "",
                                           const std::string& compilation_config_light = "");
void run_stress_scenario(const std::string& scenario, const std::string& model,
                         const std::string& device, int iterations, int threads,
                         const std::string& model2 = "",
                         const std::string& compilation_config = "",
                         const std::string& compilation_config2 = "");