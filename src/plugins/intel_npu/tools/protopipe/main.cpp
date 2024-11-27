//
// Copyright (C) 2023-2024 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#include <future>
#include <iostream>
#include <regex>

#include <gflags/gflags.h>

#include "parser/parser.hpp"
#include "scenario/scenario_graph.hpp"
#include "simulation/performance_mode.hpp"
#include "simulation/reference_mode.hpp"
#include "simulation/validation_mode.hpp"

#include "utils/error.hpp"
#include "utils/logger.hpp"

static constexpr char help_message[] = "Optional. Print the usage message.";
static constexpr char cfg_message[] = "Path to the configuration file.";
static constexpr char device_message[] =
        "Optional. Device name. If specified overwrites device specified in config file.";
static constexpr char pipeline_message[] = "Optional. Enable pipelined execution.";
static constexpr char drop_message[] = "Optional. Drop frames if they come earlier than pipeline is completed.";
static constexpr char mode_message[] = "Optional. Simulation mode: performance (default), reference, validation.";
static constexpr char niter_message[] = "Optional. Number of iterations. If specified overwrites termination criterion"
                                        " for all scenarios in configuration file.";
static constexpr char exec_time_message[] = "Optional. Time in seconds. If specified overwrites termination criterion"
                                            " for all scenarios in configuration file.";
static constexpr char inference_only_message[] =
        "Optional. Run only inference execution for every model excluding i/o data transfer."
        " Applicable only for \"performance\" mode. (default: true).";

static constexpr char exec_filter_msg[] = "Optional. Run the scenarios that match provided string pattern.";

DEFINE_bool(h, false, help_message);
DEFINE_string(cfg, "", cfg_message);
DEFINE_string(d, "", device_message);
DEFINE_bool(pipeline, false, pipeline_message);
DEFINE_bool(drop_frames, false, drop_message);
DEFINE_string(mode, "performance", mode_message);
DEFINE_uint64(niter, 0, niter_message);
DEFINE_uint64(t, 0, exec_time_message);
DEFINE_bool(inference_only, true, inference_only_message);
DEFINE_string(exec_filter, ".*", exec_filter_msg);

static void showUsage() {
    std::cout << "protopipe [OPTIONS]" << std::endl;
    std::cout << std::endl;
    std::cout << " Common options:            " << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -cfg <value>            " << cfg_message << std::endl;
    std::cout << "    -pipeline               " << pipeline_message << std::endl;
    std::cout << "    -drop_frames            " << drop_message << std::endl;
    std::cout << "    -d <value>              " << device_message << std::endl;
    std::cout << "    -mode <value>           " << mode_message << std::endl;
    std::cout << "    -niter <value>          " << niter_message << std::endl;
    std::cout << "    -t <value>              " << exec_time_message << std::endl;
    std::cout << "    -inference_only         " << inference_only_message << std::endl;
    std::cout << "    -exec_filter            " << exec_filter_msg << std::endl;
    std::cout << std::endl;
}

bool parseCommandLine(int* argc, char*** argv) {
    gflags::ParseCommandLineNonHelpFlags(argc, argv, true);

    if (FLAGS_h) {
        showUsage();
        return false;
    }

    if (FLAGS_cfg.empty()) {
        throw std::invalid_argument("Path to config file is required");
    }

    std::cout << "Parameters:" << std::endl;
    std::cout << "    Config file:             " << FLAGS_cfg << std::endl;
    std::cout << "    Pipelining is enabled:   " << std::boolalpha << FLAGS_pipeline << std::endl;
    std::cout << "    Simulation mode:         " << FLAGS_mode << std::endl;
    std::cout << "    Inference only:          " << std::boolalpha << FLAGS_inference_only << std::endl;
    std::cout << "    Device:                  " << FLAGS_d << std::endl;
    return true;
}

static ICompiled::Ptr compileSimulation(Simulation::Ptr simulation, const bool pipelined, const bool drop_frames) {
    LOG_INFO() << "Compile simulation" << std::endl;
    if (pipelined) {
        return simulation->compilePipelined(drop_frames);
    }
    return simulation->compileSync(drop_frames);
};

class ThreadRunner {
public:
    using F = std::function<void()>;
    void add(F&& func) {
        m_funcs.push_back(std::move(func));
    }
    void run();

private:
    std::vector<F> m_funcs;
};

void ThreadRunner::run() {
    std::vector<std::future<void>> futures;
    futures.reserve(m_funcs.size());
    for (auto&& func : m_funcs) {
        futures.push_back(std::async(std::launch::async, std::move(func)));
    }
    for (auto& future : futures) {
        future.get();
    };
};

class Task {
public:
    Task(ICompiled::Ptr&& compiled, std::string&& name, ITermCriterion::Ptr&& criterion);

    void operator()();
    const Result& result() const;
    const std::string& name() const;

private:
    ICompiled::Ptr m_compiled;
    std::string m_name;
    ITermCriterion::Ptr m_criterion;

    Result m_result;
};

Task::Task(ICompiled::Ptr&& compiled, std::string&& name, ITermCriterion::Ptr&& criterion)
        : m_compiled(std::move(compiled)), m_name(std::move(name)), m_criterion(std::move(criterion)) {
}

void Task::operator()() {
    try {
        m_result = m_compiled->run(m_criterion);
    } catch (const std::exception& e) {
        m_result = Error{e.what()};
    }
}

const Result& Task::result() const {
    return m_result;
}

const std::string& Task::name() const {
    return m_name;
}

static Simulation::Ptr createSimulation(const std::string& mode, StreamDesc&& stream, const bool inference_only,
                                        const Config& config) {
    Simulation::Ptr simulation;
    // NB: Common parameters for all simulations
    Simulation::Config cfg{stream.name, stream.frames_interval_in_us, config.disable_high_resolution_timer,
                           std::move(stream.graph), std::move(stream.infer_params_map)};
    if (mode == "performance") {
        PerformanceSimulation::Options opts{config.initializer, std::move(stream.initializers_map),
                                            std::move(stream.input_data_map), inference_only,
                                            std::move(stream.target_latency)};
        simulation = std::make_shared<PerformanceSimulation>(std::move(cfg), std::move(opts));
    } else if (mode == "reference") {
        CalcRefSimulation::Options opts{config.initializer, std::move(stream.initializers_map),
                                        std::move(stream.input_data_map), std::move(stream.output_data_map)};
        simulation = std::make_shared<CalcRefSimulation>(std::move(cfg), std::move(opts));
    } else if (mode == "validation") {
        ValSimulation::Options opts{config.metric, std::move(stream.metrics_map), std::move(stream.input_data_map),
                                    std::move(stream.output_data_map), std::move(stream.per_iter_outputs_path)};
        simulation = std::make_shared<ValSimulation>(std::move(cfg), std::move(opts));
    } else {
        throw std::logic_error("Unsupported simulation mode: " + mode);
    }
    ASSERT(simulation);
    return simulation;
}

int main(int argc, char* argv[]) {
    // NB: Intentionally wrapped into try-catch to display exceptions occur on windows.
    try {
        if (!parseCommandLine(&argc, &argv)) {
            return 0;
        }
        ReplaceBy replace_by{FLAGS_d};

        auto parser = std::make_shared<ScenarioParser>(FLAGS_cfg);

        LOG_INFO() << "Parse scenarios from " << FLAGS_cfg << " config file" << std::endl;
        auto config = parser->parseScenarios(replace_by);
        LOG_INFO() << "Found " << config.scenarios.size() << " scenario(s)" << std::endl;

        // NB: Overwrite termination criteria for all scenarios if specified via CLI
        ITermCriterion::Ptr global_criterion;
        if (FLAGS_niter != 0u) {
            LOG_INFO() << "Termination criterion of " << FLAGS_niter << " iteration(s) will be used for all scenarios"
                       << std::endl;
            global_criterion = std::make_shared<Iterations>(FLAGS_niter);
        }
        if (FLAGS_t != 0u) {
            if (global_criterion) {
                // TODO: In fact, it make sense to have them both enabled.
                THROW_ERROR("-niter and -t options can't be specified together!");
            }
            LOG_INFO() << "Termination criterion of " << FLAGS_t << " second(s) will be used for all scenarios"
                       << std::endl;
            // NB: TimeOut accepts microseconds
            global_criterion = std::make_shared<TimeOut>(FLAGS_t * 1'000'000);
        }

        std::regex filter_regex{FLAGS_exec_filter};
        bool any_scenario_failed = false;
        for (auto&& scenario : config.scenarios) {
            // NB: Skip the scenarios that don't match provided filter pattern
            if (!std::regex_match(scenario.name, filter_regex)) {
                LOG_INFO() << "Skip the scenario " << scenario.name << " as it doesn't match the -exec_filter=\""
                           << FLAGS_exec_filter << "\" pattern" << std::endl;
                continue;
            }
            LOG_INFO() << "Start processing " << scenario.name << std::endl;

            ThreadRunner runner;
            std::vector<Task> tasks;
            tasks.reserve(scenario.streams.size());
            for (auto&& stream : scenario.streams) {
                auto criterion = stream.criterion;
                auto stream_name = stream.name;
                if (global_criterion) {
                    if (criterion) {
                        LOG_INFO() << "Stream: " << stream_name
                                   << " termination criterion is overwritten by CLI parameter" << std::endl;
                    }
                    criterion = global_criterion->clone();
                }
                auto simulation = createSimulation(FLAGS_mode, std::move(stream), FLAGS_inference_only, config);
                auto compiled = compileSimulation(simulation, FLAGS_pipeline, FLAGS_drop_frames);
                tasks.emplace_back(std::move(compiled), std::move(stream_name), std::move(criterion));
                runner.add(std::ref(tasks.back()));
            }

            LOG_INFO() << "Run " << tasks.size() << " stream(s) asynchronously" << std::endl;
            runner.run();
            LOG_INFO() << "Execution has finished" << std::endl;

            for (const auto& task : tasks) {
                if (!task.result()) {
                    // NB: Scenario failed if any of the streams failed
                    any_scenario_failed = true;
                }
                std::cout << "stream " << task.name() << ": " << task.result().str() << std::endl;
            }
            std::cout << "\n";
        }
        if (any_scenario_failed) {
            return EXIT_FAILURE;
        }
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
        throw;
    } catch (...) {
        std::cout << "Unknown error" << std::endl;
        throw;
    }
    return 0;
}
