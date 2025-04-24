// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <vector>
#include <functional>

#include "../tests_utils.h"

enum ManagerStatus {
    NOT_STARTED = -2,
    NOT_FINISHED = -1,
    FINISHED_SUCCESSFULLY = 0,
    FINISHED_UNEXPECTEDLY
};

template<typename Type>
using Task = std::pair<ManagerStatus, std::function<Type()>>;

template<typename Type>
class TaskManager {
public:
    std::vector<Task<Type>> tasks;
    std::vector<Type> tasks_results;

    TaskManager() {}

    TaskManager(const std::initializer_list<std::function<Type()>> &tasks_list) {
        tasks.reserve(tasks_list.size());
        for (const auto &task : tasks_list)
            add_task(task);
    }

    void add_task(const std::function<Type()> &task) {
        auto _task = Task<Type>(ManagerStatus::NOT_STARTED, task);
        tasks.push_back(_task);
    }

    void run_sequentially() {
        // TODO: make it asynchronous
        tasks_results.reserve(tasks.size());
        for (auto task : tasks) {
            task.first = ManagerStatus::NOT_FINISHED;
            tasks_results.push_back(task.second());
        }
    }

    void run_parallel_n_wait() {
        run_parallel();
        wait_all();
    }

    void wait_all() {
        size_t numtasks = tasks.size();
        for (size_t i = 0; i < numtasks; i++)
            if (tasks[i].first == ManagerStatus::NOT_FINISHED)
                wait_task(i);
    }

    std::vector<ManagerStatus> get_all_statuses() {
        std::vector<ManagerStatus> statuses;

        size_t numtasks = tasks.size();
        for (size_t i = 0; i < numtasks; i++)
            statuses.push_back(get_task_status(i));
        return statuses;
    }

    std::vector<TestResult> get_all_results() {
        return tasks_results;
    }

    TestResult get_task_result(int task_index) {
        if (tasks_results.empty() ||
            tasks_results.size() < task_index ||
            task_index < 0)
            throw std::out_of_range("Task index " + std::to_string(task_index) + " out of number of tasks");

        return tasks_results[task_index];
    }

    virtual void run_parallel() = 0;

    virtual void wait_task(int task_index) = 0; // TODO: implement for run_sequentially

    virtual ManagerStatus get_task_status(int task_index) = 0;

};
