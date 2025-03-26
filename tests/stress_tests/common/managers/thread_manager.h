// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "task_manager.h"

#include <future>

template <typename Type>
class ThreadManager : public TaskManager<Type> {
public:
    using TaskManager<Type>::tasks;
    using TaskManager<Type>::tasks_results;
    std::vector<std::future<TestResult>> threads;

    using TaskManager<Type>::TaskManager;

    void run_parallel() override {
        // TODO: implement run_task function according to wait_task
        int numtasks = tasks.size();
        threads.reserve(numtasks);
        tasks_results.reserve(numtasks);

        for (int i = 0; i < numtasks; i++)
            if (tasks[i].first == ManagerStatus::NOT_STARTED) {
                tasks[i].first = ManagerStatus::NOT_FINISHED;
                threads.push_back(std::async(std::launch::async, tasks[i].second));
            }
    }

    void wait_task(int task_index) override {
        if (threads.empty() ||
            static_cast<int>(threads.size()) < task_index ||
            task_index < 0)
            throw std::out_of_range("Task index " + std::to_string(task_index) + " out of number of tasks");

        try {
            tasks_results.push_back(threads[task_index].get());
            tasks[task_index].first = ManagerStatus::FINISHED_SUCCESSFULLY;
        } catch (std::exception &err) { // TODO: catch any exception
            std::exception_ptr p = std::current_exception();
            tasks[task_index].first = ManagerStatus::FINISHED_UNEXPECTEDLY;
            tasks_results.push_back(TestResult(TestStatus::TEST_FAILED, "Test finished unexpectedly: " + (std::string)err.what()));
        }
    }

    ManagerStatus get_task_status(int task_index) override {
        if (threads.empty() ||
            static_cast<int>(threads.size()) < task_index ||
            task_index < 0)
            throw std::out_of_range("Task index " + std::to_string(task_index) + " out of number of tasks");

        return tasks[task_index].first;
    }
};
