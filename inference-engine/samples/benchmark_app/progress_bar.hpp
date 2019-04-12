// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <samples/console_progress.hpp>

/// @brief Responsible for progress bar handling within the benchmark_app
class ProgressBar {
public:
    ProgressBar(size_t totalNum, bool stream_output) {
        _bar.reset(new ConsoleProgress(totalNum, stream_output));
        _isFinished = true;
    }

    void addProgress(size_t num) {
        _isFinished = false;
        _bar->addProgress(num);
    }

    void finish() {
        _isFinished = true;
        _bar->finish();
        std::cout << std::endl;
    }

    void newBar(size_t totalNum) {
        if (_isFinished) {
            _bar.reset(new ConsoleProgress(totalNum));
        } else {
            throw std::logic_error("Can't create new bar. Current progress bar is still in progress");
        }
    }

private:
    std::unique_ptr<ConsoleProgress> _bar;
    bool _isFinished;
};