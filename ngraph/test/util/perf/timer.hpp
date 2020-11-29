//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace ngraph
{
    namespace test
    {
        using time_point_t = std::chrono::time_point<std::chrono::system_clock>;

        struct TimeBlock
        {
            TimeBlock(std::string name);
            void start();
            void end();
            double elapsed_time() const;
            const std::string& name() const;

        private:
            const std::string m_name;
            time_point_t m_start;
            time_point_t m_end;
        };

        struct ScopedTimer
        {
            explicit ScopedTimer(TimeBlock& block);
            ~ScopedTimer();
            
            ScopedTimer() = delete;
            ScopedTimer(const ScopedTimer&) = delete;
            ScopedTimer& operator=(const ScopedTimer&) = delete;

            ScopedTimer(ScopedTimer&& other) = default;
            ScopedTimer& operator=(ScopedTimer&&) = default;

        private:
            TimeBlock& m_time_block;
        };

        class Timer
        {
        public:
            Timer(std::string timer_name);
            ~Timer();

            void finish();

            ScopedTimer measure_scope_time(std::string name);

        private:
            const std::string m_name;
            std::vector<TimeBlock> m_time_blocks;
        };
    } // namespace test
} // namespace ngraph