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

#include "timer.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace ngraph
{
    namespace test
    {
        Timer::Timer(std::string timer_name)
            : m_name{std::move(timer_name)}
        {
            m_time_blocks.emplace_back("total");
            m_time_blocks.front().start();
        }

        void Timer::finish()
        {
            // end the total time measurement for this timer
            m_time_blocks.front().end();

            std::stringstream total_duration;
            total_duration << std::setw(10) << std::fixed << m_time_blocks.front().elapsed_time()
                           << " [s]";

            std::vector<std::string> output;
            if (m_time_blocks.size() == 1)
            {
                const std::string timer_name =
                    "  Total execution time for timer \"" + m_name + "\": " + total_duration.str();
                const std::string separator_line(timer_name.length() + 2, '=');
                output.push_back(separator_line);
                output.push_back(timer_name);
                output.push_back(separator_line);
            }
            else
            {
                const std::string timer_name = "  Execution times for timer: " + m_name + "  ";
                const std::string separator_line(timer_name.length(), '=');
                output.push_back(separator_line);
                output.push_back(timer_name);
                output.push_back(separator_line);

                const auto max_len_block =
                    std::max_element(std::next(m_time_blocks.begin()),
                                     m_time_blocks.end(),
                                     [](const TimeBlock& tb1, const TimeBlock& tb2) {
                                         return tb1.name().length() < tb2.name().length();
                                     });
                const auto longest_name = max_len_block->name().length();

                const auto block_to_str = [&longest_name](const TimeBlock& block) {
                    std::stringstream stringifier;
                    const std::string padding(longest_name - block.name().length() + 2, ' ');
                    stringifier << "  " << block.name() << padding << std::setw(10)
                                << std::fixed << block.elapsed_time() << " [s]";
                    return stringifier.str();
                };

                std::transform(std::next(m_time_blocks.cbegin()),
                               m_time_blocks.cend(),
                               std::back_inserter(output),
                               block_to_str);

                output.push_back(separator_line);
                output.push_back("  Total time: " + total_duration.str());
                output.push_back(separator_line);
            }

            for (const auto& line : output)
            {
                std::cout << line << std::endl;
            }
        }

        Timer::~Timer() {}

        ScopedTimer Timer::measure_scope_time(std::string name)
        {
            if (name.empty())
            {
                name = "unnamed scope";
            }

            m_time_blocks.emplace_back(std::move(name));

            return ScopedTimer{m_time_blocks.back()};
        }

        TimeBlock::TimeBlock(std::string name)
            : m_name{std::move(name)}
        {
        }
        void TimeBlock::start() { m_start = std::chrono::system_clock::now(); }
        void TimeBlock::end() { m_end = std::chrono::system_clock::now(); }
        double TimeBlock::elapsed_time() const
        {
            // TODO: stop the timer if end() hasn't been called
            const std::chrono::duration<double> elapsed_time = m_end - m_start;
            return elapsed_time.count();
        }
        const std::string& TimeBlock::name() const { return m_name; }

        ScopedTimer::ScopedTimer(TimeBlock& block)
            : m_time_block{block}
        {
            m_time_block.start();
        }

        ScopedTimer::~ScopedTimer() { m_time_block.end(); }
    } // namespace test
} // namespace ngraph