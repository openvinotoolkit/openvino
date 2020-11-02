/*
// Copyright (c) 2016 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#pragma once
#include <chrono>
#include <sstream>
#include <iomanip>
#include "api/memory.hpp"

#define DUMP_DIRECTORY "./"

namespace instrumentation {

    template<class Rep, class Period>
    std::string to_string(const std::chrono::duration<Rep, Period> val) {
        namespace  ch = std::chrono;
        const ch::microseconds us(1);
        const ch::milliseconds ms(1);
        const ch::seconds s(1);
        const std::chrono::duration<Rep, Period> abs_val(std::abs(val.count()));

        std::ostringstream os;
        os << std::setprecision(3) << std::fixed;
        if (abs_val > s)       os << ch::duration_cast<ch::duration<double, ch::seconds::period>>(val).count() << " s";
        else if (abs_val > ms) os << ch::duration_cast<ch::duration<double, ch::milliseconds::period>>(val).count() << " ms";
        else if (abs_val > us) os << ch::duration_cast<ch::duration<double, ch::microseconds::period>>(val).count() << " us";
        else               os << ch::duration_cast<ch::nanoseconds>(val).count() << " ns";
        return os.str();
    }

    struct logger
    {
        static void log_memory_to_file(const cldnn::memory&, std::string prefix = "", bool single_batch = false, cldnn::tensor::value_type batch_id = 0, cldnn::tensor::value_type feature_id = 0);
        static void log_weights_to_file(const cldnn::memory&, std::string prefix = "");
    private:
        static const std::string dump_dir;
    };
}
