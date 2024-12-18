#pragma once

#include <oneapi/dnnl/dnnl.hpp>
#include <vector>

struct MemoryFormatFilter {
    std::vector<dnnl::memory::format_tag> input;
    std::vector<dnnl::memory::format_tag> output;

    bool empty() const {
        return input.empty() && output.empty();
    }
};
