// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_common.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>

#include "frontend_manager/frontend_manager.hpp"
#include "frontend_manager/parameters.hpp"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    define _WINSOCKAPI_

#    include <windows.h>

#    include "Psapi.h"
#endif

namespace ov {
namespace test {

inline size_t getVmSizeInKB() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    pmc.cb = sizeof(PROCESS_MEMORY_COUNTERS);
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, pmc.cb);
    return pmc.WorkingSetSize;
#else
    auto parseLine = [](char* line) {
        // This assumes that a digit will be found and the line ends in " Kb".
        size_t i = strlen(line);
        const char* p = line;
        while (*p < '0' || *p > '9')
            p++;
        line[i - 3] = '\0';
        i = (size_t)atoi(p);
        return i;
    };

    FILE* file = fopen("/proc/self/status", "r");
    size_t result = 0;
    if (file != nullptr) {
        char line[128];

        while (fgets(line, 128, file) != NULL) {
            if (strncmp(line, "VmSize:", 7) == 0) {
                result = parseLine(line);
                break;
            }
        }
        fclose(file);
    }
    return result;
#endif
}

TestsCommon::~TestsCommon() = default;

TestsCommon::TestsCommon() {
    auto memsize = getVmSizeInKB();
    if (memsize != 0) {
        std::cout << "\nMEM_USAGE=" << memsize << "KB\n";
    }
}

std::string TestsCommon::GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto epoch = now.time_since_epoch();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(epoch);
    return std::to_string(ns.count());
}

std::string TestsCommon::GetTestName() const {
    std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    std::replace_if(
        test_name.begin(),
        test_name.end(),
        [](char c) {
            return !std::isalnum(c);
        },
        '_');
    return test_name;
}

namespace {
ov::Extensions get_extensions_map(const std::vector<InferenceEngine::IExtensionPtr>& exts) {
    ov::Extensions extensions;
    for (const auto& ext : exts) {
        for (const auto& item : ext->getOpSets()) {
            if (extensions.count(item.first)) {
                IE_THROW() << "Extension with " << item.first << " name already exists";
            }
            extensions[item.first] = item.second;
        }
    }
    return extensions;
}

}  // namespace

std::shared_ptr<ov::Function> TestsCommon::read(const std::string& model,
                                                const std::string& weights,
                                                const std::vector<InferenceEngine::IExtensionPtr>& exts) {
    ngraph::frontend::FrontEndManager manager;
    ngraph::frontend::FrontEnd::Ptr FE;
    ngraph::frontend::InputModel::Ptr inputModel;

    ov::VariantVector params{ov::make_variant(model)};
    if (!weights.empty())
        params.emplace_back(ov::make_variant(weights));
    if (!exts.empty()) {
        params.emplace_back(ov::make_variant(get_extensions_map(exts)));
    }
    FE = manager.load_by_model(params);

    if (!FE)
        IE_THROW() << "Cannot load frontend!";
    inputModel = FE->load(params);
    if (!inputModel)
        IE_THROW() << "Cannot load model!";

    auto result = FE->convert(inputModel);
    return result;
}

}  // namespace test
}  // namespace ov
