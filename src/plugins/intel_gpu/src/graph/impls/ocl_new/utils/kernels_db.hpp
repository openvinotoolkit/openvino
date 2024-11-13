// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <vector>
#include <cctype>
#include <string>

namespace ov {
namespace intel_gpu {
namespace ocl {

using HeaderID = std::string;
using KernelTemplateID = std::string;
using Code = std::string;
using KernelTemplateDesc = std::pair<std::vector<Code>, std::vector<HeaderID>>;

struct CaseInsensitiveComparator {
    bool operator()(const KernelTemplateID& lhs, const KernelTemplateID& rhs) const {
        return std::lexicographical_compare(lhs.begin(),
                                            lhs.end(),
                                            rhs.begin(),
                                            rhs.end(),
                                            [](const char& a, const char& b) { return tolower(a) < tolower(b); });
    }
};

using HeadersMap = std::map<HeaderID, Code>;
using KernelsMap = std::map<KernelTemplateID, KernelTemplateDesc, CaseInsensitiveComparator>;

struct KernelsDB {
    KernelsDB();

    const KernelTemplateDesc& get_template(const KernelTemplateID& id) const;
    const HeadersMap& get_headers() const { return m_headers; }

private:
    KernelsMap m_kernels;
    HeadersMap m_headers;
};

}  // namespace ocl
}  // namespace intel_gpu
}  // namespace ov
