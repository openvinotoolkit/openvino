
// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "debug_capabilities.h"
#include "node.h"

#ifdef CPU_DEBUG_CAPS

namespace ov {
namespace intel_cpu {

DebugLogEnabled::DebugLogEnabled(const char* file, const char* func, int line) {
    // check ENV
    const char* p_filters = std::getenv("OV_CPU_DEBUG_LOG");
    if (!p_filters) {
        enabled = false;
        return;
    }

    // extract file name from __FILE__
    std::string file_path(file);
    std::string file_name(file);
    auto last_sep = file_path.find_last_of('/');
    if (last_sep == std::string::npos)
        last_sep = file_path.find_last_of('\\');
    if (last_sep != std::string::npos)
        file_name = file_path.substr(last_sep + 1);

    std::string file_name_with_line = file_name + ":" + std::to_string(line);
    tag = file_name_with_line + " " + func + "()";
    // check each filter patten:
    bool filter_match_action;
    if (p_filters[0] == '-') {
        p_filters++;
        filter_match_action = false;
    } else {
        filter_match_action = true;
    }

    bool match = false;
    const char* p0 = p_filters;
    const char* p1;
    while (*p0 != 0) {
        p1 = p0;
        while (*p1 != ';' && *p1 != 0)
            ++p1;
        std::string patten(p0, p1 - p0);
        if (patten == file_name || patten == func || patten == tag || patten == file_name_with_line) {
            match = true;
            break;
        }
        p0 = p1;
        if (*p0 == ';')
            ++p0;
    }

    if (match)
        enabled = filter_match_action;
    else
        enabled = !filter_match_action;
}

void DebugLogEnabled::break_at(const std::string & log) {
    static const char* p_brk = std::getenv("OV_CPU_DEBUG_LOG_BRK");
    if (p_brk && log.find(p_brk) != std::string::npos) {
        std::cout << "[ DEBUG ] " << " Debug log breakpoint hit" << std::endl;
#if defined(_MSC_VER)
        __asm int 3;
#else
        asm("int3");
#endif
    }
}

std::ostream & operator<<(std::ostream & os, const dnnl::memory::desc& desc) {
    char sep = '(';
    os << "dims:";
    for (int i = 0; i < desc.data.ndims; i++) {
        os << sep << desc.data.dims[i];
        sep = ',';
    }
    os << ")";

    sep = '(';
    os << "strides:";
    for (int i = 0; i < desc.data.ndims; i++) {
        os << sep << desc.data.format_desc.blocking.strides[i];
        sep = ',';
    }
    os << ")";

    for (int i = 0; i < desc.data.format_desc.blocking.inner_nblks; i++) {
        os << desc.data.format_desc.blocking.inner_blks[i] << static_cast<char>('a' + desc.data.format_desc.blocking.inner_idxs[i]);
    }

    os << " " << dnnl_dt2str(desc.data.data_type);
    return os;
}

std::ostream & operator<<(std::ostream & os, const MemoryDesc& desc) {
    os << desc.getShape().toString()
       << " " << desc.getPrecision().name()
       << " " << desc.serializeFormat();
    return os;
}

std::ostream & operator<<(std::ostream & os, const NodeDesc& desc) {
    os << "    ImplementationType: " << impl_type_to_string(desc.getImplementationType()) << std::endl;
    for (auto & conf : desc.getConfig().inConfs) {
        os << "    inConfs: " << *conf.getMemDesc() << std::endl;
    }
    for (auto & conf : desc.getConfig().outConfs) {
        os << "    outConfs: " << *conf.getMemDesc() << std::endl;
    }
    return os;
}


}   // namespace intel_cpu
}   // namespace ov

#endif