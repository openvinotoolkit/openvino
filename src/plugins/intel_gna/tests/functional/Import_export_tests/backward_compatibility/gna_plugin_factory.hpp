#pragma once

#include "2dot7/gna_model_serial.hpp"
#include "2dot8/gna_model_serial.hpp"
#include "gna_plugin.hpp"

namespace ov {
namespace intel_gna {
namespace test {

enum kExportModelVersion {
    V2_7,
    V2_8,
    UNKNOWN
};

inline const char* ExportModelVersionToStr(kExportModelVersion ver) {
    const char* ver_str = "UNKNOWN";
    switch (ver) {
    case kExportModelVersion::V2_7:
        ver_str = "2.7";
        break;
    case kExportModelVersion::V2_8:
        ver_str = "2.8";
        break;
    case kExportModelVersion::UNKNOWN:
        ver_str = "UNKNOWN";
        break;
    }
    return ver_str;
}

class GNAPluginLegacyFactory {
public:
    static std::shared_ptr<GNAPlugin> CreatePluginLegacy(kExportModelVersion model_ver) {
        switch (model_ver)
        {
        case kExportModelVersion::V2_7:
            return std::shared_ptr<GNAPlugin>(new ov::intel_gna::header_2_dot_7::GNAPluginLegacy());
            break;
        case kExportModelVersion::V2_8:
            return std::shared_ptr<GNAPlugin>(new ov::intel_gna::header_2_dot_8::GNAPluginLegacy());
            break;
        default:
            break;
        }
    }
};

} // test
} // intel_gna
} // ov