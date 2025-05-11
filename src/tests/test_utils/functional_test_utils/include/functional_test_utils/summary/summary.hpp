// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <fstream>
#include <map>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/test_constants.hpp"

namespace ov {
namespace test {
namespace utils {

extern bool is_print_rel_influence_coef;

struct PassRate {
    enum Statuses { PASSED, FAILED, SKIPPED, CRASHED, HANGED };
    unsigned long passed = 0;
    unsigned long failed = 0;
    unsigned long skipped = 0;
    unsigned long crashed = 0;
    unsigned long hanged = 0;
    bool isImplemented = false;

    double rel_passed = 0;
    double rel_all = 0;

    bool isCrashReported = false;
    bool isHangReported = false;

    PassRate() = default;

    PassRate(unsigned long p,
             unsigned long f,
             unsigned long s,
             unsigned long c,
             unsigned long h,
             double rel_p = 0,
             double rel_a = 0);

    void setImplementationStatus(bool implStatus);

    float getPassrate() const;

    double getRelPassrate() const;
};

class Summary {
protected:
    std::string deviceName;
    const char* reportFilename;
    bool isReported = false;
    std::string ts = ov::test::utils::GetTimestamp();

    static size_t saveReportTimeout;
    static bool extendReport;
    static bool saveReportWithUniqueName;
    static const char* outputFolder;

    Summary() = default;

    virtual ~Summary() = default;

public:
    void setDeviceName(std::string device);

    std::string getDeviceName() const;

    virtual void saveReport() {}

    void setReportFilename(const std::string& val);

    static void setExtendReport(bool val);
    static bool getExtendReport();

    static void setSaveReportWithUniqueName(bool val);
    static bool getSaveReportWithUniqueName();

    static void setSaveReportTimeout(size_t val);
    static size_t getSaveReportTimeout();

    static void setOutputFolder(const std::string& val);
};

}  // namespace utils
}  // namespace test
}  // namespace ov
