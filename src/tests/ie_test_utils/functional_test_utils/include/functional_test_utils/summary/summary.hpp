// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <fstream>

#include "openvino/openvino.hpp"

#include "common_test_utils/test_constants.hpp"
#include "common_test_utils/common_utils.hpp"

namespace ov {
namespace test {
namespace utils {

struct PassRate {
    enum Statuses {
        PASSED,
        FAILED,
        SKIPPED,
        CRASHED,
        HANGED
    };
    unsigned long passed = 0;
    unsigned long failed = 0;
    unsigned long skipped = 0;
    unsigned long crashed = 0;
    unsigned long hanged = 0;
    bool isImplemented = false;

    double rel_passed = 0;
    double rel_all = 0;

    PassRate() = default;

    PassRate(unsigned long p, unsigned long f, unsigned long s, unsigned long c, unsigned long h, double rel_p = 0, double rel_a = 0) {
        passed = p;
        failed = f;
        skipped = s;
        crashed = c;
        hanged = h;
        rel_passed = rel_p;
        rel_all = rel_a;
        if (!isImplemented && passed > 0) {
            isImplemented = true;
        }
    }

    void setImplementationStatus(bool implStatus) {
        isImplemented = implStatus;
    }

    float getPassrate() const {
        if (passed + failed + crashed + hanged == 0) {
            return 0.f;
        } else {
            return passed * 100.f / (passed + failed + skipped + crashed + hanged);
        }
    }

    double getRelPassrate() const {
        if (rel_all == 0) {
            return 100.f;
        } else {
            return rel_passed * 100.f / rel_all;
        }
    }
};

class Summary {
protected:
    std::string deviceName;
    const char* reportFilename;
    bool isReported = false;
    std::string ts = CommonTestUtils::GetTimestamp();

    static size_t saveReportTimeout;
    static bool isCrashReported;
    static bool isHangReported;
    static bool extendReport;
    static bool saveReportWithUniqueName;
    static const char *outputFolder;

    Summary() = default;

    virtual ~Summary() = default;

public:
    void setDeviceName(std::string device);

    std::string getDeviceName() const;


    // #define IE_TEST_DEBUG

#ifdef IE_TEST_DEBUG
    void saveDebugReport(const char* className, const char* opName, unsigned long passed, unsigned long failed,
                         unsigned long skipped, unsigned long crashed, unsigned long hanged);
#endif  //IE_TEST_DEBUG

    virtual void saveReport() {}

    void setReportFilename(const std::string& val);

    static void setExtendReport(bool val);
    static bool getExtendReport();

    static void setSaveReportWithUniqueName(bool val);
    static bool getSaveReportWithUniqueName();

    static void setSaveReportTimeout(size_t val);
    static size_t getSaveReportTimeout();

    static void setOutputFolder(const std::string &val);
};

}  // namespace utils
}  // namespace test
}  // namespace ov
