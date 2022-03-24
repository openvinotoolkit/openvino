// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <fstream>

#include "ngraph/ngraph.hpp"

#include "common_test_utils/test_constants.hpp"

namespace LayerTestsUtils {

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

    PassRate() = default;

    PassRate(unsigned long p, unsigned long f, unsigned long s, unsigned long c, unsigned long h) {
        passed = p;
        failed = f;
        skipped = s;
        crashed = c;
        hanged = h;
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
};

class Summary {
protected:
    std::string deviceName;
    bool isReported = false;
    static size_t saveReportTimeout;
    static bool extendReport;
    static bool saveReportWithUniqueName;
    static const char *outputFolder;

    Summary() = default;
    virtual ~Summary() = default;

public:
    void setDeviceName(std::string device) { deviceName = device; }

    virtual std::map<std::string, PassRate> getStatisticFromReport() { return std::map<std::string, PassRate>(); }

    std::string getDeviceName() const { return deviceName; }


    // #define IE_TEST_DEBUG

    #ifdef IE_TEST_DEBUG
    void saveDebugReport(const char* className, const char* opName, unsigned long passed, unsigned long failed,
                         unsigned long skipped, unsigned long crashed, unsigned long hanged);
    #endif  //IE_TEST_DEBUG

    virtual void saveReport() {}

    static void setExtendReport(bool val) { extendReport = val; }
    static bool getExtendReport() { return extendReport; }

    static void setSaveReportWithUniqueName(bool val) { saveReportWithUniqueName = val; }
    static bool getSaveReportWithUniqueName() { return saveReportWithUniqueName; }

    static void setSaveReportTimeout(size_t val) { saveReportTimeout = val; }
    static size_t getSaveReportTimeout() { return saveReportTimeout; }

    static void setOutputFolder(const std::string &val) { outputFolder = val.c_str(); }
};

}  // namespace LayerTestsUtils
