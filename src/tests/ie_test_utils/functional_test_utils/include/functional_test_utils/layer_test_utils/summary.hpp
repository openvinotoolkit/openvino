// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <fstream>

#include "ngraph/ngraph.hpp"

#include "common_test_utils/test_constants.hpp"

namespace LayerTestsUtils {

class Summary;

class SummaryDestroyer {
private:
    Summary *p_instance;
public:
    ~SummaryDestroyer();

    void initialize(Summary *p);
};


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
private:
    static Summary *p_instance;
    static SummaryDestroyer destroyer;
    std::map<ngraph::NodeTypeInfo, PassRate> opsStats = {};
    std::string deviceName;
    bool isReported = false;
    static size_t saveReportTimeout;
    static bool extendReport;
    static bool extractBody;
    static bool saveReportWithUniqueName;
    static const char *outputFolder;
    std::vector<ngraph::OpSet> opsets;

    friend class SummaryDestroyer;

    std::string getOpVersion(const ngraph::NodeTypeInfo &type_info);

protected:
    Summary();

    ~Summary() = default;

public:
    void setDeviceName(std::string device) { deviceName = device; }

    std::map<std::string, PassRate> getOpStatisticFromReport();

    std::string getDeviceName() const { return deviceName; }

    std::map<ngraph::NodeTypeInfo, PassRate> getOPsStats() { return opsStats; }

    void updateOPsStats(const std::shared_ptr<ngraph::Function> &function, const PassRate::Statuses &status);
    void updateOPsImplStatus(const std::shared_ptr<ngraph::Function> &function, const bool implStatus);

    void updateOPsStats(const ngraph::NodeTypeInfo &op, const PassRate::Statuses &status);
    void updateOPsImplStatus(const ngraph::NodeTypeInfo &op, const bool implStatus);

    static Summary &getInstance();
    std::vector<ngraph::OpSet> getOpSets() {
        return opsets;
    }

    // #define IE_TEST_DEBUG

    #ifdef IE_TEST_DEBUG
    void saveDebugReport(const char* className, const char* opName, unsigned long passed, unsigned long failed,
                         unsigned long skipped, unsigned long crashed, unsigned long hanged);
    #endif  //IE_TEST_DEBUG

    void saveReport();

    static void setExtractBody(bool val) { extractBody = val; }
    static bool getExtractBody() { return extractBody; }

    static void setExtendReport(bool val) { extendReport = val; }
    static bool getExtendReport() { return extendReport; }

    static void setSaveReportWithUniqueName(bool val) { saveReportWithUniqueName = val; }
    static bool getSaveReportWithUniqueName() { return saveReportWithUniqueName; }

    static void setSaveReportTimeout(size_t val) { saveReportTimeout = val; }
    static size_t getSaveReportTimeout() { return saveReportTimeout; }

    static void setOutputFolder(const std::string &val) { outputFolder = val.c_str(); }
};

}  // namespace LayerTestsUtils
