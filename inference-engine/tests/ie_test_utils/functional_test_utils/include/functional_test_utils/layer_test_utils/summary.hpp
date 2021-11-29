// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <fstream>

#include <pugixml.hpp>

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
        CRASHED
    };
    unsigned long passed = 0;
    unsigned long failed = 0;
    unsigned long skipped = 0;
    unsigned long crashed = 0;

    PassRate() = default;

    PassRate(unsigned long p, unsigned long f, unsigned long s, unsigned long c) {
        passed = p;
        failed = f;
        skipped = s;
        crashed = c;
    }

    float getPassrate() const {
        if (passed + failed + crashed == 0) {
            return 0.f;
        } else {
            return passed * 100.f / (passed + failed + skipped + crashed);
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

    void updateOPsStats(const ngraph::NodeTypeInfo &op, const PassRate::Statuses &status);

    static Summary &getInstance();

    void saveReport();

    static void setExtendReport(bool val) { extendReport = val; }

    static bool getExtendReport() { return extendReport; }

    static void setSaveReportWithUniqueName(bool val) { saveReportWithUniqueName = val; }

    static bool getSaveReportWithUniqueName() { return saveReportWithUniqueName; }

    static void setSaveReportTimeout(size_t val) { saveReportTimeout = val; }

    static size_t getSaveReportTimeout() { return saveReportTimeout; }

    static void setOutputFolder(const std::string &val) { outputFolder = val.c_str(); }
};

}  // namespace LayerTestsUtils