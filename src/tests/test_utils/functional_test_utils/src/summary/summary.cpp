// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/summary/summary.hpp"

namespace ov {
namespace test {
namespace utils {

bool is_print_rel_influence_coef = false;

PassRate::PassRate(unsigned long p,
                   unsigned long f,
                   unsigned long s,
                   unsigned long c,
                   unsigned long h,
                   double rel_p,
                   double rel_a) {
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

void PassRate::setImplementationStatus(bool implStatus) {
    isImplemented = implStatus;
}

float PassRate::getPassrate() const {
    if (passed + failed + crashed + hanged == 0) {
        return 0.f;
    } else {
        return passed * 100.f / (passed + failed + skipped + crashed + hanged);
    }
}

double PassRate::getRelPassrate() const {
    if (rel_all == 0) {
        return 100.f;
    } else {
        return rel_passed * 100.f / rel_all;
    }
}

bool Summary::extendReport = false;
bool Summary::saveReportWithUniqueName = false;
size_t Summary::saveReportTimeout = 0;
const char* Summary::outputFolder = ".";

void Summary::setDeviceName(std::string device) {
    deviceName = device;
}

std::string Summary::getDeviceName() const {
    return deviceName;
}

void Summary::setReportFilename(const std::string& val) {
    reportFilename = val.c_str();
}

void Summary::setExtendReport(bool val) {
    extendReport = val;
}
bool Summary::getExtendReport() {
    return extendReport;
}

void Summary::setSaveReportWithUniqueName(bool val) {
    saveReportWithUniqueName = val;
}
bool Summary::getSaveReportWithUniqueName() {
    return saveReportWithUniqueName;
}

void Summary::setSaveReportTimeout(size_t val) {
    saveReportTimeout = val;
}
size_t Summary::getSaveReportTimeout() {
    return saveReportTimeout;
}

void Summary::setOutputFolder(const std::string& val) {
    outputFolder = val.c_str();
}

}  // namespace utils
}  // namespace test
}  // namespace ov
