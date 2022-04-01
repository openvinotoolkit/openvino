// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/summary/summary.hpp"

using namespace ov::test::utils;

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

void Summary::setExtendReport(bool val) { extendReport = val; }
bool Summary::getExtendReport() { return extendReport; }

void Summary::setSaveReportWithUniqueName(bool val) { saveReportWithUniqueName = val; }
bool Summary::getSaveReportWithUniqueName() { return saveReportWithUniqueName; }

void Summary::setSaveReportTimeout(size_t val) { saveReportTimeout = val; }
size_t Summary::getSaveReportTimeout() { return saveReportTimeout; }

void Summary::setOutputFolder(const std::string &val) { outputFolder = val.c_str(); }