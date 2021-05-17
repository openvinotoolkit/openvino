// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "functional_test_utils/layer_test_utils/summary.hpp"
#include "common_test_utils/file_utils.hpp"

using namespace LayerTestsUtils;

#ifdef _WIN32
# define getpid _getpid
#endif

Summary *Summary::p_instance = nullptr;
bool Summary::extendReport = false;
bool Summary::saveReportWithUniqueName = false;
size_t Summary::saveReportTimeout = 0;
const char* Summary::outputFolder = ".";
SummaryDestroyer Summary::destroyer;

SummaryDestroyer::~SummaryDestroyer() {
    delete p_instance;
}

void SummaryDestroyer::initialize(Summary *p) {
    p_instance = p;
}

Summary::Summary() {
    opsets.push_back(ngraph::get_opset1());
    opsets.push_back(ngraph::get_opset2());
    opsets.push_back(ngraph::get_opset3());
    opsets.push_back(ngraph::get_opset4());
    opsets.push_back(ngraph::get_opset5());
    opsets.push_back(ngraph::get_opset6());
    opsets.push_back(ngraph::get_opset7());
}

Summary &Summary::getInstance() {
    if (!p_instance) {
        p_instance = new Summary();
        destroyer.initialize(p_instance);
    }
    return *p_instance;
}

void Summary::updateOPsStats(const ngraph::NodeTypeInfo &op, const PassRate::Statuses &status) {
    auto it = opsStats.find(op);
    if (it != opsStats.end()) {
        auto &passrate = it->second;
        switch (status) {
            case PassRate::PASSED:
                passrate.passed++;
                passrate.crashed--;
                break;
            case PassRate::FAILED:
                passrate.failed++;
                passrate.crashed--;
                break;
            case PassRate::SKIPPED:
                passrate.skipped++;
                break;
            case PassRate::CRASHED:
                passrate.crashed++;
                break;
        }
    } else {
        switch (status) {
            case PassRate::PASSED:
                opsStats[op] = PassRate(1, 0, 0, 0);
                break;
            case PassRate::FAILED:
                opsStats[op] = PassRate(0, 1, 0, 0);
                break;
            case PassRate::SKIPPED:
                opsStats[op] = PassRate(0, 0, 1, 0);
                break;
            case PassRate::CRASHED:
                opsStats[op] = PassRate(0, 0, 0, 1);
                break;
        }
    }
}

std::string Summary::getOpVersion(const ngraph::NodeTypeInfo &type_info) {
    for (size_t i = 0; i < opsets.size(); i++) {
        if (opsets[i].contains_type(type_info)) {
            return std::to_string(i+1);
        }
    }
    return "undefined";
}

std::map<std::string, PassRate> Summary::getOpStatisticFromReport() {
    pugi::xml_document doc;

    std::ifstream file;
    file.open(CommonTestUtils::REPORT_FILENAME);

    pugi::xml_node root;
    doc.load_file(CommonTestUtils::REPORT_FILENAME);
    root = doc.child("report");

    pugi::xml_node resultsNode = root.child("results");
    pugi::xml_node currentDeviceNode = resultsNode.child(deviceName.c_str());
    std::map<std::string, PassRate> oldOpsStat;
    for (auto &child : currentDeviceNode.children()) {
        std::string entry = child.name();
        auto p = std::stoi(child.attribute("passed").value());
        auto f = std::stoi(child.attribute("failed").value());
        auto s = std::stoi(child.attribute("skipped").value());
        auto c = std::stoi(child.attribute("crashed").value());
        PassRate obj(p, f, s, c);
        oldOpsStat.insert({entry, obj});
    }
    return oldOpsStat;
}

void Summary::updateOPsStats(const std::shared_ptr<ngraph::Function> &function, const PassRate::Statuses &status) {
    bool isFunctionalGraph = false;
    for (const auto &op : function->get_ordered_ops()) {
        if (!ngraph::is_type<ngraph::op::Parameter>(op) &&
            !ngraph::is_type<ngraph::op::Constant>(op) &&
            !ngraph::is_type<ngraph::op::Result>(op)) {
            isFunctionalGraph = true;
            break;
        }
    }

    for (const auto &op : function->get_ordered_ops()) {
        if ((ngraph::is_type<ngraph::op::Parameter>(op) ||
            ngraph::is_type<ngraph::op::Constant>(op) ||
            ngraph::is_type<ngraph::op::Result>(op)) && isFunctionalGraph) {
            continue;
        } else if (ngraph::is_type<ngraph::op::TensorIterator>(op)) {
            updateOPsStats(op->get_type_info(), status);
            auto ti = ngraph::as_type_ptr<ngraph::op::TensorIterator>(op);
            auto ti_body = ti->get_function();
            updateOPsStats(ti_body, status);
        } else if (ngraph::is_type<ngraph::op::v5::Loop>(op)) {
            updateOPsStats(op->get_type_info(), status);
            auto loop = ngraph::as_type_ptr<ngraph::op::v5::Loop>(op);
            auto loop_body = loop->get_function();
            updateOPsStats(loop_body, status);
        } else {
            updateOPsStats(op->get_type_info(), status);
        }
    }
}

void Summary::saveReport() {
    if (isReported) {
        return;
    }

    std::string filename = CommonTestUtils::REPORT_FILENAME;
    if (saveReportWithUniqueName) {
        auto processId = std::to_string(getpid());
        filename += "_" + processId + "_" + std::string(CommonTestUtils::GetTimestamp());
    }
    filename += CommonTestUtils::REPORT_EXTENSION;

    if (!CommonTestUtils::directoryExists(outputFolder)) {
        CommonTestUtils::createDirectoryRecursive(outputFolder);
    }

    std::string outputFilePath = outputFolder + std::string(CommonTestUtils::FileSeparator) + filename;

    std::set<ngraph::NodeTypeInfo> opsInfo;
    for (const auto &opset : opsets) {
        const auto &type_info_set = opset.get_type_info_set();
        opsInfo.insert(type_info_set.begin(), type_info_set.end());
    }

    auto &summary = Summary::getInstance();
    auto stats = summary.getOPsStats();

    pugi::xml_document doc;

    std::ifstream file;
    file.open(outputFilePath);

    time_t rawtime;
    struct tm *timeinfo;
    char timeNow[80];

    time(&rawtime);
    // cpplint require to use localtime_r instead which is not available in C++11
    timeinfo = localtime(&rawtime); // NOLINT

    strftime(timeNow, sizeof(timeNow), "%d-%m-%Y %H:%M:%S", timeinfo);

    pugi::xml_node root;
    if (file) {
        doc.load_file(outputFilePath.c_str());
        root = doc.child("report");
        //Ugly but shorter than to write predicate for find_atrribute() to update existing one
        root.remove_attribute("timestamp");
        root.append_attribute("timestamp").set_value(timeNow);

        root.remove_child("ops_list");
        root.child("results").remove_child(summary.deviceName.c_str());
    } else {
        root = doc.append_child("report");
        root.append_attribute("timestamp").set_value(timeNow);
        root.append_child("results");
    }

    pugi::xml_node opsNode = root.append_child("ops_list");
    for (const auto &op : opsInfo) {
        std::string name = std::string(op.name) + "-" + getOpVersion(op);
        pugi::xml_node entry = opsNode.append_child(name.c_str());
        (void) entry;
    }

    pugi::xml_node resultsNode = root.child("results");
    pugi::xml_node currentDeviceNode = resultsNode.append_child(summary.deviceName.c_str());
    std::unordered_set<std::string> opList;
    for (const auto &it : stats) {
        std::string name = std::string(it.first.name) + "-" + getOpVersion(it.first);
        opList.insert(name);
        pugi::xml_node entry = currentDeviceNode.append_child(name.c_str());
        entry.append_attribute("passed").set_value(it.second.passed);
        entry.append_attribute("failed").set_value(it.second.failed);
        entry.append_attribute("skipped").set_value(it.second.skipped);
        entry.append_attribute("crashed").set_value(it.second.crashed);
        entry.append_attribute("passrate").set_value(it.second.getPassrate());
    }

    if (extendReport && file) {
        auto opStataFromReport = summary.getOpStatisticFromReport();
        for (auto &item : opStataFromReport) {
            pugi::xml_node entry;
            if (opList.find(item.first) == opList.end()) {
                entry = currentDeviceNode.append_child(item.first.c_str());
                entry.append_attribute("passed").set_value(item.second.passed);
                entry.append_attribute("failed").set_value(item.second.failed);
                entry.append_attribute("skipped").set_value(item.second.skipped);
                entry.append_attribute("crashed").set_value(item.second.crashed);
                entry.append_attribute("passrate").set_value(item.second.getPassrate());
            } else {
                entry = currentDeviceNode.child(item.first.c_str());
                auto p = std::stoi(entry.attribute("passed").value()) + item.second.passed;
                auto f = std::stoi(entry.attribute("failed").value()) + item.second.failed;
                auto s = std::stoi(entry.attribute("skipped").value()) + item.second.skipped;
                auto c = std::stoi(entry.attribute("crashed").value()) + item.second.crashed;
                PassRate obj(p, f, s, c);

                entry.attribute("passed").set_value(obj.passed);
                entry.attribute("failed").set_value(obj.failed);
                entry.attribute("skipped").set_value(obj.skipped);
                entry.attribute("crashed").set_value(obj.crashed);
                entry.attribute("passrate").set_value(obj.getPassrate());
            }
        }
    }

    auto exitTime = std::chrono::system_clock::now() + std::chrono::seconds(saveReportTimeout);
    bool result = false;
    do {
        result = doc.save_file(outputFilePath.c_str());
    } while (!result && std::chrono::system_clock::now() < exitTime);

    if (!result) {
        std::string errMessage = "Failed to write report to " + outputFilePath;
        throw std::runtime_error(errMessage);
    } else {
        isReported = true;
    }
    file.close();
}
