// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <pugixml.hpp>

#include "functional_test_utils/layer_test_utils/summary.hpp"
#include "common_test_utils/file_utils.hpp"

using namespace LayerTestsUtils;

#ifdef _WIN32
# define getpid _getpid
#endif

Summary *Summary::p_instance = nullptr;
bool Summary::extendReport = false;
bool Summary::extractBody = false;
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
    opsets.push_back(ngraph::get_opset8());
    opsets.push_back(ngraph::get_opset9());
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
                if (!passrate.isImplemented) {
                    passrate.isImplemented = true;
                }
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
            case PassRate::HANGED:
                passrate.hanged++;
                passrate.crashed--;
                break;
        }
    } else {
        switch (status) {
            case PassRate::PASSED:
                opsStats[op] = PassRate(1, 0, 0, 0, 0);
                break;
            case PassRate::FAILED:
                opsStats[op] = PassRate(0, 1, 0, 0, 0);
                break;
            case PassRate::SKIPPED:
                opsStats[op] = PassRate(0, 0, 1, 0, 0);
                break;
            case PassRate::CRASHED:
                opsStats[op] = PassRate(0, 0, 0, 1, 0);
                break;
            case PassRate::HANGED:
                opsStats[op] = PassRate(0, 0, 0, 0, 1);
                break;
        }
    }
}

void Summary::updateOPsImplStatus(const ngraph::NodeTypeInfo &op, const bool implStatus) {
    auto it = opsStats.find(op);
    if (it != opsStats.end()) {
        if (!it->second.isImplemented && implStatus) {
            it->second.isImplemented = true;
        }
    } else {
        opsStats[op] = PassRate(0, 0, 0, 0, 0);
        opsStats[op].isImplemented = implStatus;
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
        auto h = std::stoi(child.attribute("hanged").value());
        PassRate obj(p, f, s, c, h);
        oldOpsStat.insert({entry, obj});
    }
    return oldOpsStat;
}

void Summary::updateOPsStats(const std::shared_ptr<ngraph::Function> &function, const PassRate::Statuses &status) {
    if (function->get_parameters().empty()) {
        return;
    }
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
        }
        if (extractBody) {
            if (ngraph::is_type<ngraph::op::TensorIterator>(op)) {
                updateOPsStats(op->get_type_info(), status);
                auto ti = ngraph::as_type_ptr<ngraph::op::TensorIterator>(op);
                auto ti_body = ti->get_function();
                updateOPsStats(ti_body, status);
            } else if (ngraph::is_type<ngraph::op::v5::Loop>(op)) {
                updateOPsStats(op->get_type_info(), status);
                auto loop = ngraph::as_type_ptr<ngraph::op::v5::Loop>(op);
                auto loop_body = loop->get_function();
                updateOPsStats(loop_body, status);
            } else if (ngraph::is_type<ngraph::op::v8::If>(op)) {
                updateOPsStats(op->get_type_info(), status);
                auto if_op = ngraph::as_type_ptr<ngraph::op::v8::If>(op);
                std::vector<std::shared_ptr<ngraph::Function>> bodies;
                for (size_t i = 0; i < if_op->get_internal_subgraphs_size(); i++) {
                    auto if_body = if_op->get_function(i);
                    updateOPsStats(if_body, status);
                }
            }
        }
        updateOPsStats(op->get_type_info(), status);
    }
}

void Summary::updateOPsImplStatus(const std::shared_ptr<ngraph::Function> &function, const bool implStatus) {
    if (function->get_parameters().empty()) {
        return;
    }
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
            updateOPsImplStatus(op->get_type_info(), implStatus);
            auto ti = ngraph::as_type_ptr<ngraph::op::TensorIterator>(op);
            auto ti_body = ti->get_function();
            updateOPsImplStatus(ti_body, implStatus);
        } else if (ngraph::is_type<ngraph::op::v5::Loop>(op)) {
            updateOPsImplStatus(op->get_type_info(), implStatus);
            auto loop = ngraph::as_type_ptr<ngraph::op::v5::Loop>(op);
            auto loop_body = loop->get_function();
            updateOPsImplStatus(loop_body, implStatus);
        } else {
            updateOPsImplStatus(op->get_type_info(), implStatus);
        }
    }
}

#ifdef IE_TEST_DEBUG
void Summary::saveDebugReport(const char* className, const char* opName, unsigned long passed, unsigned long failed,
                              unsigned long skipped, unsigned long crashed, unsigned long hanged) {
    std::string outputFilePath = "./part_report.txt";
    std::ofstream file;
    file.open(outputFilePath, std::ios_base::app);
    file << className << ' ' << opName << ' ' << passed << ' ' << failed << ' ' << skipped << ' ' << crashed << ' ' << hanged << '\n';
    file.close();
}
#endif  //IE_TEST_DEBUG

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

    const bool fileExists = CommonTestUtils::fileExists(outputFilePath);

    time_t rawtime;
    struct tm *timeinfo;
    char timeNow[80];

    time(&rawtime);
    // cpplint require to use localtime_r instead which is not available in C++11
    timeinfo = localtime(&rawtime); // NOLINT

    strftime(timeNow, sizeof(timeNow), "%d-%m-%Y %H:%M:%S", timeinfo);

    pugi::xml_node root;
    if (fileExists) {
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
        entry.append_attribute("implemented").set_value(it.second.isImplemented);
        entry.append_attribute("passed").set_value(it.second.passed);
        entry.append_attribute("failed").set_value(it.second.failed);
        entry.append_attribute("skipped").set_value(it.second.skipped);
        entry.append_attribute("crashed").set_value(it.second.crashed);
        entry.append_attribute("hanged").set_value(it.second.hanged);
        entry.append_attribute("passrate").set_value(it.second.getPassrate());
    }

    if (extendReport && fileExists) {
        auto opStataFromReport = summary.getOpStatisticFromReport();
        for (auto &item : opStataFromReport) {
            pugi::xml_node entry;
            if (opList.find(item.first) == opList.end()) {
                entry = currentDeviceNode.append_child(item.first.c_str());
                entry.append_attribute("implemented").set_value(item.second.isImplemented);
                entry.append_attribute("passed").set_value(item.second.passed);
                entry.append_attribute("failed").set_value(item.second.failed);
                entry.append_attribute("skipped").set_value(item.second.skipped);
                entry.append_attribute("crashed").set_value(item.second.crashed);
                entry.append_attribute("hanged").set_value(item.second.hanged);
                entry.append_attribute("passrate").set_value(item.second.getPassrate());
            } else {
                entry = currentDeviceNode.child(item.first.c_str());
                auto implStatus = entry.attribute("implemented").value() == std::string("true") ? true : false;
                auto p = std::stoi(entry.attribute("passed").value()) + item.second.passed;
                auto f = std::stoi(entry.attribute("failed").value()) + item.second.failed;
                auto s = std::stoi(entry.attribute("skipped").value()) + item.second.skipped;
                auto c = std::stoi(entry.attribute("crashed").value()) + item.second.crashed;
                auto h = std::stoi(entry.attribute("hanged").value()) + item.second.hanged;
                PassRate obj(p, f, s, c, h);

                (implStatus || obj.isImplemented)
                    ? entry.attribute("implemented").set_value(true)
                    : entry.attribute("implemented").set_value(false);
                entry.attribute("passed").set_value(obj.passed);
                entry.attribute("failed").set_value(obj.failed);
                entry.attribute("skipped").set_value(obj.skipped);
                entry.attribute("crashed").set_value(obj.crashed);
                entry.attribute("hanged").set_value(obj.hanged);
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
}
