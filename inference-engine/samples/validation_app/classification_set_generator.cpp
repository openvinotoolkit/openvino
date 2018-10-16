/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "classification_set_generator.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <string>

#include "user_exception.hpp"
#include "details/ie_exception.hpp"

#ifdef _WIN32
# include "w_dirent.h"
#else
# include <sys/stat.h>
# include <dirent.h>
#endif

#include "../common/samples/common.hpp"

/**
 * @brief Gets path part of a filename including separator
 * @param filepath - filename to extract path part from
 * @return string with path part of the filename
 */
inline std::string folderOf(const std::string &filepath) {
    auto pos = filepath.rfind("/");
    if (pos == std::string::npos) pos = filepath.rfind("\\");
    if (pos == std::string::npos) return "";
    return filepath.substr(0, pos + 1);
}

void readFile(std::string filename, std::function<void(std::string&, int lineNumber)> perLine) {
    std::ifstream inputFile;
    inputFile.open(filename, std::ios::in);
    std::string strLine = "";

    if (!inputFile.is_open())
        THROW_IE_EXCEPTION << "Cannot open file: " << filename;

    size_t lineNumber = 0;
    while (std::getline(inputFile, strLine)) {
        lineNumber++;
        perLine(strLine, lineNumber);
    }
}

std::map<std::string, int> ClassificationSetGenerator::readLabels(const std::string& labels) {
    _classes.clear();
    int i = 0;

    readFile(labels, [&](std::string& line, size_t lineNumber) {
        trim(line);
        _classes[line] = i++;
    });

    return _classes;
}

std::string getFullName(const std::string& name, const std::string& dir) {
    return dir + "/" + name;
}

std::list<std::string> ClassificationSetGenerator::getDirContents(const std::string& dir, bool includePath) {
    struct stat sb;
    if (stat(dir.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)) {
        return std::list<std::string>();
        // THROW_USER_EXCEPTION(1) << "Can't read contents of directory " << dir << ". It isn't a directory or not accessible";
    }

    std::list<std::string> list;
    DIR *dp;
    dp = opendir(dir.c_str());
    if (dp == nullptr) {
        THROW_USER_EXCEPTION(1) << "Can't open directory " << dir;
    }

    struct dirent *ep;
    while (nullptr != (ep = readdir(dp))) {
        std::string fileName = ep->d_name;
        if (fileName == "." || fileName == "..") continue;
        list.push_back(includePath ? getFullName(ep->d_name, dir) : ep->d_name);
    }
    closedir(dp);
    return list;
}

std::multimap<int, std::string> ClassificationSetGenerator::validationMapFromTxt(const std::string& file) {
    std::string ext = fileExt(file);
    if (ext != "txt") {
        THROW_USER_EXCEPTION(1) << "Unknown dataset data file format: " << ext << "";
    }

    std::string dir = folderOf(file);
    std::multimap<int, std::string> validationMap;
    std::string imgPath = "";
    int classId = -1;

    readFile(file, [&](std::string& line, size_t lineNumber) {
        trim(line);
        size_t pos = line.rfind(" ");
        if (pos == std::string::npos) {
            THROW_USER_EXCEPTION(1) << "Bad file format! Cannot parse line " << lineNumber << ":\n> " << line;
        }
        try {
            classId = std::stoi(line.substr(pos + 1));
        } catch (const std::invalid_argument& e) {
            THROW_USER_EXCEPTION(1) << "Invalid class id specified at line " << lineNumber << ":\n> " << line;
        }
        imgPath = line.substr(0, pos);
        validationMap.insert({ classId, dir + imgPath });
    });

    return validationMap;
}

std::multimap<int, std::string> ClassificationSetGenerator::validationMapFromFolder(const std::string& dir) {
    std::multimap<int, std::string> validationMap;
    std::list<std::string> validation_labels = getDirContents(dir, false);

    for (auto& label : validation_labels) {
        auto val = _classes.find(label);
        if (val == _classes.end()) continue;

        int id = val->second;
        for (auto& image : getDirContents(getFullName(label, dir))) {
            validationMap.insert({ id + 1, image });        // [CVS-8200] line in .labels file is counted from 0, but classes are counted from 1
        }
    }
    return validationMap;
}

std::multimap<int, std::string> ClassificationSetGenerator::getValidationMap(const std::string& path) {
    struct stat sb;
    if (stat(path.c_str(), &sb) == 0) {
        if (S_ISDIR(sb.st_mode)) {
            return validationMapFromFolder(path);
        } else {
            return validationMapFromTxt(path);
        }
    } else {
        if (errno == ENOENT || errno == EINVAL || errno == EACCES) {
            THROW_USER_EXCEPTION(3) << "The specified path \"" << path << "\" can not be found or accessed";
        }
    }
    return{};
}
