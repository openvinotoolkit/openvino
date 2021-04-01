// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>

//------------------------------------------------------------------------------
// class Model
//------------------------------------------------------------------------------

class Model {
public:
    //Constructors
    Model() = default;
    explicit Model(const char *that) {
        fileName_ = folderName_ = that;
    }

    Model(const std::string &folderName,
            const std::string &fileName,
            const std::string &resolution,
            const std::string & extension = "xml");

    // Accessors
    inline std::string folderName() const { return folderName_; };
    inline std::string fileName() const { return fileName_; };
    inline std::string resolution() const { return resolutionName_; };
    inline std::string extension() const { return extensionName_; };

private:
    std::string folderName_;
    std::string fileName_;
    std::string resolutionName_;
    std::string extensionName_;
};

