// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "net_model.hpp"

//------------------------------------------------------------------------------
// Implementation of methods of class Model
//------------------------------------------------------------------------------

Model::Model(const std::string &folderName,
             const std::string &fileName,
             const std::string &resolution,
             const std::string & extension) :
        folderName_(folderName),
        fileName_(fileName),
        resolutionName_(resolution),
        extensionName_(extension) {
};
