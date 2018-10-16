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

#pragma once

#include <list>
#include <map>
#include <vector>
#include <string>
#include <memory>

/**
 * @class SetGenerator
 * @brief A SetGenerator provides utility functions to read labels and create a multimap of images for pre-processing
 */
class ClassificationSetGenerator {
    std::map<std::string, int> _classes;

    std::multimap<int, std::string> validationMapFromTxt(const std::string& file);
    std::multimap<int, std::string> validationMapFromFolder(const std::string& dir);

protected:
    std::list<std::string> getDirContents(const std::string& dir, bool includePath = true);


public:
    /**
     * @brief Reads file with a list of classes names. Every found line is considered to be
     *        a class name with ID equal to line number - 1 (zero based)
     * @param labels - name of a file with labels
     * @return <class name, ID> map
     */
    std::map<std::string, int> readLabels(const std::string& labels);

    /**
     * @brief Creates an {ID: IMAGEPATH} multimap to reflect images data reflected by path provided
     * @param path - can be a .txt file or a folder. In case of file parses it assuming format is
     *               relative_path_from_folder_with_txt_extension/image_id. In case of folder searches
     *               all subfolders which are named exactly like known classes and adds all containing
     *               files to a map with ID corresponding to subfolder name
     * @return Multimap {ID: IMAGEPATH} multimap describing all found images. In case folder path was
     *         provided and no class names are known returns empty map
     */
    std::multimap<int, std::string> getValidationMap(const std::string& path);
};
