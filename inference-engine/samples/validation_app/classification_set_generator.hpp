// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include <utility>

/**
 * @class SetGenerator
 * @brief A SetGenerator provides utility functions to read labels and create a multimap of images for pre-processing
 */
class ClassificationSetGenerator {
    std::map<std::string, int> _classes;

    std::vector<std::pair<int, std::string>> validationMapFromTxt(const std::string& file);
    std::vector<std::pair<int, std::string>> validationMapFromFolder(const std::string& dir);

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
     * @brief Creates a  vector of pairs <class id, path to picture> to reflect
     * images data reflected by path provided
     * @param path - can be a .txt file or a folder. In case of file parses it assuming format is
     *               relative_path_from_folder_with_txt_extension/image_id. In case of folder searches
     *               all subfolders which are named exactly like known classes and adds all containing
     *               files to a map with ID corresponding to subfolder name
     * @return vector of pairs {ID: IMAGEPATH} describing all found images. In case folder path was
     *         provided and no class names are known returns empty map
     */
    std::vector<std::pair<int, std::string>> getValidationMap(const std::string& path);
};
