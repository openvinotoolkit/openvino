// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef SHAPES_H
#define SHAPES_H

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <xml_net_builder.hpp>
#include <random>
#include <chrono>

using namespace testing;

struct Maps{
    std::map<std::string, int> mapOfEqualShapes {
            // Layer name, Correct num of input, Correct num of output
            { "Convolution", 1},
            { "Deconvolution", 1},
            { "Crop", 1},
            { "Interp", 1}
    };

    std::map<std::string, std::pair<int, int>> mapOfUnequalShapes {
            // Layer name, Correct num of input, Correct num of output
            { "Convolution", {3, 1}},
            { "Deconvolution", {3, 1}},
            { "Crop", {2, 1}},
            { "DetectionOutput", {3, 1}},
            { "Interp", {2, 1}}
    };

    std::map<std::string, std::pair<std::string, std::string>> mapOfContinuousShapes {
            // Layer name, Correct num of input, Correct num of output
            { "Slice", {"1", "N"}},
            { "Eltwise", {"N", "1"}}
    };
} maps;

class ShapesHelper {
protected:
    std::string type;
public:
    ShapesHelper() = default;

    explicit ShapesHelper(std::string& type) {
        this->type = type;
    }

    std::string getType() {return type;}

    virtual testing::InOutShapes getValidShapes() = 0;
    virtual testing::InOutShapes getInvalidInputShapes() = 0;

    std::vector<std::vector<size_t>> generateShapes(const int& numOfShapes) {
        std::mt19937 gen(static_cast<unsigned long>(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
        std::uniform_int_distribution<unsigned long> dist(1, 256);

        std::vector<std::vector<size_t>> shape;
        shape.reserve(static_cast<unsigned long>(numOfShapes));
        for (int i = 0; i < numOfShapes; ++i) {
            shape.push_back({dist(gen), dist(gen), dist(gen), 7});
        }
        return shape;
    }
    virtual ~ShapesHelper() = default;
};

class EqualIOShapesHelper : public ShapesHelper {
public:
    explicit EqualIOShapesHelper(std::string& type) : ShapesHelper(type) {};

    testing::InOutShapes getValidShapes() override {
        int numOfInput = {maps.mapOfEqualShapes[type]};
        int numOfOutput = {maps.mapOfEqualShapes[type]};
        std::vector<std::vector<size_t>> inputs = generateShapes(numOfInput);
        std::vector<std::vector<size_t>> outputs = generateShapes(numOfOutput);
        return {inputs, outputs};
    }

    testing::InOutShapes getInvalidInputShapes()  override {
        int numOfOutput = maps.mapOfEqualShapes[type];
        int numOfInput = maps.mapOfEqualShapes[type]  + numOfOutput;
        std::vector<std::vector<size_t>> inputs = generateShapes(numOfInput);
        std::vector<std::vector<size_t>> outputs = generateShapes(numOfOutput);
        return {inputs, outputs};
    }
    ~EqualIOShapesHelper() override = default;
};

class NotEqualConcreteIOShapesHelper : public ShapesHelper {
public:
    explicit NotEqualConcreteIOShapesHelper(std::string& type) : ShapesHelper(type) {};

    testing::InOutShapes getValidShapes() override {
        int numOfInput = maps.mapOfUnequalShapes[type].first;
        int numOfOutput = maps.mapOfUnequalShapes[type].second;
        std::vector<std::vector<size_t>> inputs = generateShapes(numOfInput);
        std::vector<std::vector<size_t>> outputs = generateShapes(numOfOutput);
        return {inputs, outputs};
    }

    testing::InOutShapes getInvalidInputShapes()  override {
        int numOfOutput = maps.mapOfUnequalShapes[type].second;
        int numOfInput = maps. mapOfUnequalShapes[type].first + numOfOutput;

        std::vector<std::vector<size_t>> inputs = generateShapes(numOfInput);
        std::vector<std::vector<size_t>> outputs = generateShapes(numOfOutput);
        return {inputs, outputs};
    }
    ~NotEqualConcreteIOShapesHelper() override = default;
};

class NotEqualIOShapesHelper : public ShapesHelper {
private:
    bool is_number(const std::string& s)
    {
        return !s.empty() && std::find_if(s.begin(),
                                          s.end(), [](char c) { return !std::isdigit(c); }) == s.end();
    }

public:

    explicit NotEqualIOShapesHelper(std::string& type) : ShapesHelper(type) {};

    testing::InOutShapes getValidShapes() override {
        int numOfInput;
        int numOfOutput;
        std::vector<std::vector<size_t>> inputs;
        std::vector<std::vector<size_t>> outputs;
        if (is_number(maps.mapOfContinuousShapes[type].first)) {
            numOfInput = std::stoi(maps.mapOfContinuousShapes[type].first);
            inputs = generateShapes(numOfInput);
            outputs = generateShapes(100);
        } else {
            numOfOutput = std::stoi(maps.mapOfContinuousShapes[type].second);
            outputs = generateShapes(numOfOutput);
            inputs = generateShapes(100);
        }

        return {inputs, outputs};
    }

    testing::InOutShapes getInvalidInputShapes()  override {
        int numOfInput;
        int numOfOutput;
        std::vector<std::vector<size_t>> inputs;
        std::vector<std::vector<size_t>> outputs;
        if (is_number(maps.mapOfContinuousShapes[type].first)) {
            numOfInput = std::stoi(maps.mapOfContinuousShapes[type].first) * 2;
            inputs = generateShapes(numOfInput);
            outputs = generateShapes(100);
        } else {
            numOfOutput = std::stoi(maps.mapOfContinuousShapes[type].second);
            outputs = generateShapes(numOfOutput);
            inputs = generateShapes(100);
        }
        return {inputs, outputs};
    }

    ~NotEqualIOShapesHelper() override = default;
};

class Layers {
public:
    virtual bool containLayer(std::string concrete_layer) = 0;
    virtual ShapesHelper* factoryShape() = 0;
    virtual ~Layers() = default;
};

class LayersWithEqualIO : public Layers {
private:
    std::string layer = "";
public:
    bool containLayer(std::string concrete_layer) override {
        for (const auto& layer : maps.mapOfEqualShapes) {
            if (concrete_layer == layer.first) {
                this->layer = concrete_layer;
                return true;
            }
        }
        return false;
    }
    ShapesHelper* factoryShape() override {
        return new EqualIOShapesHelper(this->layer);
    }
    ~LayersWithEqualIO() override = default;
};

class LayersWithNotEqualIO : public Layers{
private:
    std::string layer = "";
public:
    bool containLayer(std::string concrete_layer) override {
        for (const auto& layer : maps.mapOfUnequalShapes) {
            if (concrete_layer == layer.first) {
                this->layer = concrete_layer;
                return true;
            }
        }
        return false;
    }
    ShapesHelper* factoryShape() override {
        return new NotEqualConcreteIOShapesHelper(this->layer);
    }
    ~LayersWithNotEqualIO() override = default;
};

class LayersWithNIO : public Layers{
private:
    std::string layer = "";
public:
    bool containLayer(std::string concrete_layer) override {
        for (const auto& layer : maps.mapOfContinuousShapes) {
            if (concrete_layer == layer.first) {
                this->layer = concrete_layer;
                return true;
            }
        }
        return false;
    }
    ShapesHelper* factoryShape() override {
        return new NotEqualIOShapesHelper(this->layer);
    }
    ~LayersWithNIO() override = default;
};

#endif // SHAPES_H
