#ifndef INFERENCEENGINE_BRIDGE_INPUT_OUTPUT_INFO_H
#define INFERENCEENGINE_BRIDGE_INPUT_OUTPUT_INFO_H
namespace InferenceEngineBridge {
    class DataInfo {
    public:
        void setPrecision(std::string precision);

    private:
        InferenceEngine::InputInfo::Ptr actual;
        std::vector<std::size_t> dims;
        std::string precision;
        std::string layout;

    };

    class InputInfo : public DataInfo {
    public:
        void setLayout(std::string layout);
    };

    class OutputInfo : public DataInfo {
    };
}
#endif //INFERENCEENGINE_BRIDGE_INPUT_OUTPUT_INFO_H
