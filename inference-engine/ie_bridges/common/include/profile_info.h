#ifndef INFERENCEENGINE_PROFILE_INFO_H
#define INFERENCEENGINE_PROFILE_INFO_H
namespace InferenceEngineBridge {
    struct ProfileInfo {
        std::string status;
        std::string exec_type;
        std::string layer_type;
        int64_t real_time;
        int64_t cpu_time;
        unsigned execution_index;
    };
}
#endif //INFERENCEENGINE_PROFILE_INFO_H
