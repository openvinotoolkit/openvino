#include <random>
#include <iostream>

int main () {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> leftDistr(550, 1000);
    std::uniform_int_distribution<> rightDistr(0, 450);

    const char *patchGenerator = R"V0G0N(
[
    {
        "str": "std::cout << \"prefix\\nThroughput: \" << leftDistr(gen) << \".0 FPS\\n\";",
        "comment": "version_1"
    },
    {
        "str": "std::cout << \"prefix\\nThroughput: \" << rightDistr(gen) << \".0 FPS\\n\";",
        "comment": "version_2"
    }
]
)V0G0N";
    
    return 0;
}