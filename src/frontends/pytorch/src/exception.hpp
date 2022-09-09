
namespace ov {
namespace frontend {
namespace pytorch {

#define OV_FRONTEND_REQUIRE(X)                                                                          \
    do                                                                                                  \
        if (!(X)) {                                                                                     \
            throw std::runtime_error(std::string("[ ERROR ] Failed: ") + #X + " at " + __FILE__ + ":" + \
                                     std::to_string(__LINE__));                                         \
        }                                                                                               \
    while (false)

}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
