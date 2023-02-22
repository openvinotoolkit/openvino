#include "errors.hpp"

void reportError(const Napi::Env& env, std::string msg) {
    Napi::Error::New(env, msg).ThrowAsJavaScriptException();
}