#define NGRAPH_PASS(NAME, NAMESPACE) \
transforms.push_back(manager.register_pass<NAMESPACE::NAME>());

#define REGISTER_GRAPH_REWRITE_PASS(A) \
class A : public ngraph::pass::GraphRewrite { public: A() : GraphRewrite() {} }; \
transforms.push_back(manager.register_pass<A>()); \
if (auto pass = std::dynamic_pointer_cast<ngraph::pass::GraphRewrite>(transforms.back())) { \
    anchor = pass; \
} else { throw ngraph::ngraph_error(""); }

#define REGISTER_MATCHER(NAME, NAMESPACE) \
auto NAME = std::make_shared<NAMESPACE::NAME>(); \
anchor->copy_matchers(NAME);

