#pragma once

//#undef DEBUG_USE_NEW_PASS
#define DEBUG_USE_NEW_PASS 1

//#undef EMUTEX_DEBUG_SAVE_NGRAPH_ENABLE
#define EMUTEX_DEBUG_SAVE_NGRAPH_ENABLE 1

#ifdef EMUTEX_DEBUG_SAVE_NGRAPH_ENABLE
#include "ngraph/pass/visualize_tree.hpp"
#define EMUTEX_DEBUG_SAVE_NGRAPH(manager, name) manager.register_pass<ngraph::pass::VisualizeTree>("/home/ekotov/ngraph_debug/" name ".png")
#else
#define EMUTEX_DEBUG_SAVE_NGRAPH(manager, name)
#endif

#define EMUTEX_DEBUG_VALUE(x) std::cout << __FILE__ << ":" << __LINE__ << " " << #x << " " << x << std::endl;
#define EMUTEX_DEBUG_CHECKPOINT std::cout << __FILE__ << ":" << __LINE__ << std::endl;