# Writing ngraph transformations {#new_ngraph_transformation}

1. Code of such transformation MUST be directly in template plugin soruces.
2. It must be mark with doxygen markeds like


// ! [new_transformation:part1]

bla-bla

// ! [new_transformation:part1]

And this file must refer to that code via

@snippet src/template_transformation.cpp new_transformation:part1
