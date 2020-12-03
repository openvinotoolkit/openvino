#include <api/memory.hpp>
#include <api/topology.hpp>
#include <api/reorder.hpp>
#include <api/input_layout.hpp>
#include <api/convolution.hpp>
#include <api/data.hpp>
#include <api/pooling.hpp>
#include <api/fully_connected.hpp>
#include <api/softmax.hpp>
#include <api/engine.hpp>
#include <api/network.hpp>

#include <iostream>

using namespace cldnn;
using namespace std;

const tensor::value_type
        input_channels = 1,
        input_size = 28,
        conv1_out_channels = 20,
        conv2_out_channels = 50,
        conv_krnl_size = 5,
        fc1_num_outs = 500,
        fc2_num_outs = 10;

// Create layout with same sizes but new format.
layout create_reordering_layout(format new_format, const layout& src_layout)
{
    return { src_layout.data_type, new_format, src_layout.size };
}

// Create MNIST topology
topology create_topology(const layout& in_layout, const memory& conv1_weights_mem, const memory& conv1_bias_mem )
{
    auto data_type = in_layout.data_type;

    // Create input_layout description
    // "input" - is the primitive id inside topology
    input_layout input("input", in_layout);

    // Create topology object with 2 primitives
    cldnn:: topology topology(
        // 1. input layout primitive.
        input,
        // 2. reorder primitive with id "reorder_input"
        reorder("reorder_input",
            // input primitive for reorder (implicitly converted to primitive_id)
            input,
            // output layout for reorder
            create_reordering_layout(format::yxfb, in_layout))
    );

    // Create data primitive - its content should be set already.
    cldnn::data conv1_weights( "conv1_weights", conv1_weights_mem );

    // Add primitive to topology
    topology.add(conv1_weights);

    // Emplace new primitive to topology
    topology.add<cldnn::data>({ "conv1_bias", conv1_bias_mem });

    // Emplace 2 primitives
    topology.add(
        // Convolution primitive with id "conv1"
        convolution("conv1",
            "reorder_input",    // primitive id of the convolution's input
            { conv1_weights },  // weights primitive id is taken from the object
            { "conv1_bias" }    // bias primitive id
        ),
        // Pooling id: "pool1"
        pooling("pool1",
            "conv1",                // Input: "conv1"
            pooling_mode::max,      // Pooling mode: MAX
            tensor(spatial(2,2)),  // stride: 2
            tensor(spatial(2,2))   // kernel_size: 2
        )
    );

    // Conv2 weights data is not available now, so just declare its layout
    layout conv2_weights_layout(data_type, format::bfyx,{ conv2_out_channels, conv1_out_channels, conv_krnl_size, conv_krnl_size });

    // Define the rest of topology.
    topology.add(
        // Input layout for conv2 weights. Data will passed by network::set_input_data()
        input_layout("conv2_weights", conv2_weights_layout),
        // Input layout for conv2 bias.
        input_layout("conv2_bias", { data_type, format::bfyx, tensor(spatial(conv2_out_channels)) }),
        // Second convolution id: "conv2"
        convolution("conv2",
            "pool1",                // Input: "pool1"
            { "conv2_weights" },    // Weights: input_layout "conv2_weights"
            { "conv2_bias" }        // Bias: input_layout "conv2_bias"
        ),
        // Second pooling id: "pool2"
        pooling("pool2",
            "conv2",                // Input: "conv2"
            pooling_mode::max,      // Pooling mode: MAX
            tensor(spatial(2, 2)), // stride: 2
            tensor(spatial(2, 2))  // kernel_size: 2
        ),
        // Fully connected (inner product) primitive id "fc1"
        fully_connected("fc1",
            "pool2",        // Input: "pool2"
            "fc1_weights",  // "fc1_weights" will be added to the topology later
            "fc1_bias"     // will be defined later
        ),
        // Second FC/IP primitive id: "fc2", input: "fc1".
        // Weights ("fc2_weights") and biases ("fc2_bias") will be defined later.
        // Built-in Relu is disabled by default.
        fully_connected("fc2", "fc1", "fc2_weights", "fc2_bias"),
        // The "softmax" primitive is not an input for any other,
        // so it will be automatically added to network outputs.
        softmax("softmax", "fc2")
    );
    return topology;
}

// Copy from a vector to cldnn::memory
void copy_to_memory(memory& mem, const vector<float>& src)
{
    cldnn::pointer<float> dst(mem);
    std::copy(src.begin(), src.end(), dst.begin());
}

// Execute network
int recognize_image(network& network, const memory& input_memory)
{
    // Set/update network input
    network.set_input_data("input", input_memory);

    // Start network execution
    auto outputs = network.execute();

    // get_memory() blocks output generation completed
    auto output = outputs.at("softmax").get_memory();

    // Get direct access to output memory
    cldnn::pointer<float> out_ptr(output);

    // Analyze result
    auto max_element_pos = max_element(out_ptr.begin(), out_ptr.end());
    return static_cast<int>(distance(out_ptr.begin(), max_element_pos));
}

// User-defined helpers which are out of this example scope
// //////////////////////////////////////////////////////////////
// Loads file to a vector of floats.
vector<float> load_data(const string&) { return{ 0 }; }

// Allocates memory and loads data from file.
// Memory layout is taken from file.
memory load_mem(const engine& eng, const string&) {
    //return a dummy value
    return memory::allocate(eng, layout{ data_types::f32, format::bfyx, { 1, 1, 1, 1 } });
}

// Load image, resize to [x,y] and store in a vector of floats
// in the order "bfyx".
vector<float> load_image_bfyx(const string&, int, int) { return{ 0 }; }
// //////////////////////////////////////////////////////////////

int main_func()
{
    // Use data type: float
    auto data_type = type_to_data_type<float>::value;

    // Network input layout
    layout in_layout(
        data_type, // stored data type
        format::bfyx, // data stored in order batch-channel-Y-X, where X coordinate changes first.
            {1, input_channels, input_size, input_size} // batch: 1, channels: 1, Y: 28, X: 28
        );

    // Create memory for conv1 weights
    layout conv1_weights_layout(data_type, format::bfyx,{ conv1_out_channels, input_channels, conv_krnl_size, conv_krnl_size });
    vector<float> my_own_buffer = load_data("conv1_weights.bin");
    // The conv1_weights_mem is attached to my_own_buffer, so my_own_buffer should not be changed or descroyed until network execution completion.
    auto conv1_weights_mem = memory::attach(conv1_weights_layout, my_own_buffer.data(), my_own_buffer.size());

    // Create default engine
    cldnn::engine engine;

    // Create memory for conv1 bias
    layout conv1_bias_layout(data_type, format::bfyx, tensor(spatial(20)));
    // Memory allocation requires engine
    auto conv1_bias_mem = memory::allocate(engine, conv1_bias_layout);
    // The memory is allocated by library, so we do not need to care about buffer lifetime.
    copy_to_memory(conv1_bias_mem, load_data("conv1_bias.bin"));

    // Get new topology
    cldnn::topology topology = create_topology(in_layout, conv1_weights_mem, conv1_bias_mem);

    // Define network data not defined in create_topology()
    topology.add(
        cldnn::data("fc1_weights", load_mem(engine, "fc1_weights.data")),
        cldnn::data("fc1_bias",    load_mem(engine, "fc1_bias.data")),
        cldnn::data("fc2_weights", load_mem(engine, "fc2_weights.data")),
        cldnn::data("fc2_bias",    load_mem(engine, "fc2_bias.data"))
        );

    // Build the network. Allow implicit data optimizations.
    // The "softmax" primitive is not used as an input for other primitives,
    // so we do not need to explicitly select it in build_options::outputs()
    cldnn::build_options options;
    options.set_option(cldnn::build_option::optimize_data(true));
    cldnn::network network(engine, topology, options);

    // Set network data which was not known at topology creation.
    network.set_input_data("conv2_weights", load_mem(engine, "conv2_weights.data"));
    network.set_input_data("conv2_bias", load_mem(engine, "conv2_bias.data"));

    // Allocate memory for input image.
    auto input_memory = memory::allocate(engine, in_layout);

    // Run network 2 times with different images.
    for (auto img_name : { "one.jpg", "two.jpg" })
    {
        // Reuse image memory.
        copy_to_memory(input_memory, load_image_bfyx("one.jpg", in_layout.size.spatial[0], in_layout.size.spatial[1]));
        auto result = recognize_image(network, input_memory);
        cout << img_name << " recognized as" << result << endl;
    }

    return 0;
}
