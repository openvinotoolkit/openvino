import os
import pytest


from openvino.inference_engine import IENetwork, IEPlugin, ExecutableNetwork

SAMPLENET_XML = os.path.join(os.path.dirname(__file__), 'test_data', 'models', 'SampLeNet.xml')
SAMPLENET_BIN = os.path.join(os.path.dirname(__file__), 'test_data', 'models', 'SampLeNet.bin')


def test_init_plugin(device):
    plugin = IEPlugin(device, None)
    assert isinstance(plugin, IEPlugin)


def test_device_attr(device):
    plugin = IEPlugin(device, None)
    assert plugin.device == device


def test_get_version(device):
    plugin = IEPlugin(device, None)
    assert not len(plugin.version) == 0


def test_load_network(device):
    plugin = IEPlugin(device, None)
    net = IENetwork(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    exec_net = plugin.load(net)
    assert isinstance(exec_net, ExecutableNetwork)


def test_load_network_many_requests(device):
    plugin = IEPlugin(device)
    net = IENetwork(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    exec_net = plugin.load(net, num_requests=5)
    assert len(exec_net.requests) == 5


def test_get_supported_layers(device):
    plugin = IEPlugin(device)
    net = IENetwork(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    supported = plugin.get_supported_layers(net)
    layers = ['conv1', 'conv2', 'data', 'fc1', 'fc2', 'fc3', 'pool1', 'pool2',
              'relu_conv1', 'relu_conv2', 'relu_fc1', 'relu_fc2']
    if device == "GPU":
        layers.remove("data")
    assert sorted(supported) == layers


@pytest.mark.skip(reason="Plugiin specific test.")
def test_set_config(device):
    plugin = IEPlugin("HETERO:CPU")
    plugin.set_config({"TARGET_FALLBACK": "CPU,GPU"})


@pytest.mark.skip(reason="Sporadically fail in CI, not reproducible locally")
def test_set_initial_affinity():
    plugin = IEPlugin("HETERO:CPU", None)
    net = IENetwork(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
    plugin.set_initial_affinity(net)
    for l, params in net.layers.items():
        assert params.affinity == "CPU", "Incorrect affinity for {}".format(l)


def test_set_initial_affinity_wrong_device(device):
    with pytest.raises(RuntimeError) as e:
        plugin = IEPlugin("CPU", None)
        net = IENetwork(model=SAMPLENET_XML, weights=SAMPLENET_BIN)
        plugin.set_initial_affinity(net)
    assert "set_initial_affinity method applicable only for HETERO device" in str(e.value)


def test_add_cpu_extenstion_wrong_device():
    with pytest.raises(RuntimeError) as e:
        plugin = IEPlugin("GPU", None)
        plugin.add_cpu_extension("./")
    if "Cannot find plugin to use" in str(e.value):
        pytest.skip("No GPU found. Skipping test")
    else:
        assert "add_cpu_extension method applicable only for CPU or HETERO devices" in str(e.value)


def test_unknown_plugin():
    with pytest.raises(ValueError) as e:
        IEPlugin("BLA")
    assert "Unknown plugin: BLA, expected one of:" in str(e.value)
