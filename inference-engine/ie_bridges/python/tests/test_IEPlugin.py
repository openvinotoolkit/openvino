import warnings
import pytest


from openvino.inference_engine import IENetwork, IEPlugin, ExecutableNetwork
from conftest import model_path

test_net_xml, test_net_bin = model_path()


def test_init_plugin(device):
    with warnings.catch_warnings(record=True) as w:
        plugin = IEPlugin(device, None)
        assert isinstance(plugin, IEPlugin)
    assert len(w) == 1
    assert "IEPlugin class is deprecated. " \
                "Please use IECore class instead." in str(w[0].message)


def test_device_attr(device):
    with warnings.catch_warnings(record=True) as w:
        plugin = IEPlugin(device, None)
        assert plugin.device == device
    assert len(w) == 1
    assert "IEPlugin class is deprecated. " \
               "Please use IECore class instead." in str(w[0].message)


def test_get_version(device):
    with warnings.catch_warnings(record=True) as w:
        plugin = IEPlugin(device, None)
        assert not len(plugin.version) == 0
    assert len(w) == 1
    assert "IEPlugin class is deprecated. " \
               "Please use IECore class instead." in str(w[0].message)


def test_load_network(device):
    with warnings.catch_warnings(record=True) as w:
        plugin = IEPlugin(device, None)
        net = IENetwork(model=test_net_xml, weights=test_net_bin)
        exec_net = plugin.load(net)
        assert isinstance(exec_net, ExecutableNetwork)
    assert len(w) == 2
    assert "IEPlugin class is deprecated. " \
               "Please use IECore class instead." in str(w[0].message)
    assert "Reading network using constructor is deprecated. " \
            "Please, use IECore.read_network() method instead"  in str(w[1].message)


def test_load_network_many_requests(device):
    with warnings.catch_warnings(record=True) as w:
        plugin = IEPlugin(device)
        net = IENetwork(model=test_net_xml, weights=test_net_bin)
        exec_net = plugin.load(net, num_requests=5)
        assert len(exec_net.requests) == 5
    assert len(w) == 2
    assert "IEPlugin class is deprecated. " \
               "Please use IECore class instead." in str(w[0].message)
    assert "Reading network using constructor is deprecated. " \
            "Please, use IECore.read_network() method instead"  in str(w[1].message)


def test_get_supported_layers(device):
    with warnings.catch_warnings(record=True) as w:
        plugin = IEPlugin(device)
        net = IENetwork(model=test_net_xml, weights=test_net_bin)
        supported = plugin.get_supported_layers(net)
        layers = ['19/Fused_Add_', '21', '22', '23', '24/Fused_Add_', '26', '27', '29', 'data', 'fc_out']
        if device == "GPU":
            layers.remove("data")
        assert sorted(supported) == layers
    assert len(w) == 2
    assert "IEPlugin class is deprecated. " \
               "Please use IECore class instead." in str(w[0].message)
    assert "Reading network using constructor is deprecated. " \
            "Please, use IECore.read_network() method instead"  in str(w[1].message)


@pytest.mark.skip(reason="Plugiin specific test.")
def test_set_config(device):
    with warnings.catch_warnings(record=True) as w:
        plugin = IEPlugin("HETERO:CPU")
        plugin.set_config({"TARGET_FALLBACK": "CPU,GPU"})
    assert len(w) == 1
    assert "IEPlugin class is deprecated. " \
               "Please use IECore class instead." in str(w[0].message)


@pytest.mark.skip(reason="Sporadically fail in CI, not reproducible locally")
def test_set_initial_affinity():
    with warnings.catch_warnings(record=True) as w:
        plugin = IEPlugin("HETERO:CPU", None)
        net = IENetwork(model=test_net_xml, weights=test_net_bin)
        plugin.set_initial_affinity(net)
        for l, params in net.layers.items():
            assert params.affinity == "CPU", "Incorrect affinity for {}".format(l)
    assert len(w) == 1
    assert "IEPlugin class is deprecated. " \
               "Please use IECore class instead." in str(w[0].message)


def test_set_initial_affinity_wrong_device(device):
    with pytest.raises(RuntimeError) as e:
        with warnings.catch_warnings(record=True) as w:
            plugin = IEPlugin("CPU", None)
            net = IENetwork(model=test_net_xml, weights=test_net_bin)
            plugin.set_initial_affinity(net)
        assert len(w) == 1
        assert "IEPlugin class is deprecated. " \
               "Please use IECore class instead." in str(w[0].message)
    assert "set_initial_affinity method applicable only for HETERO device" in str(e.value)


def test_add_cpu_extenstion_wrong_device():
    with pytest.raises(RuntimeError) as e:
        with warnings.catch_warnings(record=True) as w:
            plugin = IEPlugin("GPU", None)
            plugin.add_cpu_extension("./")
        assert len(w) == 1
        assert "IEPlugin class is deprecated. " \
               "Please use IECore class instead." in str(w[0].message)
    if "Cannot find plugin to use" in str(e.value):
        pytest.skip("No GPU found. Skipping test")
    else:
        assert "add_cpu_extension method applicable only for CPU or HETERO devices" in str(e.value)


def test_unknown_plugin():
    with pytest.raises(ValueError) as e:
        with warnings.catch_warnings(record=True) as w:
            IEPlugin("BLA")
        assert len(w) == 1
        assert "IEPlugin class is deprecated. " \
               "Please use IECore class instead." in str(w[0].message)
    assert "Unknown plugin: BLA, expected one of:" in str(e.value)
