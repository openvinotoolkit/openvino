import binascii
import json
import os
import base64

from conan.internal.api.audit.providers import ConanCenterProvider, PrivateProvider
from conan.errors import ConanException
from conan.internal.api.remotes.encrypt import encode, decode
from conan.internal.model.recipe_ref import RecipeReference
from conan.internal.util.files import save, load

CONAN_CENTER_AUDIT_PROVIDER_NAME = "conancenter"
CYPHER_KEY = "private"


class AuditAPI:
    """
    This class provides the functionality to scan references for vulnerabilities.
    """

    def __init__(self, conan_api):
        self._conan_api = conan_api
        self._home_folder = conan_api.home_folder
        self._providers_path = os.path.join(self._home_folder, "audit_providers.json")
        self._provider_cls = {
            "conan-center-proxy": ConanCenterProvider,
            "private": PrivateProvider,
        }

    @staticmethod
    def scan(deps_graph, provider, context=None):
        """
        Scan a given recipe for vulnerabilities in its dependencies.
        """
        refs = sorted(set(RecipeReference.loads(f"{node.ref.name}/{node.ref.version}")
                          for node in deps_graph.nodes[1:]
                          if context is None or node.context == context),
                      key=lambda ref: ref.name)
        return provider.get_cves(refs)

    @staticmethod
    def list(references, provider):
        """
        List the vulnerabilities of the given reference.
        """
        refs = [RecipeReference.loads(ref) for ref in references]
        for ref in refs:
            ref.validate_ref()
        return provider.get_cves(refs)

    def get_provider(self, provider_name):
        """
        Get the provider by name.
        """
        # TODO: More work remains to be done here, hardcoded for now for testing
        providers = _load_providers(self._providers_path)
        if provider_name not in providers:
            add_arguments = (
                "--url=https://audit.conan.io/ --type=conan-center-proxy"
                if provider_name == CONAN_CENTER_AUDIT_PROVIDER_NAME
                else "--url=<url> --type=<type>"
            )

            register_message = (
                f"If you don't have a valid token, register at: https://audit.conan.io/register."
                if provider_name == CONAN_CENTER_AUDIT_PROVIDER_NAME
                else ""
            )

            raise ConanException(
                f"Provider '{provider_name}' not found. Please specify a valid provider name or add "
                f"it using: 'conan audit provider add {provider_name} {add_arguments} "
                f"--token=<token>'\n{register_message}"
            )

        provider_data = providers[provider_name]
        safe_provider_name = provider_name.replace("-", "_")
        env_token = os.getenv(f"CONAN_AUDIT_PROVIDER_TOKEN_{safe_provider_name.upper()}")

        if env_token:
            # Always override the token with the environment variable
            provider_data["token"] = env_token
        elif "token" in provider_data:
            try:
                enc_token = base64.standard_b64decode(provider_data["token"]).decode()
                provider_data["token"] = decode(enc_token, CYPHER_KEY)
            except binascii.Error:
                raise ConanException(f"Invalid token format for provider '{provider_name}'. "
                                     f"The token might be corrupt.")

        provider_cls = self._provider_cls.get(provider_data["type"])

        return provider_cls(self._conan_api, provider_name, provider_data)

    def list_providers(self):
        """
        Get all available providers.
        """
        providers = _load_providers(self._providers_path)
        result = []
        for name, provider_data in providers.items():
            provider_cls = self._provider_cls.get(provider_data["type"])
            result.append(provider_cls(self._conan_api, name, provider_data))
        return result

    def add_provider(self, name, url, provider_type):
        """
        Add a provider.
        """
        providers = _load_providers(self._providers_path)
        if name in providers:
            raise ConanException(f"Provider '{name}' already exists")

        if provider_type not in self._provider_cls:
            raise ConanException(f"Provider type '{provider_type}' not found")

        providers[name] = {
            "name": name,
            "url": url,
            "type": provider_type
        }

        _save_providers(self._providers_path, providers)

    def remove_provider(self, provider_name):
        """
        Remove a provider.
        """
        providers = _load_providers(self._providers_path)
        if provider_name not in providers:
            raise ConanException(f"Provider '{provider_name}' not found")

        del providers[provider_name]

        _save_providers(self._providers_path, providers)

    def auth_provider(self, provider, token):
        """
        Authenticate a provider.
        """
        if not provider:
            raise ConanException("Provider not found")

        providers = _load_providers(self._providers_path)

        assert provider.name in providers
        encode_token = encode(token, CYPHER_KEY).encode()
        providers[provider.name]["token"] = base64.standard_b64encode(encode_token).decode()
        setattr(provider, "token", token)
        _save_providers(self._providers_path, providers)


def _load_providers(providers_path):
    if not os.path.exists(providers_path):
        default_providers = {
            CONAN_CENTER_AUDIT_PROVIDER_NAME: {
                "url": "https://audit.conan.io/",
                "type": "conan-center-proxy"
            }
        }
        save(providers_path, json.dumps(default_providers, indent=4))

    return json.loads(load(providers_path))


def _save_providers(providers_path, providers):
    save(providers_path, json.dumps(providers, indent=4))
    # Make readable & writeable only by current user
    os.chmod(providers_path, 0o600)
