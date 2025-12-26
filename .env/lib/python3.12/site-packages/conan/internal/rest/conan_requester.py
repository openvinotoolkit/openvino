import fnmatch
import json
import logging
import os
import platform

import requests
import urllib3
from jinja2 import Template
from requests.adapters import HTTPAdapter

from conan.api.output import ConanOutput
from conan.internal.cache.home_paths import HomePaths

from conan import __version__
from conan.internal.loader import load_python_file
from conan.internal.errors import scoped_traceback
from conan.errors import ConanException

# Capture SSL warnings as pointed out here:
# https://urllib3.readthedocs.org/en/latest/security.html#insecureplatformwarning
# TODO: Fix this security warning
from conan.internal.util.files import load

logging.captureWarnings(True)


DEFAULT_TIMEOUT = (30, 60)  # connect, read timeouts
INFINITE_TIMEOUT = -1


class _SourceURLCredentials:
    """
    Only for sources download (get(), download(), conan config install
    """
    def __init__(self, cache_folder):
        self._urls = {}
        self._auth_source_plugin = None
        if not cache_folder:
            return
        auth_source_plugin_path = HomePaths(cache_folder).auth_source_plugin_path
        self._auth_source_plugin = _load_auth_source_plugin(auth_source_plugin_path)
        creds_path = os.path.join(cache_folder, "source_credentials.json")
        if not os.path.exists(creds_path):
            return

        def _get_auth(credentials):
            if "token" in credentials or ("user" in credentials and "password" in credentials):
                return credentials
            raise ConanException(f"Unknown credentials method for '{credentials['url']}'")

        try:
            template = Template(load(creds_path))
            content = template.render({"platform": platform, "os": os})
            content = json.loads(content)
            self._urls = {credentials["url"]: _get_auth(credentials)
                          for credentials in content["credentials"]}
        except Exception as e:
            raise ConanException(f"Error loading 'source_credentials.json' {creds_path}: {repr(e)}")

    def add_auth(self, url, kwargs):
        # First, try to use "auth_source_plugin"
        if self._auth_source_plugin:
            try:
                c = self._auth_source_plugin(url)
            except Exception as e:
                msg = f"Error while processing 'auth_source_remote.py' plugin"
                msg = scoped_traceback(msg, e, scope="/extensions/plugins")
                raise ConanException(msg)
            if c:
                if c.get("token"):
                    kwargs["headers"]["Authorization"] = f"Bearer {c.get('token')}"
                if c.get("user") and c.get("password"):
                    kwargs["auth"] = (c.get("user"), c.get("password"))
                if c.get("headers"):
                    kwargs.setdefault("headers", {}).update(c["headers"])
                return

        # Then, try to find the credentials in "_urls"
        for u, creds in self._urls.items():
            if url.startswith(u):
                token = creds.get("token")
                if token:
                    kwargs["headers"]["Authorization"] = f"Bearer {token}"
                user = creds.get("user")
                password = creds.get("password")
                if user and password:
                    kwargs["auth"] = (user, password)
                headers = creds.get("headers")
                if headers:
                    kwargs.setdefault("headers", {}).update(headers)
                break


class ConanRequester:

    def __init__(self, config, cache_folder=None):
        self._url_creds = _SourceURLCredentials(cache_folder)
        _max_retries = config.get("core.net.http:max_retries", default=2, check_type=int)
        self._http_requester = requests.Session()
        _adapter = HTTPAdapter(max_retries=self._get_retries(_max_retries))
        self._http_requester.mount("http://", _adapter)
        self._http_requester.mount("https://", _adapter)
        self._timeout = config.get("core.net.http:timeout", default=DEFAULT_TIMEOUT)
        self._no_proxy_match = config.get("core.net.http:no_proxy_match", check_type=list)
        self._proxies = config.get("core.net.http:proxies")
        self._cacert_path = config.get("core.net.http:cacert_path", check_type=str)
        self._client_certificates = config.get("core.net.http:client_cert")
        self._clean_system_proxy = config.get("core.net.http:clean_system_proxy", default=False,
                                              check_type=bool)
        platform_info = "; ".join([" ".join([platform.system(), platform.release()]),
                                   "Python " + platform.python_version(),
                                   platform.machine()])
        self._user_agent = "Conan/%s (%s)" % (__version__, platform_info)

    @staticmethod
    def _get_retries(max_retries):
        retry = max_retries
        if retry == 0:
            return 0
        retry_status_code_set = {
            requests.codes.internal_server_error,
            requests.codes.bad_gateway,
            requests.codes.service_unavailable,
            requests.codes.gateway_timeout,
            requests.codes.variant_also_negotiates,
            requests.codes.insufficient_storage,
            requests.codes.bandwidth_limit_exceeded
        }
        return urllib3.Retry(
            total=retry,
            backoff_factor=0.05,
            status_forcelist=retry_status_code_set
        )

    def _should_skip_proxy(self, url):
        if self._no_proxy_match:
            for entry in self._no_proxy_match:
                if fnmatch.fnmatch(url, entry):
                    return True
        return False

    def _add_kwargs(self, url, kwargs):
        # verify is the kwargs that comes from caller, RestAPI, it is defined in
        # Conan remote "verify_ssl"
        source_credentials = kwargs.pop("source_credentials", None)
        if kwargs.get("verify", None) is not False:  # False means de-activate
            if self._cacert_path is not None:
                kwargs["verify"] = self._cacert_path
        kwargs["cert"] = self._client_certificates
        if self._proxies:
            if not self._should_skip_proxy(url):
                kwargs["proxies"] = self._proxies
        if self._timeout and self._timeout != INFINITE_TIMEOUT:
            kwargs["timeout"] = self._timeout
        if not kwargs.get("headers"):
            kwargs["headers"] = {}

        if source_credentials:
            self._url_creds.add_auth(url, kwargs)

        # Only set User-Agent if none was provided
        if not kwargs["headers"].get("User-Agent"):
            kwargs["headers"]["User-Agent"] = self._user_agent

        return kwargs

    def get(self, url, **kwargs):
        return self._call_method("get", url, **kwargs)

    def head(self, url, **kwargs):
        return self._call_method("head", url, **kwargs)

    def put(self, url, **kwargs):
        return self._call_method("put", url, **kwargs)

    def delete(self, url, **kwargs):
        return self._call_method("delete", url, **kwargs)

    def post(self, url, **kwargs):
        return self._call_method("post", url, **kwargs)

    def _call_method(self, method, url, **kwargs):
        popped = False
        if self._clean_system_proxy:
            old_env = dict(os.environ)
            # Clean the proxies from the environ and use the conan specified proxies
            for var_name in ("http_proxy", "https_proxy", "ftp_proxy", "all_proxy", "no_proxy"):
                popped = True if os.environ.pop(var_name, None) else popped
                popped = True if os.environ.pop(var_name.upper(), None) else popped
        ConanOutput(scope="HttpRequest").trace(f"{method}: {url}")
        try:
            all_kwargs = self._add_kwargs(url, kwargs)
            tmp = getattr(self._http_requester, method)(url, **all_kwargs)
            return tmp
        finally:
            if popped:
                os.environ.clear()
                os.environ.update(old_env)


def _load_auth_source_plugin(auth_source_plugin_path):
    if os.path.exists(auth_source_plugin_path):
        mod, _ = load_python_file(auth_source_plugin_path)
        return getattr(mod, "auth_source_plugin", None)
