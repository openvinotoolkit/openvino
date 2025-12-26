import textwrap
from urllib.parse import urljoin
from conan.api.output import Color, ConanOutput
from conan.errors import ConanException


def _build_headers(token):
    return {"Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}"}


class ConanCenterProvider:
    def __init__(self, conan_api, name, provider_data):
        self.name = name
        self.url = provider_data["url"]
        self.type = provider_data["type"]
        self._token = provider_data.get("token")
        self._session = conan_api._api_helpers.requester  # noqa
        self._query_url = urljoin(self.url, "api/v1/query")

    def get_cves(self, refs):
        if not self._token:
            from conan.api.subapi.audit import CONAN_CENTER_AUDIT_PROVIDER_NAME
            if self.name == CONAN_CENTER_AUDIT_PROVIDER_NAME:
                output = ConanOutput()
                output.write("\n")
                output.write("Authentication required for the CVE provider: ", fg=Color.BRIGHT_RED,
                             newline=False)
                output.write(f"'{self.name}'\n", fg=Color.BRIGHT_WHITE)
                output.write("\nTo resolve, please:\n")
                output.write("  1. Visit: ", fg=Color.BRIGHT_WHITE, newline=False)
                output.write("https://audit.conan.io/register\n", fg=Color.BRIGHT_BLUE)
                output.write("  2. Register to obtain the access token and activate it.\n",
                             fg=Color.BRIGHT_WHITE)
                output.write("  3. Use the command below to authenticate:\n", fg=Color.BRIGHT_WHITE)
                output.write(f"\n     conan audit provider auth {self.name} --token=<your_token>",
                             fg=Color.BRIGHT_GREEN, newline=True)
                output.write("\nOnce authenticated, re-run the command.\n\n")

            raise ConanException("Missing authentication token. Please authenticate and retry.")

        result = {"data": {}, "error": None, "provider_url": None}

        for ref in refs:
            ConanOutput().info(f"Requesting vulnerability info for: {ref}")
            response = self._session.post(self._query_url, headers=_build_headers(self._token),
                                          json={"reference": str(ref)})
            if response.status_code == 200:
                result["data"][str(ref)] = response.json().get("data", {}).get("query", {})
            elif response.status_code == 400:
                ConanOutput().warning(f"Package '{ref}' not found.\n"
                                      f"Only libraries available in Conan Center can be queried for vulnerabilities.\n"
                                      f"Please ensure the package exists in the official repository: https://conan.io/center\n"
                                      f"If the package exists in the repository, please report it to conan-research@jfrog.com.\n")
                result["data"][str(ref)] = {"error": {"details": f"Package '{ref}' not scanned: Not found."}}
                continue
            elif response.status_code == 403:
                # TODO: How to report auth error to the user
                ConanOutput().error(f"Authentication error ({response.status_code}).\n"
                                    f"Your token may be invalid or not yet validated. If you recently registered, please check your email to validate your token.\n"
                                    f" - Set a valid token using: 'conan audit provider auth {self.name} --token=<your_token>'\n"
                                    f" - If you don’t have a token, register at: https://audit.conan.io/register")

                result["conan_error"] = f"Authentication error ({response.status_code})."
                break
            elif response.status_code == 429:
                reset_seconds = int(response.headers.get("retry-after", 0))
                reset_in_hours = reset_seconds // 3600
                reset_in_minutes = (reset_seconds % 3600) // 60

                output = ConanOutput()
                output.write("\n")

                if reset_in_hours > 0:
                    output.write(
                        f"You have exceeded the number of allowed requests. "
                        f"The limit will reset in {reset_in_hours} "
                        f"hour{'s' if reset_in_hours > 1 else ''} and {reset_in_minutes} "
                        f"minute{'s' if reset_in_minutes > 1 else ''}.\n",
                        fg=Color.BRIGHT_WHITE,
                    )
                else:
                    output.write(
                        f"You have exceeded the number of allowed requests. "
                        f"The limit will reset in {reset_in_minutes} "
                        f"minute{'s'if reset_in_minutes > 1 else ''}.\n",
                        fg=Color.BRIGHT_WHITE,
                    )
                output.write("Visit our website to learn more about JFrog's DevSecOps solution: ",
                             fg=Color.BRIGHT_WHITE, newline=False)
                output.write("https://audit.conan.io/limit", newline=True,
                             fg=Color.BRIGHT_BLUE)
                output.write("\n")

                result["conan_error"] = "Rate limit exceeded."
                break
            elif response.status_code == 500:
                # TODO: How to report internal server error to the user
                result["conan_error"] = f"Internal server error ({response.status_code})"
                break
            else:
                result["conan_error"] = f"Error in {ref} ({response.status_code})"
                break
        return result


class PrivateProvider:
    def __init__(self, conan_api, name, provider_data):
        self.name = name
        self.url = provider_data["url"]
        self.type = provider_data["type"]
        self._token = provider_data.get("token")
        self._session = conan_api._api_helpers.requester  # noqa
        self._query_url = urljoin(self.url, "catalog/api/v0/public/graphql")

    def get_cves(self, refs):
        if not self._token:
            raise ConanException(f"Missing authentication token for '{self.name}' provider.\n"
                                 f"Please authenticate using 'conan audit provider auth' and retry.")

        result = {"data": {}, "error": None, "provider_url": self.url}

        for ref in refs:
            try:
                response = self._get(ref)
                if "error" in response:
                    result["data"][str(ref)] = {"error": {"details": response["error"]}}
                else:
                    result["data"][str(ref)] = response.get("data", {}).get("query", {})
            except Exception as e:
                result["conan_error"] = str(e)
                break

        return result

    @staticmethod
    def _build_query(ref):
        name, version = ref.name, ref.version
        full_query = f"""query packageVersionDetails {{
            query: packageVersion(name: "{name}", type: "conan", version: "{version}") {{
                vulnerabilities(first: 100) {{
                    totalCount
                    edges {{
                        node {{
                            name
                            description
                            severity
                            cvss {{
                                preferredBaseScore
                            }}
                            aliases
                            withdrawn
                            publishedAt
                            advisories {{
                                name
                                ...on JfrogAdvisory {{
                                          name
                                          shortDescription
                                          severity
                                          impactReasons {{
                                                name
                                                isPositive
                                          }}
                                }}
                            }}
                            references
                            vulnerablePackages(first: 100) {{
                                totalCount
                                edges {{
                                     node {{
                                        fixVersions {{
                                            version
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}"""
        return full_query

    @staticmethod
    def _parse_error(errors, ref):
        """This function removes the errors array that comes from the catalog and returns a
        more user-friendly message if we know how to parse it, or a generic one if we don't
        find such one"""

        def _replace_message(message):
            if "not found" in message:
                return f"Package '{ref}' not scanned: Not found."
            return None

        error_msgs = filter(bool, [_replace_message(error["message"]) for error in errors])
        return next(error_msgs, "Unknown error")

    def _get(self, ref):
        full_query = self._build_query(ref)
        try:
            response = self._session.post(
                self._query_url,
                headers=_build_headers(self._token),
                json={
                    "query": textwrap.dedent(full_query)
                }
            )

            if response.status_code == 404:
                ConanOutput().write("\n")
                ConanOutput().warning(
                    f"An error occurred while connecting to the '{self.name}' provider.\n"
                    "This is likely because your JFrog Platform instance does not have JFrog Curation.\n"
                    "For more information, visit: https://audit.conan.io/missing-curation\n"
                )

            # Raises if some HTTP error was found
            response.raise_for_status()
        except Exception as e:
            raise e

        response_json = response.json()
        # filter the extensions key with graphql data
        response_json.pop('extensions', None)

        if "errors" in response_json:
            return {"error": self._parse_error(response_json["errors"], ref)}
        return response_json
