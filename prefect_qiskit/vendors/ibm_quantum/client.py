# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.
"""Adaptor for Qiskit IBM Runtime Client.

See the following link for the REST API specification.
https://quantum.cloud.ibm.com/docs/en/api/qiskit-runtime-rest#qiskit-runtime-rest-api
"""

import asyncio
import functools
import json
import threading
from datetime import datetime
from typing import Any, Literal

import aiohttp
from cachetools import LRUCache
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from pydantic import SecretStr, ValidationError
from qiskit.primitives.containers import PrimitiveResult
from qiskit.primitives.containers.estimator_pub import EstimatorPub
from qiskit.primitives.containers.sampler_pub import SamplerPub
from qiskit.transpiler.target import Target
from qiskit_ibm_runtime.models import BackendConfiguration, BackendProperties
from qiskit_ibm_runtime.utils.backend_converter import convert_to_target
from qiskit_ibm_runtime.utils.json import RuntimeEncoder
from qiskit_ibm_runtime.utils.result_decoder import ResultDecoder

from prefect_qiskit.exceptions import RuntimeJobFailure
from prefect_qiskit.models import JOB_STATUS, JobMetrics
from prefect_qiskit.utils.logging import LoggingMixin
from prefect_qiskit.vendors.ibm_quantum.models import EstimatorV2Schema, SamplerV2Schema

DEFAULT_AUTH_URL: str = "https://iam.cloud.ibm.com"
DEFAULT_RUNTIME_URL: str = "https://quantum.cloud.ibm.com/api/v1/"


def handle_error(method):
    """A method decorator to prevent accidental secrets print out and error typecast."""

    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        try:
            return await method(self, *args, **kwargs)
        except aiohttp.ClientResponseError as ex:
            raise RuntimeJobFailure(
                reason=f"HTTP {ex.status} on {ex.request_info.url!r}.",
                retry=True,
            ) from None
        except aiohttp.ClientConnectionError:
            raise RuntimeJobFailure(
                reason=f"Connection failed to {self.runtime_endpoint_url!r}.",
                retry=True,
            ) from None
        except TimeoutError:
            raise RuntimeJobFailure(
                reason="HTTP request timed out.",
                retry=True,
            ) from None
        except aiohttp.ClientError as ex:
            raise RuntimeJobFailure(
                reason=f"General HTTP client error {ex.__class__.__name__}.",
                retry=True,
            ) from None

    return wrapper


class SessionCache(LRUCache):
    """LRU cache with session closing at expire."""

    def popitem(self):
        """Remove session with termination of removed session."""
        endpoint, session = super().popitem()
        if not session.closed:
            asyncio.run(session.close())
        return endpoint, session


class IBMQuantumPlatformClient(LoggingMixin):
    """Adaptor interface for IBM Quantum Platform cloud client.

    .. note::
        This class maintains a **class-level cache** of HTTP sessions, keyed by endpoint URL.
        The cache enables efficient reuse of TCP connections for frequent HTTP requests.

        By design, each cached session keeps its underlying TCP socket open
        for the duration of the TCP keep-alive cycle, rather than closing it after each request.

        When this class is initialized across multiple processes (for example, via
        :class:`concurrent.futures.ProcessPoolExecutor`), each process establishes its
        own socket per endpoint—even if no HTTP request is made.

        Because this class implements :class:`AsyncRuntimeClientInterface`, it assumes
        that concurrent operations are coordinated by **asyncio**, rather than by
        multiprocessing or distributed task runners.

    """

    AVOID_RETRY = [9999]

    _sessions = SessionCache(maxsize=3)
    _lock = threading.Lock()

    def __init__(
        self,
        api_key: SecretStr,
        crn: str,
        auth_endpoint_url: str = DEFAULT_AUTH_URL,
        runtime_endpoint_url: str = DEFAULT_RUNTIME_URL,
    ):
        """Create new client.

        Args:
            api_key: API key to access cloud platform.
            crn: Cloud resource name to identify the service to use.
            auth_endpoint_url: Endpoint URL for IAM authentication service.
            api_endpoint_url: Endpoint URL for IBM Qiskit Runtime REST API.
        """
        self.auth = IAMAuthenticator(
            apikey=api_key.get_secret_value(),
            url=auth_endpoint_url,
        )
        self.crn = crn
        self.runtime_endpoint_url = runtime_endpoint_url

    def __repr__(self):
        return f"<{self.__class__.__name__} runtime_endpoint_url={self.runtime_endpoint_url!r}>"

    def _get_session(
        self,
    ) -> aiohttp.ClientSession:
        # Lock is necessary because Prefect tasks calling this client
        # might be run by the ThreadPoolTaskRunner. In this case,
        # race condition may occur and multiple HTTP sessions are initialized for the same key.
        # This eventually results in the socket leakage.
        with self._lock:
            session = self._sessions.get(self.runtime_endpoint_url)
            if session is None or session.closed:
                self.logger.debug(f"Creating new HTTP session for {self.runtime_endpoint_url!r}")
                session = aiohttp.ClientSession(
                    base_url=self.runtime_endpoint_url,
                    timeout=aiohttp.ClientTimeout(30),
                    raise_for_status=True,
                )
                self._sessions[self.runtime_endpoint_url] = session
        return session

    def _get_headers(
        self,
    ) -> dict[str, str]:
        headers = {
            "IBM-API-Version": "2025-01-01",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.auth.token_manager.get_token()}",
            "Service-CRN": self.crn,
        }
        return headers

    @handle_error
    async def check_resource_available(
        self,
        resource_name: str,
    ) -> bool:
        session = self._get_session()
        async with session.get(
            f"backends/{resource_name}/status",
            headers=self._get_headers(),
        ) as resp:
            self.logger.debug(f"GET request for {resp.url}")
            ret = await resp.json()
        return ret.get("state", False)

    @handle_error
    async def get_resources(
        self,
    ) -> list[str]:
        session = self._get_session()
        async with session.get(
            "backends",
            headers=self._get_headers(),
        ) as resp:
            self.logger.debug(f"GET request for {resp.url}")
            ret = await resp.json()
        devices = ret.get("devices", [])
        return [d["name"] for d in devices if d["status"]["name"] == "online"]

    @handle_error
    async def get_target(
        self,
        resource_name: str,
    ) -> Target:
        session = self._get_session()
        headers = self._get_headers()
        async with session.get(
            f"backends/{resource_name}/configuration",
            headers=headers,
        ) as resp:
            self.logger.debug(f"GET request for {resp.url}")
            configuration_dict = await resp.json()
        async with session.get(
            f"backends/{resource_name}/properties",
            headers=headers,
        ) as resp:
            self.logger.debug(f"GET request for {resp.url}")
            properties_dict = await resp.json()
        return convert_to_target(
            configuration=BackendConfiguration.from_dict(configuration_dict),
            properties=BackendProperties.from_dict(properties_dict),
        )

    @handle_error
    async def run_primitive(
        self,
        program_id: Literal["sampler", "estimator"],
        inputs: list[SamplerPub] | list[EstimatorPub],
        resource_name: str,
        options: dict[str, Any],
    ) -> str:
        payload = {
            "program_id": None,
            "backend": resource_name,
            **options,
        }
        if "params" in payload:
            params = payload.pop("params").copy()
        else:
            params = {}
        params.update(
            {
                "support_qiskit": True,
                "version": 2,
            }
        )
        match program_id:
            case "sampler":
                params["pubs"] = [
                    [
                        pub.circuit,
                        pub.parameter_values.as_array(pub.circuit.parameters),
                        pub.shots,
                    ]
                    for pub in inputs
                ]
                try:
                    SamplerV2Schema.model_validate(params)
                except ValidationError as ex:
                    raise RuntimeJobFailure(
                        "Primitive input doesn't match the data schema for Runtime REST API.",
                        retry=False,
                    ) from ex
                payload["program_id"] = "sampler"
                payload["params"] = params
            case "estimator":
                params["pubs"] = [
                    [
                        pub.circuit,
                        pub.observables.tolist(),
                        pub.parameter_values.as_array(pub.circuit.parameters),
                        pub.precision,
                    ]
                    for pub in inputs
                ]
                try:
                    EstimatorV2Schema.model_validate(params)
                except ValidationError as ex:
                    raise RuntimeJobFailure(
                        "Primitive input doesn't match the data schema for Runtime REST API.",
                        retry=False,
                    ) from ex
                payload["program_id"] = "estimator"
                payload["params"] = params
            case _:
                raise Exception("Unreachable")
        self.logger.debug(f"Submitting the following payload: {payload}")
        data = json.dumps(payload, cls=RuntimeEncoder)

        session = self._get_session()
        async with session.post(
            "jobs",
            data=data,
            headers=self._get_headers(),
            timeout=aiohttp.ClientTimeout(900),
        ) as resp:
            self.logger.debug(f"POST request for {resp.url}")
            ret = await resp.json()
        if job_id := ret.get("id", None):
            self.logger.info(f"Job started with job ID {job_id}.")
        else:
            raise RuntimeJobFailure(
                reason="Server didn't return Job ID.",
                retry=True,
            )
        return job_id

    @handle_error
    async def get_primitive_result(
        self,
        job_id: str,
    ) -> PrimitiveResult:
        session = self._get_session()
        async with session.get(
            f"jobs/{job_id}/results",
            headers=self._get_headers(),
        ) as resp:
            self.logger.debug(f"GET request for {resp.url}")
            ret = await resp.text()
        # Assume job ID is valid.
        # This is true as long as job is not exposed to user program.
        results = ResultDecoder.decode(ret)

        if spans := results.metadata.pop("execution", {}).pop("execution_spans", {}):
            # Remove execution span object
            # Add a simple dictionary instead to avoid IBM specific contexts
            for res, span in zip(results, spans):
                span_info = {
                    "timestamp_start": span.start.isoformat(),
                    "timestamp_completed": span.stop.isoformat(),
                    "duration": span.duration,
                }
                res.metadata["span"] = span_info
        return results

    @handle_error
    async def get_job_status(
        self,
        job_id: str,
    ) -> JOB_STATUS:
        session = self._get_session()
        async with session.get(
            f"jobs/{job_id}",
            headers=self._get_headers(),
        ) as resp:
            self.logger.debug(f"GET request for {resp.url}")
            ret = await resp.json()
        job_state = ret.get("state", {})

        match status := ret.get("status", "unknown").upper():
            case "QUEUED" | "RUNNING" | "COMPLETED":
                return status
            case "CANCELLED - RAN TOO LONG":
                raise RuntimeJobFailure(
                    reason="Job ran longer than maximum execution time.",
                    job_id=job_id,
                    error_code=job_state.get("reason_code", None),
                    retry=True,
                )
            case "CANCELLED":
                raise RuntimeJobFailure(
                    reason="Job is manually cancelled.",
                    job_id=job_id,
                    retry=False,
                )
            case "FAILED":
                msg_parts = [f"Job {job_id} failed"]
                if reason := job_state.get("reason", None):
                    msg_parts.append(f"The following causes were reported: {reason}")
                if sol := job_state.get("reason_solution", None):
                    msg_parts.append(f"Please consider suggested solution: {sol}")
                msg = ". ".join(msg_parts)
                if not msg.endswith("."):
                    msg += "."
                code = job_state.get("reason_code", None)
                raise RuntimeJobFailure(
                    reason=msg,
                    job_id=job_id,
                    error_code=code,
                    retry=code is not None and code not in self.AVOID_RETRY,
                )
            case _:
                raise RuntimeJobFailure(
                    reason=f"Unknown job status reported '{status}'.",
                    job_id=job_id,
                    retry=True,
                )

    @handle_error
    async def get_job_metrics(
        self,
        job_id: str,
    ) -> JobMetrics:
        session = self._get_session()
        async with session.get(
            f"jobs/{job_id}/metrics",
            headers=self._get_headers(),
        ) as resp:
            self.logger.debug(f"GET request for {resp.url}")
            ret = await resp.json()

        if "timestamps" in ret:
            try:
                created_dt = datetime.fromisoformat(ret["timestamps"].get("created", None))
            except (ValueError, TypeError):
                created_dt = None
            try:
                running_dt = datetime.fromisoformat(ret["timestamps"].get("running", None))
            except (ValueError, TypeError):
                running_dt = None
            try:
                finished_dt = datetime.fromisoformat(ret["timestamps"].get("finished", None))
            except (ValueError, TypeError):
                finished_dt = None

        return JobMetrics(
            qpu_usage=ret.get("usage", {}).get("quantum_seconds", None),
            timestamp_created=created_dt,
            timestamp_started=running_dt,
            timestamp_completed=finished_dt,
        )
