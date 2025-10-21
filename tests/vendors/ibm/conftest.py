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
import os
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from ibm_cloud_sdk_core.token_managers import iam_token_manager

from prefect_qiskit.vendors import IBMQuantumCredentials
from prefect_qiskit.vendors.ibm_quantum.client import IBMQuantumPlatformClient


@pytest.fixture(autouse=True, scope="function")
def register_ibm_blocks() -> Generator[None, None, None]:
    """Fixture to add Prefect blocks for IBM Quantum integration."""
    IBMQuantumCredentials.register_type_and_schema()
    yield


@pytest.fixture(autouse=True)
def mock_iam_token(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Fixture to bypass real IBM cloud IAM authentication for testing."""
    if request.node.get_closest_marker("real_auth") is not None:
        return
    monkeypatch.setattr(
        iam_token_manager.IAMTokenManager,
        "get_token",
        MagicMock(return_value="mocked-token"),
    )


@pytest.fixture
def ibm_secrets() -> Generator[IBMQuantumCredentials, None, None]:
    """Fixture to get secrets from environment variable. Skip when not defined."""
    api_key = os.environ.get("IBM_API_KEY")
    crn = os.environ.get("IBM_CRN")

    if api_key is None:
        pytest.skip("Skipping test because IBM_API_KEY is not defined.")
    if crn is None:
        pytest.skip("Skipping test because IBM_CRN is not defined.")

    IBMQuantumPlatformClient._sessions.clear()

    try:
        yield IBMQuantumCredentials(api_key=api_key, crn=crn)
    finally:
        IBMQuantumPlatformClient._sessions.clear()
