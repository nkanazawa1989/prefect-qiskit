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
"""Test IBM specific implementation of runtime client.

This module is responsible for unit test of the client interface especially for some edge cases.
The end to end workflow with realistic HTTP responses is covered by test_e2e.py
"""

import asyncio

import aiohttp
import pytest
from prefect import flow, task
from prefect.futures import wait
from pytest_mock import MockerFixture

from prefect_qiskit.vendors.ibm_quantum import IBMQuantumCredentials


@pytest.mark.real_auth
async def test_http_session(
    ibm_secrets: IBMQuantumCredentials,
):
    """Test connection to real IBM Qiskit Runtime API."""

    # Not raises
    client = ibm_secrets.get_client()
    await client.get_resources()

    # Keep session alive
    session = await client._get_session()
    assert not session.closed


@pytest.mark.real_auth
async def test_http_async_parallel(
    ibm_secrets: IBMQuantumCredentials,
    mocker: MockerFixture,
):
    """Test async API calls reuse the same HTTP session."""
    spy = mocker.spy(aiohttp.ClientSession, "__init__")

    @task
    async def _run_many():
        return await ibm_secrets.get_client().get_resources()

    res1, res2, res3 = await asyncio.gather(_run_many(), _run_many(), _run_many())
    assert spy.call_count == 1
    assert res1 == res2 == res3


@pytest.mark.real_auth
async def test_http_thread_parallel(
    ibm_secrets: IBMQuantumCredentials,
    mocker: MockerFixture,
):
    """Test thread API calls create new HTTP session per thread."""
    spy = mocker.spy(aiohttp.ClientSession, "__init__")

    @task
    async def _run_many():
        return await ibm_secrets.get_client().get_resources()

    @flow
    async def _test_flow():
        futures = []
        for _ in range(3):
            fut = _run_many.submit()
            futures.append(fut)
        responses, _ = wait(futures)
        return responses

    res1, res2, res3 = await _test_flow()

    assert spy.call_count == 3
    assert await res1.result() == await res2.result() == await res3.result()


@pytest.mark.real_auth
async def test_http_mix_parallel(
    ibm_secrets: IBMQuantumCredentials,
    mocker: MockerFixture,
):
    """Test nested async API call inside thread task."""
    spy = mocker.spy(aiohttp.ClientSession, "__init__")

    @task
    async def _run_many():
        return await ibm_secrets.get_client().get_resources()

    @task
    async def _thread_task():
        return await asyncio.gather(_run_many(), _run_many(), _run_many())

    @flow
    async def _test_flow():
        futures = []
        for _ in range(3):
            fut = _thread_task.submit()
            futures.append(fut)
        responses, _ = wait(futures)
        return responses

    outer_responses = await _test_flow()

    # HTTP session is reused within thread
    assert spy.call_count == 3

    for inner_responses in outer_responses:
        results = await inner_responses.result()
        assert results[0] == results[1] == results[2]
