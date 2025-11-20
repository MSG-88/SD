from __future__ import annotations

from unittest import mock

import pytest

from modules import civitai_client


def test_extracts_id_from_url():
    assert civitai_client._extract_model_id("https://civitai.com/models/12345") == "12345"


def test_fetch_model_metadata_uses_configured_base_url():
    cfg = civitai_client.CivitaiConfig(api_key="secret", base_url="https://example.test/api")
    fake_response = mock.Mock(status_code=200, json=mock.Mock(return_value={"id": 1}))
    with mock.patch("modules.civitai_client.requests.get", return_value=fake_response) as patched_get:
        data = civitai_client.fetch_model_metadata("42", config=cfg)

    patched_get.assert_called_once_with(
        "https://example.test/api/models/42",
        headers={"Authorization": "Bearer secret"},
        timeout=30,
    )
    assert data["id"] == 1


def test_download_model_streams_content(tmp_path):
    cfg = civitai_client.CivitaiConfig(api_key="", base_url="https://example.test/api")
    metadata = {
        "id": 99,
        "modelVersions": [
            {
                "files": [
                    {"name": "demo.safetensors", "downloadUrl": "https://example.test/file"},
                ]
            }
        ],
    }

    metadata_response = mock.Mock(status_code=200, json=mock.Mock(return_value=metadata))
    download_response = mock.Mock(status_code=200)
    download_response.iter_content.return_value = [b"abc", b"def"]

    with mock.patch("modules.civitai_client.fetch_model_metadata", return_value=metadata):
        with mock.patch("modules.civitai_client.requests.get", return_value=download_response):
            output = civitai_client.download_model("99", destination_dir=tmp_path, config=cfg)

    assert output.endswith("demo.safetensors")
    assert (tmp_path / "demo.safetensors").exists()
    assert (tmp_path / "demo.safetensors").read_bytes() == b"abcdef"


def test_download_model_errors_on_missing_file():
    metadata = {"modelVersions": [{"files": []}]}
    with mock.patch("modules.civitai_client.fetch_model_metadata", return_value=metadata):
        with pytest.raises(RuntimeError):
            civitai_client.download_model("oops", destination_dir=None)
