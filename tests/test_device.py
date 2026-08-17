import pytest

from wildlife_mlops.device import DeviceSelectionError, select_device


def test_select_device_prefers_mps_for_auto_when_available() -> None:
    assert select_device("auto", mps_available=True) == "mps"


def test_select_device_uses_cpu_for_auto_when_mps_is_unavailable() -> None:
    assert select_device("auto", mps_available=False, cuda_available=False) == "cpu"


def test_select_device_uses_cuda_when_mps_is_unavailable() -> None:
    assert select_device("auto", mps_available=False, cuda_available=True) == "cuda"


def test_select_device_honors_cpu_override() -> None:
    assert select_device("cpu", mps_available=True) == "cpu"


def test_select_device_honors_cuda_override() -> None:
    assert select_device("cuda", mps_available=True, cuda_available=True) == "cuda"


def test_select_device_rejects_unavailable_mps_override() -> None:
    with pytest.raises(DeviceSelectionError, match="explicitly requested"):
        select_device("mps", mps_available=False)


def test_select_device_rejects_unavailable_cuda_override() -> None:
    with pytest.raises(DeviceSelectionError, match="CUDA was explicitly requested"):
        select_device("cuda", mps_available=False, cuda_available=False)
