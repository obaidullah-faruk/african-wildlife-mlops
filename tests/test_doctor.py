from wildlife_mlops.doctor import choose_device


def test_choose_device_prefers_mps_when_available() -> None:
    assert choose_device(mps_available=True, cuda_available=True) == "mps"
    assert choose_device(mps_available=False, cuda_available=True) == "cuda"
    assert choose_device(mps_available=False, cuda_available=False) == "cpu"
