from wildlife_mlops.doctor import choose_device


def test_choose_device_prefers_mps_when_available() -> None:
    assert choose_device(mps_available=True) == "mps"
    assert choose_device(mps_available=False) == "cpu"
