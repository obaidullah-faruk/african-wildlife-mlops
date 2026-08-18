from wildlife_mlops.storage_verify import StorageVerificationError, parse_postgres_rows


def test_parse_postgres_rows_keeps_metadata_evidence() -> None:
    rows = parse_postgres_rows("metric|verification.artifact_bytes|1048576\nparam|source|storage\n")

    assert rows == [
        {"record_type": "metric", "key": "verification.artifact_bytes", "value": "1048576"},
        {"record_type": "param", "key": "source", "value": "storage"},
    ]


def test_parse_postgres_rows_rejects_unexpected_output() -> None:
    try:
        parse_postgres_rows("not a database row\n")
    except StorageVerificationError as error:
        assert "Unexpected PostgreSQL output" in str(error)
    else:
        raise AssertionError("Expected StorageVerificationError")
