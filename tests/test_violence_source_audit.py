import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "src/aitf_cv/tasks/violence/audit_sources.py"
spec = importlib.util.spec_from_file_location("violence_source_audit", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_audit_flags_ambiguous_missing_and_duplicate_content(tmp_path):
    (tmp_path / "fight").mkdir()
    (tmp_path / "noFight").mkdir()
    (tmp_path / "fight/fi001.mp4").write_bytes(b"same")
    (tmp_path / "noFight/nofi001.mp4").write_bytes(b"same")
    (tmp_path / "noFight/nofi002.mp4").write_bytes(b"different")
    (tmp_path / "videos.txt").write_text(
        "https://example.org/a\nfi001: 0 - 1\nnofi001: 1 - 2\n"
        "https://example.org/b\nnofi001: 2 - 3\nnofi002 - nofi003 : other dataset\n"
    )
    result = module.audit(tmp_path)
    assert result["clips"] == 3
    assert result["uniquely_mapped_clips"] == 1
    assert result["unmapped_clip_ids"] == ["nofi002"]
    assert result["ambiguous_clip_sources"]["nofi001"] == ["https://example.org/a", "https://example.org/b"]
    assert len(result["identical_file_groups"]) == 1
    assert result["unparsed_index_lines"] == ["nofi002 - nofi003 : other dataset"]


def test_empty_input_fails_clearly(tmp_path):
    (tmp_path / "videos.txt").write_text("")
    with pytest.raises(ValueError, match="No MP4"):
        module.audit(tmp_path)
