import pytest
from common import is_person, parse_label_line
from prep_sr import BUFFER, N_FOLDS, task_mappings


def test_labels():
    assert parse_label_line("2 0.5 0.4 0.2 0.3") == (2, [0.5, 0.4, 0.2, 0.3])
    cls, box = parse_label_line("1 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8")
    assert cls == 1
    assert box == pytest.approx([0.3, 0.5, 0.4, 0.6])
    with pytest.raises(ValueError):
        parse_label_line("not_a_class 0.1 0.2 0.3 0.4")
    assert is_person("Female_Sitting") and is_person("Male_Standing")
    assert not is_person("Window")


def test_mappings_and_rare_posture():
    names = ["Female_Sitting", "Male_Standing", "Male_Walking", "Window"]
    tasks = task_mappings(names)
    assert tasks["person"] == ({name: 0 for name in names[:3]}, ["person"])
    assert tasks["gender"] == ({names[0]: 0, names[1]: 1, names[2]: 1}, ["female", "male"])
    assert tasks["posture"][1] == ["sitting", "standing", "walking"]
    assert tasks["allclass"] == ({name: i for i, name in enumerate(names)}, names)
    assert "posture" not in task_mappings(["Female_Sitting", "Window"])
    assert N_FOLDS == 5 and BUFFER == {"indoor": 5, "outdoor": 10}
