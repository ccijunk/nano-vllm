"""T3b · pytest fixtures。C1 fixture 生成器在 c1_fixture.py（与 LV2 loader Job 共用，单一事实源）。"""

import pytest

from c1_fixture import gen_weights, write_model_dir


@pytest.fixture(scope="session")
def c1_weights():
    return gen_weights()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    return write_model_dir(str(tmp_path_factory.mktemp("nano_model")))
