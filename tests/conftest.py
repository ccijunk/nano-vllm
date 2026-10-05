"""T3b · pytest fixtures。C1 fixture 生成器在 c1_fixture.py（与 LV2 loader Job 共用，单一事实源）。"""

import pytest

from c1_fixture import gen_weights, write_model_dir


@pytest.fixture(scope="session")
def c1_weights():
    return gen_weights()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    return write_model_dir(str(tmp_path_factory.mktemp("nano_model")))


@pytest.fixture(scope="session")
def swa_model_dir(tmp_path_factory):
    # T3d 混合模型（design §2.4）：layer 0 = 全注意力，layer 1 = SWA(W=256=block_size)；
    # 无新模型类——NanoForCausalLM 按 config.sliding_window_layers 驱动层→池映射
    return write_model_dir(str(tmp_path_factory.mktemp("nano_model_swa")),
                           swa={"sliding_window": 256, "sliding_window_layers": [1]})
