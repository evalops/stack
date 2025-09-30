"""
Test Hydra configuration loading
"""

from pathlib import Path

import yaml


def test_main_config():
    print("Testing main config loading...")
    config_path = Path("conf/config.yaml")
    assert config_path.exists(), "Main config not found"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert "defaults" in config
    assert "task" in config
    assert "seed" in config
    assert config["seed"] == 42
    print("✓ Main config loaded successfully")


def test_model_configs():
    print("\nTesting model configs...")
    model_configs = list(Path("conf/model").glob("*.yaml"))
    assert len(model_configs) > 0, "No model configs found"

    for config_file in model_configs:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        assert "name" in config
        print(f"✓ Model config {config_file.name} valid")


def test_data_configs():
    print("\nTesting data configs...")
    data_configs = list(Path("conf/data").glob("*.yaml"))
    assert len(data_configs) > 0, "No data configs found"

    for config_file in data_configs:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        assert "dataset_name" in config
        print(f"✓ Data config {config_file.name} valid")


def test_train_configs():
    print("\nTesting train configs...")
    train_configs = list(Path("conf/train").glob("*.yaml"))
    assert len(train_configs) > 0, "No train configs found"

    for config_file in train_configs:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        assert "epochs" in config
        assert "learning_rate" in config
        print(f"✓ Train config {config_file.name} valid")


def test_eval_configs():
    print("\nTesting eval configs...")
    eval_configs = list(Path("conf/eval").glob("*.yaml"))
    assert len(eval_configs) > 0, "No eval configs found"

    for config_file in eval_configs:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        assert "batch_size" in config
        print(f"✓ Eval config {config_file.name} valid")


def test_system_configs():
    print("\nTesting system configs...")
    system_configs = list(Path("conf/system").glob("*.yaml"))
    assert len(system_configs) > 0, "No system configs found"

    for config_file in system_configs:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        assert "device" in config
        print(f"✓ System config {config_file.name} valid")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Hydra Configuration Files")
    print("=" * 60)

    test_main_config()
    test_model_configs()
    test_data_configs()
    test_train_configs()
    test_eval_configs()
    test_system_configs()

    print("\n" + "=" * 60)
    print("✅ All Hydra config tests passed!")
    print("=" * 60)
