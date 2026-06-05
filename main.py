from pathlib import Path

from src.config import load_config
from src.data_loader import parse_data


def main():
    cfg = load_config()
    parse_data(Path("data/"))
    print(cfg)
    print("Hello from sensor-drift-compensation!")


if __name__ == "__main__":
    main()
