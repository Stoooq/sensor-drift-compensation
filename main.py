from pathlib import Path

from src.config import load_config
from src.data_loader import load_data


def main():
    cfg = load_config()
    data = load_data(Path("data/"))
    print(data[:3])
    print("Hello from sensor-drift-compensation!")


if __name__ == "__main__":
    main()
