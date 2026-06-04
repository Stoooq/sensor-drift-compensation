from src.config import load_config


def main():
    cfg = load_config()
    print(cfg)
    print("Hello from sensor-drift-compensation!")


if __name__ == "__main__":
    main()
