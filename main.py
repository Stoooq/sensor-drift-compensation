import numpy as np

from src.pipeline import InferencePipeline


def main():
    pipeline = InferencePipeline()
    dummy_input = np.random.rand(128)
    result = pipeline.predict(dummy_input)
    print(result)
    print("Hello from sensor-drift-compensation!")


if __name__ == "__main__":
    main()
