import asyncio
from dotenv import load_dotenv
from src.test_runner import run_test_suite
from src.config import CONCURRENT

load_dotenv()


if __name__ == "__main__":
	asyncio.run(run_test_suite(concurrent=CONCURRENT))
