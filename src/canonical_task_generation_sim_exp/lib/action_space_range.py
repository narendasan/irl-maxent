from typing import Generator

def complex_action_space_range(start: int, end: int) -> Generator[int, None, None]:
    n = start
    while n <= end:
        print(n)
        yield n
        n += int(start * 0.25)


if __name__ == "__main__":
    for i in complex_action_space_range(10, 3):
        print(i)
