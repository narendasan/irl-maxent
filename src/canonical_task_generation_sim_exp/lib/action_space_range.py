def complex_action_space_range(start: int, end: int) -> int:
    n = start
    while n <= end:
        yield n
        n += int(start * 0.5)


if __name__ == "__main__":
    for i in complex_action_space_range(10, 3):
        print(i)