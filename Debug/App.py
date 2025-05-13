import random

numbers = [str(random.randint(1, 10000000)) for _ in range(1000000)]

with open("input.txt", "w") as file:
    file.write(" ".join(numbers))
