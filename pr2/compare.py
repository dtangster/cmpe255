def load_labels(filename):
    with open(filename) as f:
        return list(
            filter(
                None,
                f.read().split('\n')
            )
        )

results = load_labels('test.txt')
labels = load_labels('test.labels')
correct = 0
incorrect = 0

for guess, real in zip(results, labels):
    if guess == real:
        correct += 1
    else:
        incorrect += 1

print('accuracy: %r', (100 * correct / len(results)))
