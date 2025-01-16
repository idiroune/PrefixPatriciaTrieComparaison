import os
import time
import cProfile
from memory_profiler import profile
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph


class PrefixTrie:
    def __init__(self):
        self.root = {}

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node["$"] = {}

    def search(self, word):
        node = self.root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return "$" in node

    def delete(self, word):
        stack = []
        node = self.root

        for char in word:
            if char not in node:
                return False
            stack.append((node, char))
            node = node[char]

        if "$" not in node:
            return False

        del node["$"]

        while stack:
            parent, char = stack.pop()
            if not node:
                del parent[char]
                node = parent
            else:
                break
        return True

    def range_search(self, prefix):
        node = self.root
        results = []

        for char in prefix:
            if char not in node:
                return results
            node = node[char]

        self._collect_words(node, prefix, results)
        return results

    def _collect_words(self, node, current_prefix, results):
        for char, child in node.items():
            if char == "$":
                results.append(current_prefix)
            else:
                self._collect_words(child, current_prefix + char, results)

    def visualize(self, filename, view=False):
        dot = Digraph()
        self._add_edges(self.root, dot, "root")

        directory = "Prefix_Trie"
        os.makedirs(directory, exist_ok=True)

        output_path = os.path.join(directory, f"{filename}")
        dot.render(output_path, view=view)
    def _add_edges(self, node, dot, parent_id):
        for char, child in node.items():
            child_id = f"{parent_id}_{char}"
            label = "<end>" if char == "$" else char
            dot.node(child_id, label=label)
            dot.edge(parent_id, child_id)
            self._add_edges(child, dot, child_id)

# Helper functions
def load_datasets_from_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    with open(file_path, 'r') as file:
        return np.array([line.strip() for line in file if line.strip()])


@profile
def test_trie_operations_with_profile(trie_class, cumulative_words, data_sizes):
    results = []
    trie = trie_class()

    for size in data_sizes:
        words = cumulative_words[:size]

        timings = {
            "insert": lambda: [trie.insert(word) for word in words],
            "search": lambda: [trie.search(word) for word in
                               np.random.choice(words, min(len(words), 1000), replace=False)],
            "range_search": lambda: [trie.range_search(word[:3]) for word in
                                     np.random.choice(words, min(len(words), 1000), replace=False)],
            "delete": lambda: [trie.delete(word) for word in
                               np.random.choice(words, min(len(words), 1000), replace=False)],
        }

        times = {operation: measure_time(func) for operation, func in timings.items()}
        results.append((size, *times.values()))

    return results

def measure_time(func):
    start = time.time()
    func()
    return time.time() - start


def plot_results(results, title):
    sizes = [res[0] for res in results]
    insert_times, search_times, range_search_times, delete_times = zip(*[res[1:] for res in results])

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, insert_times, label="Insert", marker='o')
    plt.plot(sizes, search_times, label="Search", marker='o')
    plt.plot(sizes, range_search_times, label="Range Search", marker='o')
    plt.plot(sizes, delete_times, label="Delete", marker='o')
    plt.xlabel("Dataset Size")
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


# Main
def main():
    print("Select mode:")
    print("1: Run performance experiments")
    print("2: Visualize and test operations on a Prefix Trie")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        data_sizes = [10, 100, 500, 1000]
        dataset_file = "datasets.txt"
        print(f"Loading datasets from '{dataset_file}'...")
        cumulative_words = load_datasets_from_file(dataset_file)

        print("\nTesting PrefixTrie...")
        prefix_results = test_trie_operations_with_profile(PrefixTrie, cumulative_words, data_sizes)

        print("\nResults (PrefixTrie):")
        for size, insert_time, search_time, range_search_time, delete_time in prefix_results:
            print(
                f"Size: {size}, Insert: {insert_time:.4f}s, Search: {search_time:.4f}s, "
                f"Range Search: {range_search_time:.4f}s, Delete: {delete_time:.4f}s"
            )

        plot_results(prefix_results, "Performance of Prefix Trie")

    elif choice == "2":
        trie = PrefixTrie()
        while True:
            print("\nOperations:")
            print("1: Insert a word")
            print("2: Search for a word")
            print("3: Search for words with a prefix (range search)")
            print("4: Delete a word")
            print("5: Exit")

            operation = input("Choose an operation: ").strip()

            if operation == "1":
                word = input("Enter word to insert: ").strip()
                trie.insert(word)
                print(f"Word '{word}' inserted.")
                trie.visualize("PrefixTrie", view=True)
                print(f"Prefix Trie visualization opened in browser")

            elif operation == "2":
                word = input("Enter word to search: ").strip()
                found = trie.search(word)
                print(f"Word '{word}' {'found' if found else 'not found'}.")

            elif operation == "3":
                prefix = input("Enter prefix for range search: ").strip()
                results = trie.range_search(prefix)
                print(f"Words with prefix '{prefix}': {results}")

            elif operation == "4":
                word = input("Enter word to delete: ").strip()
                deleted = trie.delete(word)
                print(f"Word '{word}' {'deleted' if deleted else 'not found'}.")
                if deleted:
                    trie.visualize("PrefixTrie", view=True)
                    print(f"Prefix Trie visualization opened in browser")

            elif operation == "5":
                print("Exiting...")
                break

            else:
                print("Invalid choice. Please try again.")

    else:
        print("Invalid choice. Exiting...")


if __name__ == "__main__":
    cProfile.run("main()")
