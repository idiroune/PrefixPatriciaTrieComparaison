import os
import time
import cProfile
from memory_profiler import profile
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph


class PatriciaTrie:
    def __init__(self):
        self.root = {}

    def insert(self, key):
        node = self.root
        while key:
            for edge in list(node):
                prefix = self._common_prefix(edge, key)
                if prefix:
                    if prefix == edge:
                        key = key[len(prefix):]
                        node = node[edge]
                        break
                    else:
                        remaining_edge = edge[len(prefix):]
                        remaining_key = key[len(prefix):]

                        node[prefix] = {}
                        node[prefix][remaining_edge] = node.pop(edge)

                        if remaining_key:
                            node[prefix][remaining_key] = {"$": {}}
                        else:
                            node[prefix]["$"] = {}
                        return
            else:
                node[key] = {"$": {}}
                return

        if "$" not in node:
            node["$"] = {}

    def search(self, key):
        node = self.root
        while key:
            for edge in node:
                if key.startswith(edge):
                    key = key[len(edge):]
                    node = node[edge]
                    break
            else:
                return False
        return "$" in node

    def delete(self, key):
        stack = []
        node = self.root

        while key:
            for edge in node:
                if key.startswith(edge):
                    stack.append((node, edge))
                    key = key[len(edge):]
                    node = node[edge]
                    break
            else:
                return False

        if "$" in node:
            del node["$"]

        while stack:
            parent, edge_to_remove = stack.pop()

            if not node:
                del parent[edge_to_remove]
                node = parent
            elif len(node) == 1 and "$" not in node:
                child_edge, child_node = node.popitem()
                parent[edge_to_remove + child_edge] = child_node
                del parent[edge_to_remove]
                node = parent
            else:
                break
        return True

    def range_search(self, prefix):
        node = self.root
        results = []
        current_prefix = ""

        while prefix:
            for edge in node:
                if edge.startswith(prefix) or prefix.startswith(edge):
                    current_prefix += edge
                    prefix = prefix[len(edge):] if prefix.startswith(edge) else ""
                    node = node[edge]
                    break
            else:
                return results

        self._collect_keys(node, current_prefix, results)
        return results

    def _collect_keys(self, node, current_prefix, results):
        for edge, child in node.items():
            if edge == "$":
                results.append(current_prefix)
            else:
                self._collect_keys(child, current_prefix + edge, results)

    def visualize(self, filename, view=False):
        dot = Digraph()
        self._add_edges(self.root, dot, "root")

        directory = "Patricia_Trie"
        os.makedirs(directory, exist_ok=True)

        output_path = os.path.join(directory, f"{filename}")
        dot.render(output_path, view=view)

    def _add_edges(self, node, dot, parent_id):
        for edge, child in node.items():
            child_id = f"{parent_id}_{edge}"
            label = edge if edge != "$" else "<end>"
            dot.node(child_id, label=label)
            dot.edge(parent_id, child_id)
            self._add_edges(child, dot, child_id)

    @staticmethod
    def _common_prefix(s1, s2):
        min_length = min(len(s1), len(s2))
        for i in range(min_length):
            if s1[i] != s2[i]:
                return s1[:i]
        return s1[:min_length]


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
            "search": lambda: [trie.search(word) for word in np.random.choice(words, min(len(words), 1000), replace=False)],
            "range_search": lambda: [trie.range_search(word[:3]) for word in np.random.choice(words, min(len(words), 1000), replace=False)],
            "delete": lambda: [trie.delete(word) for word in np.random.choice(words, min(len(words), 1000), replace=False)],
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

def main():
    print("Select mode:")
    print("1: Run performance experiments with profiling")
    print("2: Visualize and test operations on a Patricia Trie")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        data_sizes = [10, 100, 500, 1000]
        dataset_file = "datasets.txt"
        print(f"Loading datasets.txt from '{dataset_file}'...")
        cumulative_words = load_datasets_from_file(dataset_file)

        print("\nTesting PatriciaTrie with profiling...")
        patricia_results = test_trie_operations_with_profile(PatriciaTrie, cumulative_words, data_sizes)

        print("\nResults (PatriciaTrie):")
        for size, insert_time, search_time, range_search_time, delete_time in patricia_results:
            print(
                f"Size: {size}, Insert: {insert_time:.4f}s, Search: {search_time:.4f}s, Range Search: {range_search_time:.4f}s, Delete: {delete_time:.4f}s")

        plot_results(patricia_results, "Performance of Patricia Trie")

    elif choice == "2":
        trie = PatriciaTrie()

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
                trie.visualize("PatriciaTrie", view=True)
                print(f"Patricia Trie visualization opened in browser")

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
                trie.visualize("PatriciaTrie", view=True)
                print(f"Patricia Trie visualization opened in browser")

            elif operation == "5":
                print("Exiting...")
                break

            else:
                print("Invalid choice. Please try again.")

    else:
        print("Invalid choice. Exiting...")


if __name__ == "__main__":
    cProfile.run("main()")
