import os
import time
import cProfile
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import profile
from Patricia_trie import PatriciaTrie
from Prefix_trie import PrefixTrie
from PyPDF2 import PdfMerger


def load_datasets_from_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")
    with open(file_path, 'r') as f:
        return np.array([line.strip() for line in f if line.strip()])

@profile
def compare_trie_operations(cumulative_words, data_sizes):
    patricia_results = test_trie_operations_with_profile(PatriciaTrie, cumulative_words, data_sizes)
    prefix_results = test_trie_operations_with_profile(PrefixTrie, cumulative_words, data_sizes)
    plot_comparison(patricia_results, prefix_results, "Patricia vs. Prefix Trie Performance")


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

def plot_comparison(results1, results2, title):
    sizes = [res[0] for res in results1]
    metrics1 = list(zip(*[res[1:] for res in results1]))
    metrics2 = list(zip(*[res[1:] for res in results2]))

    labels = ["Insert", "Search", "Range Search", "Delete"]
    colors = ["blue", "green", "orange", "red"]

    plt.figure(figsize=(10, 6))
    for i, label in enumerate(labels):
        plt.plot(sizes, metrics1[i], label=f"Patricia - {label}", marker='o', color=colors[i])
        plt.plot(sizes, metrics2[i], label=f"Prefix - {label}", linestyle="--", marker='o', color=colors[i])

    plt.xlabel("Dataset Size")
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()



def visualize_tries(patricia, prefix):

    patricia_folder = "Patricia_Trie"
    prefix_folder = "Prefix_Trie"

    os.makedirs(patricia_folder, exist_ok=True)
    os.makedirs(prefix_folder, exist_ok=True)

    patricia_filename = os.path.join(patricia_folder, "PatriciaTrie.pdf")
    prefix_filename = os.path.join(prefix_folder, "PrefixTrie.pdf")
    merged_filename = "CombinedTrieVisualization.pdf"

    patricia.visualize("PatriciaTrie", view=False)
    prefix.visualize("PrefixTrie", view=False)

    merge_pdfs(patricia_filename, prefix_filename, merged_filename)
    print("Both Trie visualization opened in browser")

    open_pdf(merged_filename)


def merge_pdfs(pdf1, pdf2, output_pdf):
    if not os.path.exists(pdf1) or not os.path.exists(pdf2):
        raise FileNotFoundError("One or both PDF files to merge are missing.")
    merger = PdfMerger()
    merger.append(pdf1)
    merger.append(pdf2)
    merger.write(output_pdf)
    merger.close()


def open_pdf(pdf_file):
    os.system(f"start {pdf_file}" if os.name == "nt" else f"open {pdf_file}")
# Main
def main():
    print("Select mode:")
    print("1: Run performance experiments")
    print("2: Visualize operations on Patricia and Prefix Tries")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        dataset_file = "datasets.txt"
        cumulative_words = load_datasets_from_file(dataset_file)
        data_sizes = [10, 100, 500, 1000]

        patricia_results = test_trie_operations_with_profile(PatriciaTrie, cumulative_words, data_sizes)

        print("\nResults (PatriciaTrie):")
        for size, insert_time, search_time, range_search_time, delete_time in patricia_results:
            print(
                f"Size: {size}, Insert: {insert_time:.4f}s, Search: {search_time:.4f}s, Range Search: {range_search_time:.4f}s, Delete: {delete_time:.4f}s")

        prefix_results = test_trie_operations_with_profile(PrefixTrie, cumulative_words, data_sizes)

        print("\nResults (PrefixTrie):")
        for size, insert_time, search_time, range_search_time, delete_time in prefix_results:
            print(
                f"Size: {size}, Insert: {insert_time:.4f}s, Search: {search_time:.4f}s, "
                f"Range Search: {range_search_time:.4f}s, Delete: {delete_time:.4f}s"
            )

        compare_trie_operations(cumulative_words, data_sizes)

    elif choice == "2":
        patricia = PatriciaTrie()
        prefix = PrefixTrie()

        while True:
            print("\nOperations:")
            print("1: Insert a word")
            print("2: Search for a word")
            print("3: Range search (prefix)")
            print("4: Delete a word")
            print("5: Exit")

            operation = input("Choose an operation: ").strip()

            if operation == "1":
                word = input("Enter word to insert: ").strip()
                patricia.insert(word)
                prefix.insert(word)
                print(f"Word '{word}' inserted in both tries.")
                visualize_tries(patricia, prefix)

            elif operation == "2":
                word = input("Enter word to search: ").strip()
                found_patricia = patricia.search(word)
                found_prefix = prefix.search(word)
                print(f"Patricia Trie: Word '{word}' {'found' if found_patricia else 'not found'}.")
                print(f"Prefix Trie: Word '{word}' {'found' if found_prefix else 'not found'}.")

            elif operation == "3":
                prefix_input = input("Enter prefix for range search: ").strip()
                results_patricia = patricia.range_search(prefix_input)
                results_prefix = prefix.range_search(prefix_input)
                print(f"Patricia Trie: Words with prefix '{prefix_input}': {results_patricia}")
                print(f"Prefix Trie: Words with prefix '{prefix_input}': {results_prefix}")

            elif operation == "4":
                word = input("Enter word to delete: ").strip()
                deleted_patricia = patricia.delete(word)
                deleted_prefix = prefix.delete(word)
                print(f"Patricia Trie: Word '{word}' {'deleted' if deleted_patricia else 'not found'}.")
                print(f"Prefix Trie: Word '{word}' {'deleted' if deleted_prefix else 'not found'}.")
                visualize_tries(patricia, prefix)

            elif operation == "5":
                print("Exiting...")
                break

            else:
                print("Invalid choice. Please try again.")

    else:
        print("Invalid choice. Exiting...")


if __name__ == "__main__":
    cProfile.run("main()")
