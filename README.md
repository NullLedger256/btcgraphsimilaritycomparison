# Behavioral Enhanced Analysis of Bitcoin Transaction Graphs



This approach provides a clear view of how these operational habits impact structural similarity scores compared to traditional baseline methods. It enables a more robust analysis by demonstrating how behavioral augmentation shifts similarity metrics, revealing connections and discrepancies that remain invisible when relying solely on graph topology.



##  Structure



### Data Extraction

* `graphextraction.py`: Automates the discovery and construction of Bitcoin transaction graphs by crawling the blockchain starting from a root address and moving outward to a specified depth.



### Similarity Analysis

* **`Baseline and Enhanced Similarity Methods/`**: Contains core implementations of traditional and advanced graph similarity algorithms.

* **`Graphs dataset dots/`**: Includes the raw graph data in `.dot` format, ready for processing and analysis.

* **`TEF baseline.py`**: A baseline implementation of Topological Feature Extraction (TFE) for graph similarity.

* **`TEF baseline against enhanced TEF summary.py`**: Summarizes the performance of various TFE methods (baseline, relay, neighbourhood, and split).

* **`TEF Fragmentation.py`**: Implements TFE focused on fragmentation and split-ratio metrics.

* **`TEF Neighbour activity.py`**: Analyzes the temporal neighborhood activity of nodes to compute similarity.

* **`TEF Relay.py`**: Computes similarity metrics based on temporal relay paths.

* **`WL baseline.py`**: A baseline implementation of the Weisfeiler-Lehman (WL) structural kernel similarity.

* **`WL baseline against enhanced WL summary.py`**: Summarizes the performance of various WL-based similarity methods.

* **`WL Fragmentation.py`**: Implements WL-based similarity focused on fragmentation/split-ratio metrics.

* **`WL Relay.py`**: Computes similarity metrics based on WL hashing applied to relay paths.

* **`WL Neighbour.py`**: Computes similarity metrics based on WL hashing applied to neighborhood activity.



---



## Prerequisites

To run the analysis and extraction scripts, you will need the following Python libraries:



* `networkx`: For graph construction and manipulation.

* `numpy`: For numerical and array operations.

* `pandas`: For data manipulation and reporting.

* `scipy`: For calculating similarity metrics (e.g., Euclidean, Cosine).

* `pydot` (or `pygraphviz`): For parsing/writing DOT files.

* `requests`: Required by the `graphextraction.py` script.



### Installation

```bash

pip install networkx numpy pandas scipy pydot requests
