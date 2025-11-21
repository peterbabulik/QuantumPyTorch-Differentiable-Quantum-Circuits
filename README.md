# QuantumPyTorch: A Native PyTorch Framework for Quantum Simulation
### From Differentiable Circuits to Quantum Chaos and Algorithms

This repository introduces **QuantumPyTorch**, a proof-of-concept framework for building, simulating, and training quantum circuits directly within the PyTorch ecosystem. By representing quantum states and operators as native `torch.Tensors`, we leverage the full power of PyTorch's GPU acceleration and automatic differentiation (`autograd`) engine for a wide range of quantum simulations, without relying on external quantum computing libraries.

This document serves as the primary paper and guide for the project, showcasing its versatility across six distinct domains:
1.  **Quantum Machine Learning (QML):** Creating differentiable quantum circuits that function as trainable layers in an ML pipeline.
2.  **Quantum-Inspired Optimization:** Using a variational quantum algorithm (VQE) to solve a classical optimization problem (Neural Architecture Search).
3.  **Quantum Physics Simulation:** Modeling the complex dynamics of quantum chaos and information scrambling by simulating Out-of-Time-Order Correlators (OTOCs).
4.  **General-Purpose Quantum Algorithms:** Building a canonical quantum algorithm (Grover's Search) from a "zoo" of fundamental gates to test its performance under ideal and noisy conditions.
5.  **Quantum Random Number Generation (QRNG):** A comparative analysis of QRNG circuits, demonstrating massive parallel generation and a ~56x speedup over standard simulations.
6.  **Quantum Cellular Automata & Walks:** Simulating time-evolution and propagation in 1D/2D systems, observing light-cones and decoherence.

---

## Abstract

The confluence of quantum computing and machine learning has opened new paradigms for computational science. However, the integration between these domains often relies on specialized intermediary software. This paper introduces "QuantumPyTorch," an approach for the direct implementation of quantum circuits and algorithms within PyTorch. By representing quantum states as tensors and gates as differentiable matrix operations, we demonstrate the viability of this approach through a suite of increasingly complex applications. These range from a Variational Quantum Classifier (VQC) and a VQE-based optimizer for Neural Architecture Search, to advanced simulations of quantum chaos via OTOCs and the implementation of Grover's search algorithm. Our results showcase the simplicity, flexibility, and power of a seamlessly integrated quantum-classical workflow capable of exploring not just QML, but also fundamental quantum dynamics and algorithm performance, all entirely within a classical deep learning framework.

## 1. Introduction
Quantum computing is poised to revolutionize numerous fields by leveraging the principles of quantum mechanics. A particularly promising area is Quantum Machine Learning (QML), which often involves hybrid quantum-classical models orchestrated by libraries like PyTorch or TensorFlow. This has led to the development of powerful software like Qiskit, Cirq, and PennyLane, which provide high-level abstractions for quantum circuits.

This project explores an alternative: the direct and native implementation of quantum mechanics within PyTorch. We present **QuantumPyTorch**, an approach that treats quantum statevectors and operators as fundamental `torch.Tensor` objects. This allows quantum circuits to be treated as standard `torch.nn.Module` objects, fully compatible with PyTorch's ecosystem.

We posit that this direct integration offers several advantages:
*   **Simplicity:** An intuitive model for researchers already proficient in PyTorch.
*   **Flexibility:** Quantum circuits can be seamlessly embedded within larger classical deep learning architectures.
*   **Performance:** Directly leverages PyTorch's highly optimized backend for tensor operations, including massive parallelization on GPUs.

To validate this approach, we present a suite of six case studies that demonstrate the framework's expanding capabilities: a Variational Quantum Classifier, a `VQEOptimizer` for a Neural Architecture Search (NAS) problem, a simulation of quantum chaos via Out-of-Time-Order Correlators (OTOCs), a from-scratch implementation of Grover's Search Algorithm, a high-performance Quantum Random Number Generator (QRNG), and a study of Quantum Cellular Automata and Quantum Walks.

## 2. The QuantumPyTorch Methodology
The core principle of QuantumPyTorch is to represent all quantum mechanical concepts using PyTorch's native `torch.Tensor` class.

### 2.1. Statevectors and Operators
A quantum statevector for *n* qubits is represented as a 1D tensor of size 2<sup>n</sup>. Quantum gates are represented as square matrices (2D tensors). A gate acting on a specific qubit is constructed by taking the tensor product (`torch.kron`) of the gate matrix with identity matrices for all other qubits.

### 2.2. Differentiable Circuits
A parameterized quantum circuit is a sequence of matrix-vector multiplications. Since the parameters are part of PyTorch's computation graph, the entire circuit is differentiable, allowing for gradient-based optimization.

### 2.3. Noise Models
To simulate realistic, near-term quantum devices, we introduce noise as a probabilistic mathematical operation. After a gate is applied, a probabilistic function determines if a random error (e.g., a Pauli X or Z gate) is applied to the statevector. This allows us to study the effect of decoherence on our simulations.

### 2.4. Measurement
The framework supports two types of measurement depending on the task:
1.  **Expectation Value:** For QML and VQE, we calculate the expectation value `⟨ψ|H|ψ⟩` of an observable `H`, which yields a continuous output for optimization.
2.  **Projective Measurement:** For algorithms like Grover's, we simulate a final measurement by calculating the probability of collapsing to each basis state, given by `|⟨i|ψ⟩|^2` for each basis state `|i⟩`.

## 3. Experiments and Results
We validate the framework with six distinct experiments, each showcasing a different capability.

### 3.1. Experiment 1: Quantum-Enhanced Classification
We build a variational quantum classifier to solve the `make_circles` binary classification problem. The model uses classical data to encode angles in a parameterized quantum circuit, which is then trained using standard PyTorch optimizers to classify the data, achieving a **96.67% test accuracy**. This demonstrates the core QML functionality.

### 3.2. Experiment 2: VQEOptimizer for Neural Architecture Search
We repurpose the Variational Quantum Eigensolver (VQE) as a general-purpose optimizer for a classical Neural Architecture Search (NAS) problem. By encoding 16 transformer architectures into a 4-qubit Hamiltonian, the `VQEOptimizer` successfully identifies the optimal architecture by finding the ground state, demonstrating a novel application of quantum-inspired optimization.

### 3.3. Experiment 3: Simulating Quantum Dynamics and Chaos (OTOCs)
To prove the framework's power for physics simulation, we calculate Out-of-Time-Order Correlators (OTOCs), a key diagnostic for quantum chaos and information scrambling. We simulate a 6-qubit Ising chain and show how noise systematically suppresses scrambling. The plot below shows the OTOC for an ideal system versus the averaged results for systems with increasing levels of noise, perfectly capturing the effects of decoherence.

[Effect of Noise on Information Scrambling](https://github.com/peterbabulik/QuantumPyTorch-Differentiable-Quantum-Circuits/blob/main/QuantumPyTorch_OTOCs.ipynb)

This result demonstrates that `QuantumPyTorch` is a powerful tool for computational physics research, allowing for the study of complex, many-body quantum phenomena.

### 3.4. Experiment 4: General-Purpose Algorithm Simulation (Grover's Search)
To showcase the framework's utility as a general-purpose quantum computer simulator, we built a "zoo" of fundamental quantum gates (H, X, Z, S, T, CNOT, MCZ) and used them to implement Grover's Search Algorithm from scratch. We tasked the algorithm with finding the `|11>` state in a 2-qubit search space. The results below compare the ideal performance to a noisy simulation (p=0.1 error rate).

[Grover's Algorithm Performance](https://github.com/peterbabulik/QuantumPyTorch-Differentiable-Quantum-Circuits/blob/main/QuantumPyTorchGates.ipynb)

The ideal simulation finds the correct answer with 100% probability. In contrast, the noisy simulation's success rate plummets to 68%, with the remaining 32% of probability leaking into incorrect answers. This powerfully visualizes how noise can cause quantum algorithms to fail and validates the framework's use for studying algorithmic performance and fault tolerance.

### 3.5. Experiment 5: Quantum Random Number Generation (QRNG) Analysis
We perform a comparative analysis of various Quantum Random Number Generator (QRNG) circuit designs (Basic Hadamard, Rotated, Entangled, and Parallel) to evaluate their statistical randomness and performance. The study includes generating massive 1024-bit random numbers and subjecting the output to rigorous statistical tests (Chi-Square, Shannon Entropy, Autocorrelation). We also demonstrate a **~56x speedup** over a standard `cirq` implementation by leveraging `QuantumPyTorch`'s batch processing and GPU acceleration.

[Quantum Random Number Generation Analysis](https://github.com/peterbabulik/QuantumPyTorch-Differentiable-Quantum-Circuits/blob/fa95e2e81167327f840c55a42b5812175407fcf8/Qantum_random_number_circuits2.ipynb)

### 3.6. Experiment 6: Distribution of Propagation (QCA and Quantum Walks)
We validated the framework's capability to simulate time-evolving quantum systems, focusing on how an initial state propagates and distributes itself over time. This includes 1D and 2D Quantum Cellular Automata (QCA) simulations to observe light-cone propagation and entanglement growth, as well as Quantum Walks that demonstrate the transition from ballistic quantum spreading to classical diffusion under noise.

[Distribution of Propagation Experiments](https://github.com/peterbabulik/QuantumPyTorch-Differentiable-Quantum-Circuits/blob/main/DistributionTests.ipynb)

## 4. Discussion
The success of these six experiments validates the thesis that it is practical and powerful to implement quantum simulations directly and natively in PyTorch.

*   **Advantages:** The primary advantage is the seamless integration into a mature deep learning ecosystem, leveraging GPU acceleration and a vast library of existing tools. The framework is flexible, intuitive for those familiar with PyTorch, and now includes a robust noise model.
*   **Limitations:** The statevector simulation approach is memory-intensive, scaling exponentially (O(2<sup>n</sup>)) with the number of qubits *n*. This restricts simulations to a moderate number of qubits (typically < 30). The framework also does not connect to real quantum hardware.

## 5. Conclusion
`QuantumPyTorch` has been successfully demonstrated as a versatile methodology for building, simulating, and training quantum circuits directly within PyTorch. We have validated its effectiveness across six distinct domains: quantum machine learning, quantum-inspired optimization, quantum chaos simulation, quantum algorithm analysis, high-performance random number generation, and spatiotemporal quantum propagation. This approach simplifies the quantum-classical workflow and provides a powerful platform for research, prototyping, and education. By lowering the barrier to entry, a direct, tensor-based approach can foster greater cross-pollination between the deep learning and quantum computing communities.

## 6. Code Availability
The code for all experiments is publicly available in this repository.

*   **Experiment 1 & 2 (QML and VQE-NAS):** The original notebooks for the VQC and VQE optimizer.
    *   [QuILT.ipynb](https://github.com/peterbabulik/QuILT/blob/main/QuILT.ipynb)
    *   [QuILT_NAS.ipynb](https://github.com/peterbabulik/QuILT/blob/main/QuILT_NAS.ipynb)

*   **Experiment 3 (OTOCs and Quantum Chaos):** The Jupyter notebook containing the simulation of information scrambling in an Ising chain, with and without noise.
    *   [QuantumPyTorch_OTOCs.ipynb](https://github.com/peterbabulik/QuantumPyTorch-Differentiable-Quantum-Circuits/blob/main/QuantumPyTorch_OTOCs.ipynb)

*   **Experiment 4 (Gates and Grover's Algorithm):** The Jupyter notebook containing the "zoo" of quantum gates and the implementation of Grover's Search under ideal and noisy conditions.
    *   [QuantumPyTorchGates.ipynb](https://github.com/peterbabulik/QuantumPyTorch-Differentiable-Quantum-Circuits/blob/main/QuantumPyTorchGates.ipynb)

*   **Experiment 5 (QRNG Analysis):** The Jupyter notebook containing the comparative analysis of QRNG circuits and statistical tests.
    *   [Quantum_random_number_circuits2.ipynb](https://github.com/peterbabulik/QuantumPyTorch-Differentiable-Quantum-Circuits/blob/main/Qantum_random_number_circuits2.ipynb)

*   **Experiment 6 (QCA and Quantum Walks):** The Jupyter notebook containing the "Distribution of Propagation" experiments.
    *   [DistributionTests.ipynb](https://github.com/peterbabulik/QuantumPyTorch-Differentiable-Quantum-Circuits/blob/main/DistributionTests.ipynb)


## Original Paper for:

# **QuantumPyTorch: Direct Integration of Quantum Computing in PyTorch for Optimization and Machine Learning**

## Abstract

The confluence of quantum computing and machine learning has opened new paradigms for computational science. However, the integration between these domains often relies on specialized software that acts as an intermediary between quantum simulators and classical machine learning frameworks. This paper introduces "QuantumPyTorch," a proof-of-concept approach for the direct implementation of quantum circuits and algorithms within the PyTorch ecosystem, foregoing the need for external quantum computing libraries. By representing quantum states as PyTorch tensors and quantum gates as differentiable matrix operations, we harness the full power of PyTorch's autograd engine and GPU acceleration for quantum simulation. We demonstrate the viability of this approach through two distinct applications: a Variational Quantum Classifier (VQC) trained on a synthetic dataset and a novel application of a Variational Quantum Eigensolver (VQE) as an optimizer for a Neural Architecture Search (NAS) problem. Our results showcase the simplicity, flexibility, and potential of a seamlessly integrated quantum-classical workflow entirely within a classical deep learning framework.

## 1. Introduction

Quantum computing is poised to revolutionize numerous fields by leveraging the principles of quantum mechanics to solve problems intractable for classical computers. A particularly promising area is Quantum Machine Learning (QML), which seeks to develop quantum algorithms that can enhance machine learning tasks. The dominant paradigm in QML involves hybrid quantum-classical models, where a quantum computer or simulator is used as a coprocessor, orchestrated by a classical machine learning library such as PyTorch or TensorFlow.

This hybrid approach has led to the development of powerful software libraries like Qiskit, Cirq, and PennyLane, which provide high-level abstractions for building and executing quantum circuits. While these tools have been instrumental in advancing the field, they often introduce a layer of separation between the quantum and classical components of a model.

This paper explores an alternative: the direct and native implementation of quantum mechanics within PyTorch. We present **QuantumPyTorch**, an approach that treats quantum statevectors and operators as fundamental PyTorch tensors. This allows for the construction of quantum circuits as standard `torch.nn.Module` objects, making them fully compatible with PyTorch's ecosystem, including its dynamic computation graph, automatic differentiation, and extensive optimization libraries.

We posit that this direct integration offers several advantages:
*   **Simplicity:** It provides an intuitive model for researchers already proficient in PyTorch.
*   **Flexibility:** Quantum circuits can be seamlessly embedded within larger, complex classical deep learning architectures.
*   **Performance:** It directly leverages PyTorch's highly optimized backend for tensor operations, including massive parallelization on GPUs.

To validate this approach, we present two case studies based on our project, **QuILT** (Quantum-Inspired Learning and Tuning). The first is a straightforward implementation of a Variational Quantum Classifier. The second showcases a more advanced and novel application: a `VQEOptimizer` used to solve a Neural Architecture Search (NAS) problem, demonstrating how the Variational Quantum Eigensolver (VQE) can be repurposed as a powerful optimization tool for classical machine learning challenges.

## 2. Background and Related Work

### 2.1. Principles of Quantum Computing

A quantum computer's fundamental unit of information is the **qubit**, which can exist in a superposition of two basis states, |0⟩ and |1⟩. A system of *n* qubits is described by a statevector |ψ⟩ in a 2<sup>n</sup>-dimensional complex Hilbert space. The evolution of a quantum state is governed by the application of unitary operators, known as **quantum gates**. Multi-qubit gates, such as the Controlled-NOT (CNOT) gate, can create **entanglement**, a key quantum resource where qubits become correlated in a way that has no classical analogue.

### 2.2. Variational Quantum Algorithms

Variational Quantum Algorithms (VQAs) are a class of hybrid algorithms well-suited for Noisy Intermediate-Scale Quantum (NISQ) devices. They consist of three main components:
1.  An **ansatz**, which is a parameterized quantum circuit U(θ) that prepares a trial quantum state |ψ(θ)⟩.
2.  A **cost function**, typically derived from the expectation value of a Hamiltonian operator ⟨H⟩ with respect to the trial state.
3.  A **classical optimizer** that iteratively updates the parameters θ to minimize the cost function.

The **Variational Quantum Eigensolver (VQE)** is a canonical VQA designed to find the lowest eigenvalue (ground state energy) of a given Hamiltonian. According to the variational principle of quantum mechanics, the expectation value of the Hamiltonian is always greater than or equal to its true ground state energy. VQE leverages this by training the parameters of the ansatz to prepare a state that minimizes this expectation value.

### 2.3. Existing Quantum-Classical Frameworks

Libraries like PennyLane, Qiskit's `qiskit-machine-learning` module, and TorchQuantum are designed to bridge quantum computing and classical machine learning. They provide tools to define quantum circuits and integrate them as layers within PyTorch or TensorFlow models. While powerful, our "QuantumPyTorch" approach differs by eschewing specialized quantum classes and demonstrating that a full statevector simulation can be performed using only native PyTorch tensor operations.

## 3. The QuantumPyTorch Methodology

The core principle of QuantumPyTorch is to represent all quantum mechanical concepts using PyTorch's native `torch.Tensor` class.

### 3.1. Statevectors and Operators

A quantum statevector for *n* qubits is represented as a one-dimensional PyTorch tensor of size 2<sup>n</sup> with a complex floating-point dtype. For example, the initial state |00...0⟩ is a tensor where the first element is 1.0 and all others are 0.

Quantum gates are represented as square matrices (2D tensors). A single-qubit gate acting on a specific qubit within an *n*-qubit system is constructed by taking the tensor product (via `torch.kron`) of the gate matrix with identity matrices for all other qubits. This creates a full 2<sup>n</sup> x 2<sup>n</sup> operator that can be applied to the statevector.

### 3.2. Differentiable Quantum Circuits

A parameterized quantum circuit is constructed as a sequence of matrix-vector multiplications. Since the parameters (e.g., rotation angles `theta`) are part of PyTorch's computation graph, the entire circuit is differentiable.

```python
# Example of a differentiable RY gate matrix
def ry_matrix(theta):
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    row1 = torch.stack([c, -s])
    row2 = a.stack([s, c])
    return torch.stack([row1, row2]).to(torch.cfloat)
```

By encapsulating this sequence of operations within a `torch.nn.Module`, we create a quantum layer whose parameters can be learned through backpropagation.

### 3.3. Measurement

The expectation value of an observable (represented by a Hermitian operator, H) for a given state |ψ⟩ is calculated as ⟨ψ|H|ψ⟩. In PyTorch, this translates to `torch.real(torch.vdot(psi, H @ psi))`. This scalar value serves as the output of the quantum circuit or the cost function for optimization.

## 4. Experiments and Results

We validate our approach with two experiments, demonstrating its application in both quantum machine learning and classical optimization. (See Section 7 for code).

### 4.1. Experiment 1: Quantum-Enhanced Classification

In this experiment, we build a variational quantum classifier to solve the `make_circles` binary classification problem, a task that is not linearly separable and often requires a non-linear kernel.

**Model:** The `QuantumCircuitSimulator` is a 4-qubit circuit implemented as a `nn.Module`. The architecture consists of three main parts:
1.  **Data Encoding:** Classical input features are encoded into the quantum state using parameterized rotation gates.
2.  **Variational Circuit:** A series of layers containing trainable rotation gates (RY, RZ) and fixed CNOT gates for entanglement. The trainable parameters are stored in `nn.Parameter`.
3.  **Measurement:** The expectation value of the Pauli-Z operator on the first qubit is used as the model's output logit.

**Training:** The model was trained for 80 epochs using the `Adam` optimizer and `BCEWithLogitsLoss`.

**Results:** The classifier successfully learns to separate the two circles, achieving a final test accuracy of **96.67%**. This result demonstrates that a quantum circuit, simulated and trained entirely within PyTorch, can function effectively as a non-linear machine learning model.

### 4.2. Experiment 2: VQEOptimizer for Neural Architecture Search

This experiment showcases a novel use of the VQE algorithm as a general-purpose optimizer for a discrete, classical problem: Neural Architecture Search (NAS).

**Problem Formulation:** The goal is to find the optimal architecture for a "picoTransformer" model designed for a mathematical question-answering task. A search space of 16 possible architectures is defined, each encoded by a 4-bit binary string (e.g., '0000', '0001', ..., '1111').

**Methodology:**
1.  **Hamiltonian Construction:** A cost is assigned to each of the 16 architectures by training a surrogate transformer model and scoring its performance on a test question. These costs form the diagonal elements of a 16x16 Hamiltonian matrix, `H`. The ground state of this Hamiltonian corresponds to the architecture with the minimum cost (highest performance).
2.  **The `VQEOptimizer`:** A 4-qubit VQE model is implemented. The number of qubits matches the length of the architecture-encoding bitstring. The model's objective is to find the quantum state `|ψ⟩` that minimizes the energy `⟨ψ|H|ψ⟩`.
3.  **Optimization:** The VQE is trained for 50 epochs using the `Adam` optimizer.
4.  **Result Decoding:** After training, the final quantum state `|ψ⟩` is a superposition of all 16 basis states (architectures). The architecture corresponding to the basis state with the highest probability amplitude `|⟨i|ψ⟩|^2` is selected as the optimal one.

**Results:** The `VQEOptimizer` successfully converged to a low-energy state. The most probable basis state found was **|10⟩ ('1010')**, with a probability of **0.97**. This corresponded to the architecture named **"Large"** (`n_layer=6`, `n_embd=128`, `dropout=0.2`, `lr=1e-3`), demonstrating that the VQE can be effectively repurposed to find the optimal solution in a discrete search space. This novel application highlights the potential of mapping complex combinatorial optimization problems onto a quantum-inspired framework.

## 5. Discussion

The success of both experiments validates the central thesis of this paper: it is not only possible but also practical to implement and train variational quantum algorithms directly in PyTorch.

**Advantages:**
The primary advantage is the seamless integration into a mature and highly-optimized deep learning ecosystem. The quantum circuit is just another differentiable module, allowing for the construction of sophisticated hybrid models and the use of standard training loops, optimizers, and analysis tools. The ability to harness GPU acceleration for statevector simulation is a significant performance benefit for moderate numbers of qubits.

**Limitations:**
The most significant limitation of this approach is inherent to any classical statevector simulation: scalability. The memory required to store the statevector grows exponentially (O(2<sup>n</sup>)) with the number of qubits *n*. This restricts the direct simulation to a relatively small number of qubits (typically < 30). This method does not provide a connection to actual quantum hardware or account for quantum noise, which are critical features of dedicated platforms like Qiskit and Cirq.

**Novelty:**
While other libraries enable PyTorch integration, the "QuantumPyTorch" approach is notable for its "from-scratch" nature, relying only on core PyTorch functionalities. Furthermore, the use of a VQE for Neural Architecture Search is a novel demonstration of mapping a classical hyperparameter optimization problem onto a quantum computational paradigm. This suggests that VQE can serve as a versatile optimization tool beyond its traditional applications in chemistry and physics.

## 6. Conclusion and Future Work

This paper has introduced QuantumPyTorch, a methodology for building, simulating, and training quantum circuits directly within the PyTorch framework. We have demonstrated its effectiveness through a quantum classifier and a novel VQE-based optimizer for Neural Architecture Search. This approach simplifies the quantum-classical workflow and leverages the full power of PyTorch's capabilities.

While classical simulation is limited by the number of qubits, the principles demonstrated here provide a powerful platform for research, prototyping, and education in quantum machine learning. Future work could focus on extending this framework to include more complex quantum phenomena, such as noisy simulations, or exploring other classes of optimization problems that can be mapped to a VQE Hamiltonian. By lowering the barrier to entry, a direct, tensor-based approach can foster greater cross-pollination between the deep learning and quantum computing communities.

## 7. Code Availability

The code and experiments for this project, named **QuILT** (Quantum-Inspired Learning in Tensorspace), are publicly available on GitHub.

*   **Main Repository:** [https://github.com/peterbabulik/QuILT/](https://github.com/peterbabulik/QuILT/)

*   **Experiment 1 (Quantum Classifier):** The Jupyter notebook containing the code for the `QuantumCircuitSimulator` and the `make_circles` classification task can be found at:
    [https://github.com/peterbabulik/QuILT/blob/main/QuILT.ipynb](https://github.com/peterbabulik/QuILT/blob/main/QuILT.ipynb)

*   **Experiment 2 (VQE for NAS):** The Jupyter notebook for the `VQEOptimizer` applied to Neural Architecture Search for the picoTransformer is available here:
    [https://github.com/peterbabulik/QuILT/blob/main/QuILT_NAS.ipynb](https://github.com/peterbabulik/QuILT/blob/main/QuILT_NAS.ipynb)
