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
