# Sparse Fast Gradient Encryption

This repo is the source code of paper entitled "Enabling Secure in-Memory Neural Network Computing by Sparse Fast Gradient Encryption" which was presented in International Conference on Computer-Aided Design 2019. 

## Abstract
Neural network (NN) computing is energy- consuming on traditional computing systems, owing to the in- herent memory wall bottleneck of the von Neumann architecture and the Moore’s Law being approaching the end. Non-volatile memories (NVMs) have been demonstrated as promising alter- natives for constructing computing-in-memory (CiM) systems to accelerate NN computing. However, NVM-based NN computing systems are vulnerable to the confidentiality attacks because the weight parameters persist in memory when the system is powered off, enabling an attacker with physical access to extract the well- trained NN models. The goal of this work is to find a solution for thwarting the confidentiality attacks. We define and model the weight encryption problem. Then we propose an effective framework, containing a sparse fast gradient encryption (SFGE) method and a runtime encryption scheduling (RES) scheme, to guarantee the confidentiality security of NN models with a negligible performance overhead. The experiments demonstrate that only encrypting an extremely small proportion of the weights (e.g., 20 weights per layer in ResNet-101), the NN models can be strictly protected.



