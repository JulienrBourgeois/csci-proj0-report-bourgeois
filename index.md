---
title: Project 0 — PyTorch Warmup
---

# Project 0: PyTorch Warmup

**Name:** Julien Bourgeois

This report has two independent parts: a Fashion-MNIST MLP classifier, and English–French translation with Bahdanau attention.

> Draft skeleton. Replace every *TODO* with your own writing. Keep the figures. Do not paste code here.

---

## Part 1: Multilayer Perceptrons on Fashion-MNIST

### Data

Fashion-MNIST is 70,000 grayscale 28×28 clothing images in 10 classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).

Torchvision downloads the official split (60,000 train / 10,000 test). We hold out 10% of the official training set for validation:

- train: 54,000
- val: 6,000
- test: 10,000

Pixels are scaled to `[0, 1]`. Images stay shaped `(1, 28, 28)` in the loader and are flattened to 784 inside the MLP.

*TODO (optional): one sentence on why a validation split is useful.*

### Task 1: Dataset and baseline model

![Ten Fashion-MNIST samples](figures/task1_fashion_mnist_samples.png)

**Architecture**

*TODO: layer sizes, activation, loss, optimizer, batch size, epochs. Starter facts from the run: flatten 784 → Linear(256) → ReLU → Linear(10) logits. SGD, learning rate 0.1, CrossEntropyLoss, batch 256, 10 epochs. No softmax on the last layer because CrossEntropyLoss expects logits.*

**Results**

*TODO: final train loss, val loss, val accuracy. Starter numbers from the run: val accuracy about 0.868; train and val loss both ended around 0.38.*

![Baseline training curves](figures/task1_baseline_curves.png)

**Underfitting / overfitting**

*TODO: a few sentences. Did train and val loss move together, or did val get worse while train kept falling?*

### Task 2: Hidden layers and model capacity

**Extra hidden layer** (`784 → 256 → 256 → 10`)

![Two hidden layers](figures/task2_two_hidden_layers.png)

*TODO: how did extra depth change training loss vs validation accuracy vs the Task 1 baseline? Starter number: val accuracy about 0.871.*

**One-neuron hidden layer** (`784 → 1 → 10`)

![One-neuron bottleneck](figures/task2_one_neuron.png)

*TODO: this is an extreme information bottleneck — everything after that layer only sees one number per image. Relate that to the accuracy you got. Starter number: val accuracy about 0.390.*

### Task 3: Activation functions and gradients

Same width `(256, 256)`, ReLU vs sigmoid.

![ReLU vs sigmoid loss](figures/task3_relu_vs_sigmoid.png)

*TODO: compare final val accuracy and the shape of the curves. Starter numbers: ReLU about 0.868, sigmoid about 0.764 after 10 epochs.*

![Sigmoid gradient norms](figures/task3_sigmoid_grad_norms.png)

![ReLU gradient norms](figures/task3_relu_grad_norms.png)

*TODO: vanishing gradients. Sigmoid saturates, so its derivative is near zero for large |x|. Those small factors multiply through a deep stack and starve the earliest layers. ReLU does not saturate on the positive side, so early-layer norms usually stay larger at the same depth.*

### Task 4: Computational considerations

*TODO: no extra plot. Cover, in your own words:*

- Parameters are stored in both training and prediction.
- Training also stores activations for backprop, parameter gradients, and optimizer state.
- Prediction can use `torch.no_grad()` and only needs weights plus the current layer's activations.
- Batch size, width, and depth all scale activation memory.
- These runs used MPS (Apple GPU) when available, otherwise CPU; the same memory accounting applies.

---

## Part 2: English–French translation with Bahdanau attention

*TODO: one short paragraph. Encoder–decoder GRU, additive (Bahdanau) attention, short English–French pairs, teacher forcing during training, greedy decode at test time.*

### Task 1: Train the attention model

| Setting | Hidden / embed size | Dropout | Learning rate | Train loss | Val loss | Val BLEU |
|---|---|---|---|---|---|---|
| A | 32 | 0.1 | 0.005 |  |  |  |
| B | 64 | 0.2 | 0.002 |  |  |  |

*TODO: fill the empty loss/BLEU cells from the notebook. Starter numbers from the run: A train 2.40 / val 2.71 / BLEU 0.240; B train 2.19 / val 2.52 / BLEU 0.256.*

![Setting A loss](figures/task1_loss_setting_a.png)

![Setting B loss](figures/task1_loss_setting_b.png)

**What changed and what we observed**

*TODO: width, dropout, learning rate. BLEU on this tiny corpus is noisy; qualitative translations matter as much as the number.*

**Example translations**

| English | Reference | Setting A | Setting B |
|---|---|---|---|
|  |  |  |  |
|  |  |  |  |
|  |  |  |  |

*TODO: copy a few pairs from the notebook. Be honest about failures.*

### Task 2: Visualize and interpret attention

![Attention example 0](figures/task2_attention_0.png)

*TODO: what does each generated French token attend to? Is this helpful or unsuccessful?*

![Attention example 1](figures/task2_attention_1.png)

*TODO: look for a helpful pattern (a noun attending to its English counterpart, or a roughly diagonal alignment).*

![Attention example 2](figures/task2_attention_2.png)

*TODO: look for an unsuccessful pattern (mass on `<eos>` / pad, ignored source tokens, repeated generation, off-by-one).*

**Helpful example:** *TODO: which heatmap, and why.*

**Unsuccessful example:** *TODO: which heatmap, and why.*
