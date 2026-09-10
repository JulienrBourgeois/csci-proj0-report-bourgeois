---
title: Project 0 — PyTorch Warmup
---

# Project 0: PyTorch Warmup

**Name:** Julien Bourgeois

This report has two independent parts: a Fashion-MNIST MLP classifier, and English–French translation with Bahdanau attention.

---

## Part 1: Multilayer Perceptrons on Fashion-MNIST

Fashion-MNIST is 70,000 grayscale 28×28 clothing images in 10 classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot). Torchvision downloads the official split (60,000 train / 10,000 test). We hold out 10% of the official training set for validation: 54,000 train / 6,000 val / 10,000 test. Pixels are scaled to `[0, 1]`.

### Task 1: Dataset and baseline model

![Ten Fashion-MNIST samples](figures/task1_fashion_mnist_samples.png)

With labels, my brain immediately sees what each picture is — but that is a bit of cheating. The class name is already telling me the story. Unlabeled, it is a lot harder. These are blurry 28×28 blobs. Shoes are still obviously shoes, and heels versus sneakers are easy. Coat versus pullover is hard, dresses versus coats are not that easy, and even pants are not obvious. If nobody told me these were clothes, I could force a completely different story onto them.

That is also why we flatten each image to 784 numbers before the MLP. A standard fully connected layer is built for a 1-D vector. It has no built-in mechanism for a 2-D grid of pixels, so we unroll the picture first.

The baseline is flatten → Linear(784 → 256) → ReLU → Linear(256 → 10) logits. Training used SGD (learning rate 0.1), `CrossEntropyLoss`, batch size 256, and 10 epochs. There is no softmax on the last layer: PyTorch's `CrossEntropyLoss` already applies it internally, so putting another one on the model would be redundant.

After 10 epochs: train loss 0.380, val loss 0.375, val accuracy **0.868**.

![Baseline training curves](figures/task1_baseline_curves.png)

Val bounced instead of snapping into a perfect copy of the training curve. At epoch 5 it actually got worse (0.478) before coming back down. Over the full run it gradually moved toward the training loss rather than pulling away. I do not read that as overfitting — val never ran off while train kept falling — and I do not read it as underfitting either, because both losses came down and the model improved on held-out data.

87% feels definitely good, especially since I do not have a lot of experience training models. I would expect higher with more data or with techniques that make training more efficient. This was not a pretrained model. Torchvision only downloaded Fashion-MNIST; the MLP started from random weights and we trained it ourselves.

### Task 2: Hidden layers and model capacity

**Extra hidden layer** (`784 → 256 → 256 → 10`)

![Two hidden layers](figures/task2_two_hidden_layers.png)

*TODO: compare this run to Task 1. Val accuracy landed around 0.871. Was the extra layer worth it, or basically a wash? Did the curves look any healthier?*

**One-neuron hidden layer** (`784 → 1 → 10`)

![One-neuron bottleneck](figures/task2_one_neuron.png)

*TODO: val accuracy around 0.390. Everything after that layer only sees one number per image. Tie this back to Task 1: if coat vs pullover is already hard for a human looking at all 784 pixels, what happens when the net is forced to squash the whole picture into a single scalar?*

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
