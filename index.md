---
title: Project 0 — PyTorch Warmup
---

**Name:** Julien Bourgeois

This report covers two independent PyTorch experiments: a Fashion-MNIST classifier built from a multilayer perceptron, and an English–French translator that uses Bahdanau attention.

---

## Part 1: Multilayer Perceptrons on Fashion-MNIST

The first experiment is a classification problem. Given a tiny grayscale picture of a clothing item, the model has to assign it to one of ten categories.

### Task 1: Dataset and baseline model

Fashion-MNIST is a clothing-themed replacement for the original MNIST digit dataset: 70,000 images, each 28×28 and grayscale, with a single item roughly centered in the frame. The ten labels are T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, and Ankle boot. Torchvision downloads the official 60,000 / 10,000 train/test split, and we hold out 10% of the official training set for validation, which leaves 54,000 images for training, 6,000 for validation, and 10,000 for test. Pixel intensities are scaled to the range `[0, 1]`. The numbers describe the collection; the figure below is what it actually looks like.

![Ten Fashion-MNIST samples](figures/task1_fashion_mnist_samples.png)

These are not photographs you would recognize at a glance. They are pixelated catalog thumbnails, and once a class name is printed above an image my brain fills in the rest, which is a kind of cheating: the label is already telling me what I am supposed to see. Cover the labels and the same pictures are much harder. Footwear still reads as shoes, and a heel is easy to separate from a sneaker, but coat versus pullover is genuinely difficult, dress versus coat is not much easier, and even pants are not obvious. If nobody had told me these were clothes, I could have talked myself into a completely different story. The model will not have that freedom, because it is only allowed to pick among those ten names. In practice, then, the dataset is a pile of low-resolution clothing images, a short list of labels, and a handful of pairs that still look alike even to a person.

The baseline is a multilayer perceptron. A fully connected layer is built to take a 1-D list of numbers rather than a 2-D grid, so we flatten each 28×28 picture into 784 values before the first linear layer. The architecture is flatten → Linear(784 → 256) → ReLU → Linear(256 → 10) logits, trained with SGD at learning rate 0.1, `CrossEntropyLoss`, batch size 256, and 10 epochs. There is no softmax on the last layer, because PyTorch's `CrossEntropyLoss` already applies it internally and putting another one on the model would just duplicate that work. This was also not a pretrained network: Torchvision only downloaded the images, and the MLP started from random weights.

After 10 epochs the training loss was 0.380, the validation loss was 0.375, and validation accuracy was **0.868**.

![Baseline training curves](figures/task1_baseline_curves.png)

The validation curve bounced instead of tracking the training curve smoothly. At epoch 5 the validation loss actually got worse (0.478) before coming back down, which is not a diagnosis by itself: the validation set is only 6,000 images, so that estimate is noisier than the training loss, and a learning rate of 0.1 is aggressive enough to produce an ugly epoch. Over the full run, validation drifted toward training rather than pulling away. I do not read that as overfitting, because validation never ran off while training kept falling, and I do not read it as underfitting, because both losses came down and accuracy on held-out data improved.

That 87% also matches what looking at the pictures led me to expect. Footwear was already easy for a human, so the remaining mistakes should mostly be the lookalike tops — coat versus pullover, dress versus coat. Flattening throws away the 2-D layout on top of that, which means this network is not looking at a sleeve the way a person does; it is classifying a list of 784 pixel values. We already have 54,000 training images, so the limit here is more the model than the size of the dataset.

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
