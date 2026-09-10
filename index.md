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

The baseline is a multilayer perceptron. A fully connected layer is built to take a 1-D list of numbers rather than a 2-D grid, so we flatten each 28×28 picture into 784 values before the first linear layer. The architecture is flatten → Linear(784 → 256) → ReLU → Linear(256 → 10) logits, trained with SGD at learning rate 0.1, `CrossEntropyLoss`, batch size 256, and 10 epochs. There is no softmax on the last layer, because PyTorch's `CrossEntropyLoss` expects raw logits and applies log-softmax internally; a softmax on the model would just duplicate that work. This was also not a pretrained network: Torchvision only downloaded the images, and the MLP started from random weights.

After 10 epochs the training loss was 0.380, the validation loss was 0.375, and validation accuracy was **0.868**.

![Baseline training curves](figures/task1_baseline_curves.png)

The validation curve bounced instead of tracking the training curve smoothly. At epoch 5 the validation loss actually got worse (0.478) before coming back down, which is not a diagnosis by itself: the validation set is only 6,000 images, so that estimate is noisier than the training loss, and a learning rate of 0.1 is aggressive enough to produce an ugly epoch. Over the full run, validation drifted toward training rather than pulling away. I do not read that as overfitting, because validation never ran off while training kept falling, and I do not read it as underfitting, because both losses came down and accuracy on held-out data improved.

That 87% also matches what looking at the pictures led me to expect. Footwear was already easy for a human, so I would guess the remaining mistakes are concentrated in lookalike tops — coat versus pullover, dress versus coat. We did not plot a confusion matrix, so that is a reading of the samples rather than a measured error breakdown. Flattening throws away the 2-D layout on top of that, which means this network is not looking at a sleeve the way a person does; it is classifying a list of 784 pixel values. We already have 54,000 training images, so the limit here is more the model than the size of the dataset.

### Task 2: Hidden layers and model capacity

Task 1 already suggested that the ceiling on this experiment is the model, not the number of images. This task keeps the same data, optimizer, loss, batch size, and 10 epochs, and only changes how much hidden capacity the MLP has.

**Extra hidden layer** (`784 → 256 → 256 → 10`)

![Two hidden layers](figures/task2_two_hidden_layers.png)

Adding a second 256-unit layer takes the network from about 204k parameters to about 269k. After 10 epochs the training loss was 0.361, the validation loss was 0.354, and validation accuracy was **0.871**. That is three tenths of a point above the Task 1 baseline of 0.868, which on a 6,000-image validation set is about 18 extra correct labels. I do not treat that as a real improvement. It is a wash, and I would not bother stacking the extra layer: it makes the model more complicated for no meaningful gain, which is the Occam's-razor argument against it.

The curves tell the same story as Task 1 rather than a healthier one. Validation still bumps — this time at epoch 4, when val loss rose from 0.493 to 0.513 while training kept falling — and then drifts back toward the training loss. Extra wiggles are not a sign of a better run; they are the same noisy validation estimate and the same aggressive learning rate. Extra fully connected depth also does not restore the 2-D layout we flattened away in Task 1. You can only add so many of these layers before you are just making a bigger list-processor, which is probably the wrong kind of model for these thumbnails.

**One-neuron hidden layer** (`784 → 1 → 10`)

![One-neuron bottleneck](figures/task2_one_neuron.png)

The opposite experiment is an extreme bottleneck: 784 pixels are linearly combined into a single hidden unit, ReLU is applied, and then ten class scores are read off that one number. The whole network has 805 parameters. After 10 epochs the losses were still around 1.44, and validation accuracy was **0.390**.

That one hidden value is not a secret clothing code. It is one weighted sum of the whole picture, so the network has to line all ten classes up on a single axis. One scalar cannot separate coat from pullover, and it cannot even be trusted to keep shoe, shirt, and bag in different places if those classes overlap on that axis. Random guessing among ten balanced classes is 10%, so 39% means the tiny net did find some coarse signal — maybe how filled-in the thumbnail is, or footwear versus tops — but we did not inspect the weights, so that is a guess about *what* it latched onto. What is not a guess is that it failed the actual 10-way task: 39% is far from the 87% of the 256-unit net, which is what you would expect if coat versus pullover is already hard when a person can see all 784 pixels.

The one-neuron curves were still drifting down at epoch 10, so a longer run might pick up a few more points. They will not close the gap to 87%. The losses never left the 1.4–1.9 range, which is the signature of a model that is too small, not a model that merely needed more time.

### Task 3: Activation functions and gradients

Task 2 changed how *much* hidden capacity the MLP had. This task keeps the two-layer width (`256, 256`) and only changes the bend after each linear layer: ReLU versus sigmoid. Same data, same SGD, same 10 epochs. An activation is what stops stacked linear layers from collapsing into one linear map. ReLU zeros negatives and leaves positives alone. Sigmoid squishes every number into `(0, 1)` with an S-curve whose two ends are almost flat.

![ReLU vs sigmoid loss](figures/task3_relu_vs_sigmoid.png)

ReLU finished at train loss 0.364, val loss 0.357, and validation accuracy **0.868**, the same neighborhood as Task 1. Sigmoid finished at 0.645 / 0.620 / **0.764**. ReLU won this run. Sigmoid is still climbing at epoch 10, so it is not frozen at chance, but it started far behind — about 25% val accuracy at epoch 1 against ReLU’s 74% — and it never caught up. Swapping the activation dropped about ten points. Adding a ReLU layer in Task 2 had not.

The reason is how training actually updates the first layer, the one that sees the 784 pixels. Each update follows the gradient: if a weight barely changes the loss when you nudge it, it barely learns. On the flat parts of a sigmoid, that nudge does almost nothing, and those tiny factors multiply through a stack. The last hidden layer can still see a signal; the pixel layer gets starved. That is vanishing gradients. It is a learning bottleneck, not the size bottleneck from the one-neuron net.

The next two figures are a *different* experiment, used to make that mechanism visible: four hidden layers instead of two, random weights rather than a trained classifier, five minibatches, plotting the L2 gradient norm at each hidden layer. `hidden_0` is closest to the pixels; `hidden_3` is closest to the ten class scores. This is not another accuracy run, and it is deeper than the ReLU-versus-sigmoid training plot, so it should not be read as a copy of that 76% vs 87% result. It is the probe for why sigmoid is a bad stack.

![Sigmoid gradient norms](figures/task3_sigmoid_grad_norms.png)

![ReLU gradient norms](figures/task3_relu_grad_norms.png)

On the sigmoid plot (log scale) the last hidden layer sits around `0.14` and the first around `0.0007`, roughly two hundred times smaller. The pixel layer is almost dead. On the ReLU plot the first layer is actually the largest, and the later layers stay in the same order of magnitude. That is not the same failure. ReLU does not squash everything onto a flat S, so a useful gradient can still reach the weights that look at the picture. That is why I would keep ReLU for this MLP and why sigmoid lost the 10-epoch comparison even at only two hidden layers.

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
