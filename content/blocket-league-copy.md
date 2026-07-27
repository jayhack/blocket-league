<!-- block:hero-intro -->
This lab demonstrates that video models can learn compact, interpretable, and causal representations of physical phenomena purely from raw video. We train a video transformer on a toy physics world with simple collision/scoring dynamics, then identify causal directions in activation space for velocity and use them to steer the model's hallucinations to form a video game.
<!-- /block -->

<!-- block:hero-sources -->
This is essentially an application of Anthropic's [Jacobian lens](https://www.anthropic.com/research/global-workspace) to illustrate ideas developed in *[Interpreting Physics in Video World Models](https://arxiv.org/abs/2602.07050)*, also inspired by [MIRA](https://mira-wm.com/blog-post/).
<!-- /block -->

<!-- block:play-intro -->
Put the puck in the goal! This game is the real-time output of a 3.67M-parameter video transformer running in your browser. Arrow keys write to the model activations and "steer" the model hallucinations.
<!-- /block -->

<!-- block:play-takeaway -->
This is not perfect but clearly has grokked the basic physics of the game, including collisions, bounces, and scoring. Notably, the model has never explicitly observed keyboard directions. These were discovered post-hoc via interpretability methods (J-Lens).
<!-- /block -->

<!-- block:dataset -->
Blocket League is a deliberately simple world with two freely moving discs, elastic collisions, wall bounces, and a goal. We render a fixed training cache of 16,384 autonomous worlds, each 24 frames long, with randomized initial positions and velocities. Neither disc is controlled by an agent or player, so every trajectory is a passive physics sample.
<!-- /block -->

<!-- block:model -->
For this lab, we train an 8-frame × 16 × 16-patch pixel transformer. Pixel transformers, following the autoregressive formulation of the [Image Transformer](https://arxiv.org/abs/1802.05751), ingest raw pixels and learn to predict the next frame. Our complete 3.67M-parameter architecture lives in [one Python file](https://github.com/jayhack/blocket-league/blob/main/blocket_league/pixel_direct_model.py).

The exported checkpoint is trained for 42,000 optimizer steps (30,000 base minibatches plus a 12,000-step recovery fine-tune) at batch size 16. Because training windows are sampled with replacement, conventional epochs do not apply. Across both phases, the model sees 672,000 sampled histories and 5.376 million next-frame prediction targets. Rendering the caches, training, and evaluation take 24 minutes and 59 seconds on one H100.
<!-- /block -->

<!-- block:model-results -->
This performs surprisingly well at next-frame prediction and retains coherence over long sampling rollouts. The video-completion samples below preserve collisions, bouncing, and goals, as well as simpler building blocks such as continued motion and velocity. Each sample begins with 12 observed frames; the next 36 are generated one at a time and fed back into the model.
<!-- /block -->

<!-- block:jacobian-lens -->
Inspired by Anthropic's [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html#the-jacobian-lens), we ask whether motion is not only readable from the model's hidden states, but causally addressable. We sample 512 trajectories, measure how block-5 activations affect the player disc's next-frame x/y position, and average those gradients into reusable motion directions. These directions are written into activations at inference time; the model's weights remain frozen. [See the implementation](https://github.com/jayhack/blocket-league/blob/main/blocket_league/pixel_probe.py#L130-L149).
<!-- /block -->

<!-- block:causal-intervention -->
Write the recovered +x direction for four frames, then stop. By frame 12, the player disc is 3.51 pixels farther right on average across 256 unseen worlds, and 85.9% move in the intended direction. A random activation direction has almost no effect.
<!-- /block -->

<!-- block:brain-surgery -->
If you map your keyboard to these causal direction probes, it becomes a simple video game. Note that the model has exclusively trained on raw pixels - we are not manipulating the model's input, but rather it's own internal representation for what it has "discovered" about motion.

This demo loads the full 3.67M-parameter model into your browser.
<!-- /block -->

<!-- block:representation-depth -->
Following [Interpreting Physics in Video World Models](https://arxiv.org/abs/2602.07050), we trace motion through the network block by block. At the patch embedding, direction is absent. One block later it becomes linearly decodable, then sharpens through the remaining layers as speed catches up.
<!-- /block -->
