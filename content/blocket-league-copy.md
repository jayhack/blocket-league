<!-- block:hero-intro -->
This lab demonstrates that video models can learn compact, interpretable, and causal representations of physical phenomena purely from raw video.

 We train a video transformer on a toy physics world with simple collision/scoring dynamics, then identify causal directions in activation space for velocity and use them to steer the model's hallucinations to form a video game.
<!-- /block -->

<!-- block:hero-sources -->
This is essentially an application of Anthropic's [Jacobian lens](https://www.anthropic.com/research/global-workspace) to illustrate ideas developed in *[Interpreting Physics in Video World Models](https://arxiv.org/abs/2602.07050)*, also inspired by [MIRA](https://mira-wm.com/blog-post/).
<!-- /block -->

<!-- block:play-intro -->
Put the puck in the goal! This game is the real-time output of a 3.67M-parameter video transformer running in your browser. Arrow keys write to the model activations and "steer" the model hallucinations.
<!-- /block -->

<!-- block:play-takeaway -->
This is not perfect but clearly has grokked the basic physics of the game, including collisions, bounces, and scoring. Notably, the model has never explicitly observed keyboard directions—these were discovered post-hoc via interpretability methods (J-Lens).
<!-- /block -->

<!-- block:dataset -->
Blocket League is a deliberately simple world with two freely moving discs, elastic collisions, wall bounces, and a goal. We sample *M* trajectories with randomized initial positions and velocities, then record the resulting frames. Neither disc is controlled by an agent or player, so every trajectory is an autonomous physics sample.
<!-- /block -->

<!-- block:model -->
For this lab, we train an 8-frame × 16 × 16-patch pixel transformer. Pixel transformers, following the autoregressive formulation of the [Image Transformer](https://arxiv.org/abs/1802.05751), ingest raw pixels and learn to predict the next frame. Our complete 3.67M-parameter architecture lives in [one Python file](https://github.com/jayhack/blocket-league/blob/main/blocket_league/pixel_direct_model.py).
<!-- /block -->

<!-- block:model-results -->
This performs surprisingly well at next-frame prediction. The video-completion samples below preserve collisions, bouncing, and goals, as well as simpler building blocks such as continued motion and velocity. Each sample begins with 12 observed frames; the next 36 are generated one at a time and fed back into the model.
<!-- /block -->

<!-- block:jacobian-lens -->
Inspired by Anthropic's [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html#the-jacobian-lens), we investigate this model's activations to identify directions in activation space that correspond to downstream physical observations such as velocity. Concretely, for each trajectory, we locate the green puck's hidden-state token, predict the next frame, and measure the downstream x/y centroid of the green-puck pixels. We backpropagate each centroid coordinate to that token, then average the resulting gradients over 512 randomized trajectories to produce reusable x- and y-velocity directions. [See the implementation](https://github.com/jayhack/blocket-league/blob/main/blocket_league/pixel_probe.py#L130-L149).
<!-- /block -->

<!-- block:causal-intervention -->
Write the recovered +x direction for four frames, then stop. By frame 12, the green circle is 3.51 pixels farther right on average across 256 unseen worlds, and 85.9% move in the intended direction. A random activation direction has almost no effect.
<!-- /block -->

<!-- block:brain-surgery -->
We map these directions to your keyboard and let you directly steer the model, forming the simple video game illustrated below. This loads the full 3.67M-parameter model into your browser.
<!-- /block -->

<!-- block:representation-depth -->
Following [Interpreting Physics in Video World Models](https://arxiv.org/abs/2602.07050), we trace motion through the network block by block. At the patch embedding, direction is absent. One block later it becomes linearly decodable, then sharpens through the remaining layers as speed catches up.
<!-- /block -->
