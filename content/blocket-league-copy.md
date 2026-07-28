<!-- block:play-title -->
Punchline: Steer a Video Model's Hallucinations.
<!-- /block -->

<!-- block:hero-intro -->
This lab demonstrates that video models can build a compact internal state of a physical world purely from raw video. In a toy collision game, a linear readout from one fixed hidden token recovers both objects' Cartesian x/y positions; other linear probes recover velocity and anticipate collisions before contact. We then distinguish what is merely readable from what is causally writable, using downstream Jacobian directions to steer the model's hallucinations into a video game.
<!-- /block -->

<!-- block:hero-sources -->
This is essentially an application of Anthropic's [Jacobian lens](https://www.anthropic.com/research/global-workspace) to illustrate ideas developed in *[Interpreting Physics in Video World Models](https://arxiv.org/abs/2602.07050)*, also inspired by [MIRA](https://mira-wm.com/blog-post/).
<!-- /block -->

<!-- block:play-intro -->
Put the puck in the goal! This game is the real-time output of a 3.67M-parameter video transformer running in your browser. Arrow keys write to the model activations and "steer" the model's hallucinations.
<!-- /block -->

<!-- block:play-takeaway -->
This is not perfect but clearly has grokked the basic physics of the game, including collisions, bounces, and scoring. Notably, the model has never explicitly observed keyboard directions. These were discovered post-hoc via interpretability methods (J-Lens).
<!-- /block -->

<!-- block:dataset-title -->
The toy world.
<!-- /block -->

<!-- block:dataset -->
We train it on **Blocket League**, a toy physics simulation with collision mechanics between and goal scoring. We sample rollouts and train with a next-frame-prediction objective.
<!-- /block -->

<!-- block:model-title -->
Training a toy video prediction model.
<!-- /block -->

<!-- block:model -->
To study the learned representation of video models, we can train a tiny video transformer on a toy physics simulation. For this exercise, we use a 3.67M parameter pixel transformer, which directly operates on pixels and predicts the next frame, similar to an LLM predicting the next word.
<!-- /block -->

<!-- block:model-results -->
For about **$10 of compute**, the model cleanly learns the transition function and can “hallucinate” valid game rollouts by recursively sampling frames. The learned dynamics are pretty spot on.
<!-- /block -->

<!-- block:linear-position-title -->
Decoding the Learned Representation
<!-- /block -->

<!-- block:linear-position-intro -->
How accessible is this abstract learned representation?

Some physical properties are easily accessible within the transformer's hidden state. A simple linear regression from block 5 activations, for example, can neatly predict X/Y coordinates of both the orange and the blue ball.

Similarly, we can see this same layer will "anticipate" a collision up to eight frames before contact. [See the collision probe in Appendix E ↓](#collision-representation)
<!-- /block -->

<!-- block:generalization-title -->
Generalizing to unseen scenarios
<!-- /block -->

<!-- block:generalization-intro -->
Is this model merely memorizing the training set, or is it able to learn some higher-level representation that generalizes beyond seen examples?


If you retrain this model with certain classes of samples entirely removed - for example, no collisions in the top right quadrant - it turns out the model can still accurately predict what will happen in these scenarios. This is fairly significant and implies the model has learned some higher-level representation that captures entities and their relationships, as opposed to merely interpolating between previously-seen samples.
<!-- /block -->

<!-- block:jacobian-title -->
Finding steerable directions with the Jacobian lens.
<!-- /block -->

<!-- block:jacobian-lens -->
Reading x/y is one thing. Changing it is another. With the [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html#the-jacobian-lens), we trace how block-5 activations affect the next frame, then average those gradients into reusable motion directions. [See the implementation](https://github.com/jayhack/blocket-league/blob/main/blocket_league/pixel_probe.py#L130-L149).
<!-- /block -->

<!-- block:causal-title -->
These variables are causal. Write to them and the hallucination changes.
<!-- /block -->

<!-- block:causal-intervention -->
Write the recovered +x direction for four frames, then stop. By frame 12, the player disc is 3.51 pixels farther right on average across 256 unseen worlds, and 85.9% move in the intended direction. A random activation direction has almost no effect.
<!-- /block -->

<!-- block:brain-surgery-title -->
This is a video game. You play it through brain surgery.
<!-- /block -->

<!-- block:brain-surgery -->
If you map your keyboard to these causal direction probes, it becomes a simple video game. Note that the model has exclusively trained on raw pixels - we are not manipulating the model's input, but rather it's own internal representation for what it has "discovered" about motion.

This demo loads the full 3.67M-parameter model into your browser.
<!-- /block -->

<!-- block:representation-title -->
Direction becomes a ring.
<!-- /block -->

<!-- block:representation-depth -->
Average the block-5 activations by motion direction and project them down: they wrap into a circle. Nearby directions in the world are nearby inside the model.
<!-- /block -->

<!-- block:model-scale-title -->
Appendix A: How small can the world model get?
<!-- /block -->

<!-- block:direction-holdout-title -->
Appendix B: Does the model generalize to a direction it never saw?
<!-- /block -->

<!-- block:collision-holdout-title -->
Appendix C: Does collision physics transfer across the arena?
<!-- /block -->

<!-- block:position-geometry-title -->
Appendix D: Does the model construct Cartesian coordinates?
<!-- /block -->

<!-- block:collision-representation-title -->
Appendix E: Can the model see collisions coming?
<!-- /block -->

<!-- block:collision-representation -->
Yes. A linear probe at block 5 predicts a collision up to eight frames—400 ms—before contact, even when collide and miss examples have matched positions and speeds.
<!-- /block -->

<!-- block:experiment-index-title -->
Appendix F: Registered experiments
<!-- /block -->
