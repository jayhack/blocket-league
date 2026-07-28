<!-- block:play-title -->
The Punchline: Steer a Video Model's Hallucinations.
<!-- /block -->

<!-- block:hero-intro -->
How do video models store information about the physical world?

This lab demonstrates that video transformers can build a compact and interpretable internal models of physics purely from raw video.

We show latent states in video transformers encode position, velocity, collisions and more in geometric representations in their activations. What's more, these representations are "causal" - you can write to them and, in doing so, shape the models' output to form a video game.
<!-- /block -->

<!-- block:hero-sources -->
This is essentially an application of Anthropic's [Jacobian lens](https://www.anthropic.com/research/global-workspace) to illustrate ideas developed in *[Interpreting Physics in Video World Models](https://arxiv.org/abs/2602.07050)*, also inspired by [MIRA](https://mira-wm.com/blog-post/) and [Genie 3](https://deepmind.google/models/genie/).
<!-- /block -->

<!-- block:play-intro -->
Put the puck in the goal!

The game below is the real-time output of a 3.67M-parameter video transformer running in your browser. Arrow keys write to the model activations and "steer" the model's hallucinations.
<!-- /block -->

<!-- block:play-takeaway -->
This is not perfect but clearly has grokked the basic physics of the game, including collisions, wall bounces, and scoring. Notably, **the model has never explicitly observed input controls**. These were discovered post-hoc via interpretability methods (J-Lens).
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

Some physical properties are easily accessible within the transformer's hidden state. A simple linear regression from block 5 activations, for example, can neatly predict X/Y coordinates of both the orange and the blue ball. See the cross-hairs below for decoded X/Y coordinates, overlaid on next frame prediction samples

Similarly, we can see this same layer will ["anticipate" a collision](#collision-representation) up to eight frames before contact.
<!-- /block -->

<!-- block:generalization-title -->
Generalizing to unseen scenarios
<!-- /block -->

<!-- block:generalization-intro -->
Is this model merely memorizing the training set, or is it able to learn some higher-level representation that generalizes beyond seen examples?


If you retrain this model with certain classes of samples entirely removed - for example, [collisions in the top-right quadrant](#collision-holdout) or [an entire wedge of movement directions](#direction-holdout) - it turns out the model can still accurately predict what will happen. This is fairly significant and implies the model has learned some higher-level representation that captures entities and their relationships, as opposed to merely interpolating between previously-seen samples.
<!-- /block -->

<!-- block:jacobian-title -->
Finding steerable directions with the Jacobian lens.
<!-- /block -->

<!-- block:jacobian-lens -->
Reading x/y locations or collision anticipation from weights is interesting yet limited.

Using Anthropic's recently-published [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html#the-jacobian-lens), however, we can uncover a *causal* mechanism in the model's computation that controls ball velocity, then manipulate this to change the transformer's output.

Concretely, we can trace how block-5 activations affect the next frame, then average those gradients into reusable motion directions. [See the implementation](https://github.com/jayhack/blocket-league/blob/main/blocket_league/pixel_probe.py#L130-L149).
<!-- /block -->

<!-- block:causal-title -->
These variables are causal. Write to them and the hallucination changes.
<!-- /block -->

<!-- block:causal-intervention -->
Write the recovered +x direction for four frames, then stop. By frame 12, the player disc is 3.51 pixels farther right on average across 256 unseen worlds, and 85.9% move in the intended direction. A random activation direction has almost no effect.
<!-- /block -->

<!-- block:brain-surgery-title -->
This is a video game. You play it through "brain surgery."
<!-- /block -->

<!-- block:brain-surgery -->
If you map your keyboard to these causal direction probes, it becomes a simple video game. Note that the model has exclusively trained on raw pixels - we are not manipulating the model's input, but rather it's own internal representation for what it has "discovered" about motion.

This demo (repeated from above) loads the full 3.67M-parameter model into your browser.
<!-- /block -->

<!-- block:representation-title -->
Motion direction forms a ring in activation space
<!-- /block -->

<!-- block:representation-depth -->
A further insight about the model's internal activations is that motion direction in the transformer's block-5 activations has a ring-like topology.

If you average the block-5 activations by motion direction and run PCA, they wrap into a circular manifold, with nearby directions in the world colocated within the model.

The geometry of the representation mirrors the thing it represents: turn the physical direction smoothly, and the model moves smoothly around its internal ring.
<!-- /block -->

<!-- block:conclusion-title -->
Conclusion
<!-- /block -->

<!-- block:conclusion -->
Even this toy model builds a higher-level internal state we can read, that generalizes beyond its training examples, and that interpretability techniques available in 2026 let us causally influence.

I think this scales. In the spirit of Anthropic's [global-workspace view](https://transformer-circuits.pub/2026/workspace/), models trained over trillions of tokens seem to develop detailed internal worlds that are isomorphic to the reality they observe. Mapping these worlds and learning how they shape behavior may be one of the most exciting and consequential scientific projects of the AI era.
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
