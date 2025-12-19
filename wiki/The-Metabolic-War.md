# The Metabolic War: Evolution in Silicon

*"It is not the strongest of the species that survives, nor the most intelligent. It is the one most adaptable to change."* — Charles Darwin

*"So we said: let's make the AI prove it deserves to exist."* — Project Cyberspore

---

## The Problem with Traditional Training

Usually, when you train a neural network, you do gradient descent. You calculate how wrong the network is, figure out which way to adjust the weights to make it less wrong, and take a tiny step in that direction. Repeat a million times.

It works! But here's the issue: gradient descent optimizes for **accuracy**, not **efficiency**. The network will happily use all 2 billion parameters if you let it. It has no incentive to be sparse. No incentive to be simple. It just wants to minimize loss.

But we wanted something different. We wanted a network that's **forced to be efficient**. That has to **fight for its right to exist**.

So we borrowed an idea from biology: **competition**.

---

## The Setup: Host vs. Parasite

Imagine you have two networks:

### The Host (Dense Network)
- Big, bloated, uses 32-bit floating-point weights
- Has 2 billion parameters
- Works perfectly—it's the original trained model (like Gemma-2B)
- But it's too big to fit on your laptop

### The Parasite (Cyberspore/TSSN Network)
- Small, sparse, uses ternary weights {-1, 0, +1}
- Starts with random connections
- Doesn't work at all (at first)
- But it's tiny and fast

Now here's the game: **The parasite tries to replace the host.**

---

## How the War Works

We don't do traditional training. Instead, we do **competitive evolution**:

### Step 1: The Incision
We surgically remove a layer from the host network. Maybe the "Feed-Forward Down Projection" in layer 12. We cut it out. Now there's a hole in the network.

The host is wounded. It can't produce the correct output anymore.

### Step 2: The Infection
We inject the parasite (a TSSN layer) into that hole. The parasite's job is to mimic what the host's original layer used to do.

At first, the parasite is terrible. It's just random ternary connections. The output is garbage.

### Step 3: The Selection Pressure
Here's where it gets interesting. We measure the **Mean Squared Error (MSE)** between:
- What the original host layer would have output
- What the parasite actually outputs

If MSE is high: **The parasite dies.** We try a new random parasite.

If MSE is low: **The parasite survives!** It gets to stay in the network.

But there's a catch: the parasite is also penalized for using too many connections. We want it **sparse**. So the fitness function is:

`
Fitness = 1 / (MSE + λ·sparsity_penalty)
`

Where λ controls how much we care about sparsity vs. accuracy.

### Step 4: Evolution
We don't just try one parasite. We try **thousands**. We use a genetic algorithm:

1. **Generate Population**: Create 100 random TSSN candidates
2. **Evaluate**: Run each one, measure MSE
3. **Select**: Keep the best 20
4. **Mutate**: Randomly flip some ternary weights (-1 → 0, 0 → +1, etc.)
5. **Crossover**: Combine two good candidates to make a child
6. **Repeat**: Go back to step 2

After thousands of generations, we find a parasite that's:
- Just as accurate as the original dense layer (MSE ≈ 0)
- But 70-96% sparse!

### Step 5: Move to the Next Layer
Now we "heal" the host by accepting this parasite. We move to the next layer and repeat.

Layer by layer, the parasite takes over. Eventually, the entire network is TSSN.

---

## Why This Works (The Biology Analogy)

In nature, parasites evolve to be **incredibly efficient**. They have to be! They're stealing resources from the host, so every bit of energy matters.

The same thing happens here. By forcing the TSSN network to:
1. **Compete** with a functional dense network
2. **Survive** only if it's both accurate AND sparse

...we create evolutionary pressure that traditional training doesn't have.

It's like asking: "If you had to throw away 90% of this neural network, which 10% would you keep?"

Evolution figures it out.

---

## The Carrying Capacity

Here's a concept from ecology: **carrying capacity**. It's the maximum population an environment can sustain.

For Cyberspore, the "carrying capacity" is the maximum sparsity the network can handle before it collapses.

### Our Experimental Results:

**Phase 1-4 (Simple Ternary)**
- Carrying Capacity: ~**70% sparsity**
- Beyond this, the parasite couldn't match the host's complexity

**Phase 5 (Exotic Functions)**
- We tried using more diverse ternary logic functions (the BTL library)
- Carrying Capacity: ~**80% sparsity**
- Better, but still hit a wall

**Phase 6 (Composite Logic)**
- We let TSSN neurons **chain** multiple logic gates
- Carrying Capacity: ~**96% sparsity** 🎉
- This was the breakthrough!

The key insight: **Functional diversity increases carrying capacity.**

If you give the parasite more "tools" (more types of logic gates, more ways to combine them), it can approximate more complex functions with fewer connections.

---

## The MTEB Gate

We called the collapse point the **"MTEB Gate"** (named after the semantic similarity benchmark we use).

When sparsity exceeds the carrying capacity:
- MSE spikes
- The model loses semantic understanding
- It starts outputting gibberish

But right at the edge—say, 94-96% sparsity—something beautiful happens: the model is **maximally compressed**. It's thrown away everything except the absolute essential structure.

It's like a diamond. You apply pressure (evolutionary selection), and what emerges is the crystalline core of intelligence.

---

## The Healing Phase

After we find a good TSSN candidate, we don't just drop it in and move on. We do **healing**.

Remember: the parasite was evolved to match one specific layer's output. But now it's part of the full network. The layers before and after it have changed.

So we run a few hundred more generations where:
- We slightly adjust the parasite's weights
- We keep the sparsity level fixed (no new connections)
- We let it fine-tune to the actual network context

This is like physical therapy after surgery. The parasite needs to "settle in" to its new environment.

---

## Why Not Just Use Pruning?

You might ask: "Why not just train the dense network, then prune it?"

Good question! That's actually what most people do. But pruning has problems:

### Problem 1: Local Optima
If you train a dense network, it finds a solution in "dense space." When you prune, you're trying to project that solution into "sparse space," but there's no guarantee the projection is good.

### Problem 2: No Functional Diversity
Pruning just removes weights. It doesn't explore alternative logic structures (like composite gates).

### Problem 3: No Evolutionary Dynamics
Pruning is a one-shot process. Evolution is iterative. It can explore weird solutions that pruning would never find.

Our approach is different: **we evolve directly in sparse space**. We never waste time in dense space. The parasite is born sparse and stays sparse.

---

## The Steady-State Algorithm

In later phases (V4), we switched to a **steady-state evolutionary algorithm**:

Instead of:
1. Generate full population
2. Evaluate all
3. Select best
4. Generate new population

We do:
1. Keep a circular buffer of N candidates
2. As soon as one finishes evaluating, spawn a new mutation
3. If the new candidate is better than the worst in the buffer, replace it
4. Repeat forever

This keeps the GPU/CPU saturated. No waiting. Maximum throughput.

It's like a continuous culture in biology—the population is always evolving, never static.

---

## The Numbers (What We Measured)

Here's some real data from Phase 6:

**85% Sparsity (The Barrier)**
- Initial MSE: 0.166 (terrible)
- After 1000 generations: 0.0023 (pretty good)
- Composite function: [467, 233]

**96% Sparsity (The Limit)**
- Initial MSE: 0.412 (disaster)
- After 5000 generations: **0.00000027** (basically zero!)
- Composite function: [233, 179]

Think about that. We threw away **96% of the network**, and the error is *2.7 × 10⁻⁷*. That's insane.

---

## The Philosophy

Feynman once said: *"What I cannot create, I do not understand."*

With Cyberspore, we're doing the opposite: **We let evolution create it, then we try to understand what it made.**

Because here's the thing: we don't fully understand why composite function [233, 179] works so well at 96% sparsity. We can measure that it does. We can verify the math. But the *why*—the deep reason—is still a mystery.

And that's exciting! It means there's more to discover.

---

## Next Steps

- See the quantitative results: [[Research Results]]
- Understand the hardware implications: [[Hardware Acceleration]]
- Dream about the future: [[Future Directions]]

---

*"It doesn't matter how beautiful your theory is. It doesn't matter how smart you are. If it doesn't agree with experiment, it's wrong."* — Richard Feynman

*Our experiment said: 96% sparsity works. Now we're trying to understand why.*
