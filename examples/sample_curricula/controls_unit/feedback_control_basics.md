# Feedback Control Basics

A short introductory unit on feedback control, from single loops to groups of cooperating agents.
Each section lists the sections it builds on.

## 1. Open-Loop and Closed-Loop Control

A control system makes a physical quantity, the **output**, follow a desired value called the
**reference** or **setpoint**. The device being controlled is the **plant**, the signal applied to it
is the **control input**, and the difference between the reference and the measured output is the
**error**: e = r − y.

In **open-loop control** the input is computed from the reference alone, without measuring the output.
A toaster that heats for a fixed time is open loop: if the bread is thicker than usual, nothing
corrects the result. Open-loop control is simple and cheap, but it cannot reject **disturbances**
(unexpected inputs such as a gust of wind or a change in load) and it depends on an accurate model
of the plant.

In **closed-loop control**, also called **feedback control**, a sensor measures the output and the
controller acts on the error. A home thermostat is closed loop: it measures the room temperature and
switches the heater according to the difference from the setpoint. Feedback makes a system less
sensitive to disturbances and to errors in the plant model, at the cost of needing a sensor and of
possibly making the system unstable if the controller is badly tuned.

**Negative feedback** means the controller acts to reduce the error: if the output is above the
reference, the input is decreased. Almost all practical controllers use negative feedback;
positive feedback amplifies deviations instead of correcting them.

## 2. First-Order Systems and the Time Constant

Prerequisites: 1. Open-Loop and Closed-Loop Control

Many plants, such as a heated room, a DC motor's speed or a charging RC circuit, are well described
by a **first-order differential equation**:

τ · dy/dt + y = K · u

Here u is the input, y the output, **K** the **steady-state gain** and **τ** (tau) the **time
constant**, measured in seconds. The corresponding transfer function is G(s) = K / (τs + 1).

The **step response** is the output after the input jumps from 0 to a constant value U at t = 0.
For a first-order system it is y(t) = K·U·(1 − e^(−t/τ)). The output rises without overshoot and
approaches its final value K·U exponentially:

- at t = τ the output has reached about **63.2 %** of its final value;
- at t = 2τ, about 86.5 %; at t = 3τ, about 95 %;
- at t = 4τ, about **98 %**, which is why 4τ is a common estimate of the **settling time**.

A larger time constant means a slower system. The time constant can be read from a measured step
response as the time at which the output reaches 63.2 % of its total change, which is a quick way
to identify a first-order model from an experiment.

## 3. Stability and Poles

Prerequisites: 2. First-Order Systems and the Time Constant

A linear system is **stable** (more precisely, bounded-input bounded-output stable) if every bounded
input produces a bounded output. Stability is decided by the **poles** of the transfer function:
the values of s at which the denominator is zero.

- A continuous-time linear system is stable if and only if **all its poles have strictly negative
  real parts**, that is, they lie in the left half of the complex plane.
- A pole with a positive real part makes the response grow without bound: the system is unstable.
- Poles on the imaginary axis (real part exactly zero) give responses that neither decay nor grow,
  for example a sustained oscillation; this is called marginal stability.

The first-order system G(s) = K / (τs + 1) has a single pole at **s = −1/τ**. For a positive time
constant the pole is negative, so the system is stable, and the further the pole is from the
imaginary axis, the faster the response decays.

Poles that are complex, s = −σ ± jω, produce oscillations at frequency ω that decay at a rate set by
σ. Feedback moves the poles: the closed-loop poles are the roots of 1 + C(s)G(s) = 0, where C(s) is
the controller. This is why a controller can stabilise an unstable plant, and also why too much gain
can destabilise a plant that was stable on its own.

## 4. PID Control

Prerequisites: 1. Open-Loop and Closed-Loop Control; 3. Stability and Poles

The **PID controller** is the most widely used feedback controller. It computes the input from the
error e(t) with three terms:

u(t) = Kp·e(t) + Ki·∫e dt + Kd·de(t)/dt

- The **proportional** term (gain Kp) reacts to the present error. With proportional control alone,
  a first-order plant with gain K keeps a **steady-state error** after a step in the reference:
  e_ss = 1 / (1 + Kp·K), as a fraction of the step. Increasing Kp reduces this error but never
  removes it, and in higher-order plants too much proportional gain causes oscillation.
- The **integral** term (gain Ki) reacts to the accumulated past error. As long as any error remains,
  the integral keeps growing, so a stable closed loop with integral action has **zero steady-state
  error** for a constant reference or a constant disturbance.
- The **derivative** term (gain Kd) reacts to the rate of change of the error, predicting its future
  trend. It adds **damping**, reducing overshoot, but it amplifies high-frequency measurement noise,
  so in practice the derivative is low-pass filtered.

When the actuator saturates (a valve fully open, a motor at maximum voltage), the integral keeps
accumulating error that the actuator cannot act on. This **integrator windup** causes large
overshoot once the error changes sign; **anti-windup** schemes stop or limit integration while the
actuator is saturated.

Tuning means choosing Kp, Ki and Kd. A classic starting point is the **Ziegler–Nichols** method,
which derives the gains from the gain at which the loop starts to oscillate steadily and the period
of that oscillation; the result is usually refined by hand, because it tends to give aggressive,
oscillatory responses.

## 5. Consensus in Multi-Agent Systems

Prerequisites: 3. Stability and Poles

A **multi-agent system** is a group of agents, such as robots or drones, that each control
themselves using information from their **neighbours**. Who communicates with whom is described by
a **graph**: agents are nodes, and an edge connects two agents that exchange information.

In the **consensus problem**, the agents must agree on a common value (a heading, a meeting point, a
clock time) using only neighbour information. The standard consensus protocol for single-integrator
agents is:

dx_i/dt = Σ over neighbours j of (x_j − x_i)

Each agent moves towards its neighbours' values. Stacking all agents into one vector x gives
**dx/dt = −L·x**, where **L** is the **graph Laplacian**: L = D − A, with A the adjacency matrix and D
the diagonal matrix of node degrees (the number of neighbours of each agent).

For an undirected graph, the Laplacian has these properties:

- it always has an eigenvalue **0**, with the all-ones vector as eigenvector, and all its other
  eigenvalues are non-negative;
- the eigenvalue 0 is simple (appears only once) if and only if the graph is **connected**;
- if the graph is connected, the protocol drives every agent to the **average of the initial
  values**, because the sum of the states never changes;
- the speed of convergence is set by the **second-smallest eigenvalue λ2**, called the **algebraic
  connectivity** or **Fiedler value**: a better-connected graph has a larger λ2 and agrees faster.

If the graph is not connected, each connected group reaches its own agreement and the groups do not
agree with each other. **Formation control** extends consensus: with desired offsets h, the protocol
dx/dt = −L·(x − h) makes the agents agree on a common reference point while keeping their offsets
from it, so the group converges to the desired shape.
