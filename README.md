# 2D Car Racing Game with Reinforcement Learning

A complete implementation of a 2D car racing game where an AI agent learns to drive around a track using Deep Q-Learning (DQN), without any manual instructions.

## Features

- **2D Physics Simulation**: Realistic car physics with acceleration, friction, and rotation
- **Deep Q-Learning (DQN)**: Neural network-based RL agent that learns from experience
- **Radar Sensors**: 5 distance sensors that detect track boundaries
- **Visual Training**: Watch the car learn in real-time with Pygame rendering
- **No Human Input**: The car learns completely autonomously through trial and error

## How It Works

### State Space (8 values)
- 5 radar distance measurements (normalized)
- Current speed (normalized)
- Sin and cos of current angle

### Action Space (7 discrete actions)
0. Do nothing
1. Accelerate
2. Brake
3. Turn left
4. Turn right
5. Accelerate + Turn left
6. Accelerate + Turn right

### Reward System
- **Distance reward**: +0.1 × distance traveled per step
- **Survival bonus**: +0.1 per step alive
- **Speed penalty**: -0.5 if speed < 1 (encourages movement)
- **Crash penalty**: -100 (heavy penalty for hitting walls)

### Neural Network Architecture
```
Input (8) → Dense(128) → Dense(128) → Dense(64) → Output(7)
All hidden layers use ReLU activation
```

### Training Algorithm
- **Algorithm**: Deep Q-Learning with experience replay
- **Exploration**: ε-greedy (starts at 1.0, decays to 0.01)
- **Memory**: Replay buffer of 10,000 experiences
- **Batch size**: 64
- **Discount factor (γ)**: 0.95
- **Target network**: Updated every 10 training steps

## Requirements

```bash
pip install pygame numpy torch --break-system-packages
```

## Usage

### Training the Model

**Option 1: Standard Training (with visualization)**

Run the training script to start the learning process:

```bash
python car_racing_rl.py
```

**Option 2: Fast Training (recommended)**

Use the fast version with speed control:

```bash
# Normal speed with visualization
python car_racing_rl_fast.py

# 5x speed (faster training, still visible)
python car_racing_rl_fast.py --speed 5

# 10x speed (very fast, harder to see)
python car_racing_rl_fast.py --speed 10

# Headless mode (no rendering, maximum speed!)
python car_racing_rl_fast.py --headless --episodes 500

# Custom number of episodes
python car_racing_rl_fast.py --speed 5 --episodes 300
```

The car will start by exploring randomly and gradually learn to navigate the track. You'll see:
- Live visualization of the car learning (unless headless)
- Episode statistics (reward, distance, epsilon)
- Best distance achieved so far
- Model automatically saves every 50 episodes

Training typically takes 100-500 episodes for decent performance, though you may see improvement much earlier.

**Controls during training:**
- ESC: Quit early (model will be saved automatically)

**Note for Windows users**: The model will be saved in the same directory as the script with the filename `car_racing_model.pth`.

### Testing the Trained Model

After training, test the trained agent:

```bash
python test_model.py
```

This runs the trained model without exploration, showing how well it learned to drive.

**Controls during testing:**
- R: Reset/restart the lap
- ESC: Quit

## Expected Learning Progress

- **Episodes 1-50**: Random exploration, frequent crashes
- **Episodes 50-150**: Car learns to follow the track, still crashes occasionally
- **Episodes 150-300**: Smooth driving, completes multiple laps
- **Episodes 300+**: Optimizes for speed and efficiency

## Customization

You can modify various parameters in `car_racing_rl.py`:

### Track Design
- Edit the `Track` class to create different track shapes
- Change the ellipse parameters for different oval shapes
- Add checkpoints for better reward shaping

### Training Hyperparameters
```python
# In DQNAgent class
self.gamma = 0.95              # Discount factor
self.epsilon_decay = 0.995     # Exploration decay rate
self.learning_rate = 0.001     # Neural network learning rate
self.batch_size = 64           # Training batch size
```

### Car Physics
```python
# In Car class
self.max_speed = 8            # Maximum speed
self.acceleration = 0.3       # Acceleration rate
self.friction = 0.05          # Friction/deceleration
self.rotation_speed = 4       # Turning speed
```

### Reward Structure
```python
# In train() function
distance_reward = car.distance_traveled - old_distance
reward += distance_reward * 0.1  # Modify multiplier
reward += 0.1                    # Survival bonus
reward -= 0.5                    # Speed penalty
reward = -100                    # Crash penalty
```

## File Structure

- `car_racing_rl.py` - Main training script with full implementation
- `car_racing_rl_fast.py` - **Recommended**: Training script with speed control and headless mode
- `test_model.py` - Script to test the trained model
- `car_racing_model.pth` - Saved model weights (created after training)
- `requirements.txt` - Python dependencies

## Advanced Features to Add

1. **Multiple tracks**: Create different track designs for variety
2. **Checkpoints**: Add intermediate goals for better reward shaping
3. **Lap timing**: Track and optimize lap completion times
4. **Opponent cars**: Add other cars to race against
5. **Continuous actions**: Use continuous control instead of discrete
6. **Better algorithms**: Try PPO, A3C, or SAC for potentially better performance

## Troubleshooting

**Car doesn't improve after many episodes:**
- Increase training time (some randomness in learning)
- Adjust learning rate or reward structure
- Check that epsilon is decaying properly

**Training is too slow:**
- Reduce FPS or remove rendering during training
- Use GPU if available (PyTorch will automatically use it)
- Reduce replay buffer size or batch size

**Car learns to drive in circles:**
- Adjust reward to better encourage track following
- Add checkpoints to guide the car around the track
- Increase penalty for staying in one area

## Technical Details

The implementation uses:
- **Pygame**: For 2D rendering and visualization
- **PyTorch**: For the neural network and training
- **NumPy**: For numerical computations

The DQN agent uses experience replay to break correlation between consecutive experiences and a target network to stabilize training. The ε-greedy exploration strategy balances exploration (trying new actions) with exploitation (using learned knowledge).

## Usage

Free to use and modify for learning purposes. Inspired by classic reinforcement learning racing games and tutorials.
