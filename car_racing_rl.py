import pygame
import numpy as np
import math
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
CAR_WIDTH, CAR_HEIGHT = 40, 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = 0
        self.speed = 0
        self.max_speed = 8
        self.acceleration = 0.3
        self.friction = 0.05
        self.rotation_speed = 4
        self.alive = True
        self.distance_traveled = 0
        self.checkpoints_passed = 0
        self.last_checkpoint = -1
        
        # Radar sensors (5 sensors at different angles)
        self.radar_angles = [-60, -30, 0, 30, 60]
        self.radar_distances = [0] * len(self.radar_angles)
        self.max_radar_distance = 200
        
    def update(self, action, track):
        if not self.alive:
            return
        
        # Actions: 0=nothing, 1=accelerate, 2=brake, 3=left, 4=right, 5=accel+left, 6=accel+right
        if action in [1, 5, 6]:  # Accelerate
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        elif action == 2:  # Brake
            self.speed = max(self.speed - self.acceleration * 2, 0)
        
        if action in [3, 5]:  # Turn left
            if self.speed > 0.5:
                self.angle -= self.rotation_speed
        elif action in [4, 6]:  # Turn right
            if self.speed > 0.5:
                self.angle += self.rotation_speed
        
        # Apply friction
        self.speed = max(self.speed - self.friction, 0)
        
        # Update position
        old_x, old_y = self.x, self.y
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)
        
        # Calculate distance traveled
        dx = self.x - old_x
        dy = self.y - old_y
        self.distance_traveled += math.sqrt(dx**2 + dy**2)
        
        # Update radar
        self.update_radar(track)
        
        # Check collision
        if self.check_collision(track):
            self.alive = False
            
    def update_radar(self, track):
        for i, angle_offset in enumerate(self.radar_angles):
            angle = math.radians(self.angle + angle_offset)
            
            # Cast ray
            distance = 0
            x, y = self.x, self.y
            
            while distance < self.max_radar_distance:
                x += 2 * math.cos(angle)
                y += 2 * math.sin(angle)
                distance += 2
                
                # Check if hit wall
                if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
                    break
                
                if not track.is_on_track(int(x), int(y)):
                    break
            
            self.radar_distances[i] = distance / self.max_radar_distance
    
    def check_collision(self, track):
        # Check corners of the car
        corners = self.get_corners()
        for corner in corners:
            if not track.is_on_track(int(corner[0]), int(corner[1])):
                return True
        return False
    
    def get_corners(self):
        rad = math.radians(self.angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)
        
        corners = []
        for dx, dy in [(-CAR_WIDTH/2, -CAR_HEIGHT/2), (CAR_WIDTH/2, -CAR_HEIGHT/2),
                       (CAR_WIDTH/2, CAR_HEIGHT/2), (-CAR_WIDTH/2, CAR_HEIGHT/2)]:
            x = self.x + dx * cos_a - dy * sin_a
            y = self.y + dx * sin_a + dy * cos_a
            corners.append((x, y))
        return corners
    
    def get_state(self):
        # State: radar distances (5) + speed (1) + angle sin/cos (2) = 8 values
        state = list(self.radar_distances)
        state.append(self.speed / self.max_speed)
        state.append(math.sin(math.radians(self.angle)))
        state.append(math.cos(math.radians(self.angle)))
        return np.array(state, dtype=np.float32)
    
    def draw(self, screen):
        if not self.alive:
            color = RED
        else:
            color = GREEN
        
        # Draw car body
        corners = self.get_corners()
        pygame.draw.polygon(screen, color, corners)
        
        # Draw direction indicator
        rad = math.radians(self.angle)
        end_x = self.x + CAR_WIDTH * math.cos(rad)
        end_y = self.y + CAR_WIDTH * math.sin(rad)
        pygame.draw.line(screen, YELLOW, (self.x, self.y), (end_x, end_y), 2)
        
        # Draw radar (optional, for debugging)
        # for i, angle_offset in enumerate(self.radar_angles):
        #     angle = math.radians(self.angle + angle_offset)
        #     distance = self.radar_distances[i] * self.max_radar_distance
        #     end_x = self.x + distance * math.cos(angle)
        #     end_y = self.y + distance * math.sin(angle)
        #     pygame.draw.line(screen, BLUE, (self.x, self.y), (end_x, end_y), 1)


class Track:
    def __init__(self):
        # Create a simple oval track
        self.track_surface = pygame.Surface((WIDTH, HEIGHT))
        self.track_surface.fill(GRAY)
        
        # Draw outer boundary (green grass)
        pygame.draw.ellipse(self.track_surface, BLACK, [50, 50, 700, 500])
        
        # Draw inner boundary (grass hole)
        pygame.draw.ellipse(self.track_surface, GRAY, [150, 150, 500, 300])
        
        # Starting position
        self.start_x = 400
        self.start_y = 100
        
        # Checkpoints (optional for better reward shaping)
        self.checkpoints = [
            (700, 300),  # Right
            (400, 500),  # Bottom
            (100, 300),  # Left
            (400, 100),  # Top (start)
        ]
        
    def is_on_track(self, x, y):
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return False
        color = self.track_surface.get_at((x, y))
        return color[:3] == BLACK
    
    def draw(self, screen):
        screen.blit(self.track_surface, (0, 0))
        
        # Draw checkpoints
        for cp in self.checkpoints:
            pygame.draw.circle(screen, BLUE, cp, 5)


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.update_target_every = 10
        self.train_step = 0
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Main network
        self.model = DQN(state_size, action_size).to(self.device)
        # Target network
        self.target_model = DQN(state_size, action_size).to(self.device)
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.argmax().item()
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0
        
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor(np.array([m[0] for m in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([m[1] for m in minibatch])).to(self.device)
        rewards = torch.FloatTensor(np.array([m[2] for m in minibatch])).to(self.device)
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([m[4] for m in minibatch])).to(self.device)
        
        # Current Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.train_step += 1
        if self.train_step % self.update_target_every == 0:
            self.update_target_network()
        
        return loss.item()
    
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def train():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D Car Racing RL")
    clock = pygame.time.Clock()
    
    track = Track()
    agent = DQNAgent(state_size=8, action_size=7)
    
    episode = 0
    max_episodes = 1000
    max_steps = 2000
    
    best_distance = 0
    episode_rewards = []
    
    font = pygame.font.Font(None, 30)
    
    running = True
    
    while running and episode < max_episodes:
        car = Car(track.start_x, track.start_y)
        state = car.get_state()
        total_reward = 0
        step = 0
        
        episode_done = False
        
        while not episode_done and step < max_steps:
            clock.tick(FPS)
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    episode_done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        episode_done = True
            
            # Agent selects action
            action = agent.act(state)
            
            # Car performs action
            old_distance = car.distance_traveled
            car.update(action, track)
            
            # Calculate reward
            reward = 0
            
            if not car.alive:
                reward = -100  # Heavy penalty for crashing
                episode_done = True
            else:
                # Reward for distance traveled
                distance_reward = car.distance_traveled - old_distance
                reward += distance_reward * 0.1
                
                # Small reward for staying alive
                reward += 0.1
                
                # Penalty for going too slow
                if car.speed < 1:
                    reward -= 0.5
            
            next_state = car.get_state()
            total_reward += reward
            
            # Remember experience
            agent.remember(state, action, reward, next_state, episode_done)
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            state = next_state
            step += 1
            
            # Render
            track.draw(screen)
            car.draw(screen)
            
            # Display info
            info_text = [
                f"Episode: {episode + 1}",
                f"Step: {step}",
                f"Reward: {total_reward:.1f}",
                f"Distance: {car.distance_traveled:.1f}",
                f"Speed: {car.speed:.2f}",
                f"Epsilon: {agent.epsilon:.3f}",
                f"Best: {best_distance:.1f}"
            ]
            
            y_offset = 10
            for text in info_text:
                text_surface = font.render(text, True, WHITE)
                screen.blit(text_surface, (10, y_offset))
                y_offset += 30
            
            pygame.display.flip()
        
        # Episode finished
        episode_rewards.append(total_reward)
        if car.distance_traveled > best_distance:
            best_distance = car.distance_traveled
            print(f"New best distance: {best_distance:.1f} at episode {episode + 1}")
        
        agent.decay_epsilon()
        episode += 1
        
        # Save model periodically
        if episode % 50 == 0:
            torch.save(agent.model.state_dict(), 'car_racing_model.pth')
            print(f"Model saved at episode {episode}")
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
            print(f"Episode {episode}, Avg Reward (last 10): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
    
    # Save model
    torch.save(agent.model.state_dict(), 'car_racing_model.pth')
    print(f"\nTraining complete! Model saved to 'car_racing_model.pth'")
    print(f"Best distance achieved: {best_distance:.1f}")
    
    pygame.quit()


if __name__ == "__main__":
    print("2D Car Racing with Reinforcement Learning")
    print("=" * 50)
    print("The car will learn to race around the track!")
    print("Press ESC to quit early")
    print("=" * 50)
    print("\nStarting training...")
    
    train()
