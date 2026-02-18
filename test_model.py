import pygame
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import Car and Track classes
import sys
sys.path.append('/home/claude')

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
        
        # Radar sensors
        self.radar_angles = [-60, -30, 0, 30, 60]
        self.radar_distances = [0] * len(self.radar_angles)
        self.max_radar_distance = 200
        
    def update(self, action, track):
        if not self.alive:
            return
        
        # Actions: 0=nothing, 1=accelerate, 2=brake, 3=left, 4=right, 5=accel+left, 6=accel+right
        if action in [1, 5, 6]:
            self.speed = min(self.speed + self.acceleration, self.max_speed)
        elif action == 2:
            self.speed = max(self.speed - self.acceleration * 2, 0)
        
        if action in [3, 5]:
            if self.speed > 0.5:
                self.angle -= self.rotation_speed
        elif action in [4, 6]:
            if self.speed > 0.5:
                self.angle += self.rotation_speed
        
        self.speed = max(self.speed - self.friction, 0)
        
        old_x, old_y = self.x, self.y
        rad = math.radians(self.angle)
        self.x += self.speed * math.cos(rad)
        self.y += self.speed * math.sin(rad)
        
        dx = self.x - old_x
        dy = self.y - old_y
        self.distance_traveled += math.sqrt(dx**2 + dy**2)
        
        self.update_radar(track)
        
        if self.check_collision(track):
            self.alive = False
            
    def update_radar(self, track):
        for i, angle_offset in enumerate(self.radar_angles):
            angle = math.radians(self.angle + angle_offset)
            distance = 0
            x, y = self.x, self.y
            
            while distance < self.max_radar_distance:
                x += 2 * math.cos(angle)
                y += 2 * math.sin(angle)
                distance += 2
                
                if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
                    break
                
                if not track.is_on_track(int(x), int(y)):
                    break
            
            self.radar_distances[i] = distance / self.max_radar_distance
    
    def check_collision(self, track):
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
        state = list(self.radar_distances)
        state.append(self.speed / self.max_speed)
        state.append(math.sin(math.radians(self.angle)))
        state.append(math.cos(math.radians(self.angle)))
        return np.array(state, dtype=np.float32)
    
    def draw(self, screen):
        color = RED if not self.alive else GREEN
        
        corners = self.get_corners()
        pygame.draw.polygon(screen, color, corners)
        
        rad = math.radians(self.angle)
        end_x = self.x + CAR_WIDTH * math.cos(rad)
        end_y = self.y + CAR_WIDTH * math.sin(rad)
        pygame.draw.line(screen, YELLOW, (self.x, self.y), (end_x, end_y), 2)


class Track:
    def __init__(self):
        self.track_surface = pygame.Surface((WIDTH, HEIGHT))
        self.track_surface.fill(GRAY)
        
        pygame.draw.ellipse(self.track_surface, BLACK, [50, 50, 700, 500])
        pygame.draw.ellipse(self.track_surface, GRAY, [150, 150, 500, 300])
        
        self.start_x = 400
        self.start_y = 100
        
    def is_on_track(self, x, y):
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return False
        color = self.track_surface.get_at((x, y))
        return color[:3] == BLACK
    
    def draw(self, screen):
        screen.blit(self.track_surface, (0, 0))


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


def test_model(model_path='car_racing_model.pth'):
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("2D Car Racing RL - Testing Mode")
    clock = pygame.time.Clock()
    
    track = Track()
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN(state_size=8, action_size=7).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model file not found. Please train the model first.")
        return
    
    font = pygame.font.Font(None, 30)
    
    running = True
    laps = 0
    
    while running:
        car = Car(track.start_x, track.start_y)
        state = car.get_state()
        step = 0
        max_steps = 3000
        
        lap_done = False
        
        while not lap_done and step < max_steps and running:
            clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    lap_done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        lap_done = True
                    elif event.key == pygame.K_r:
                        lap_done = True  # Reset
            
            # Model selects action (no exploration)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
            action = q_values.argmax().item()
            
            car.update(action, track)
            
            if not car.alive:
                lap_done = True
            
            state = car.get_state()
            step += 1
            
            # Render
            track.draw(screen)
            car.draw(screen)
            
            # Display info
            info_text = [
                f"Testing Mode",
                f"Lap: {laps + 1}",
                f"Step: {step}",
                f"Distance: {car.distance_traveled:.1f}",
                f"Speed: {car.speed:.2f}",
                "",
                "Press R to reset",
                "Press ESC to quit"
            ]
            
            y_offset = 10
            for text in info_text:
                if text:
                    text_surface = font.render(text, True, WHITE)
                    screen.blit(text_surface, (10, y_offset))
                y_offset += 30
            
            pygame.display.flip()
        
        laps += 1
        print(f"Lap {laps} completed - Distance: {car.distance_traveled:.1f}, Steps: {step}")
    
    pygame.quit()


if __name__ == "__main__":
    print("2D Car Racing - Testing Trained Model")
    print("=" * 50)
    test_model()
