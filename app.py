import os, math, pygame, neat, pickle, visualize, argparse
from pygame import Vector2
import glob # For finding checkpoint files
from neat import Checkpointer # For saving/loading population state


TRACK = 'data/track.png'
WINDOW_SIZE = (600, 400)
MAX_STEPS_PER_CAR = 50000000
RAY_COUNT = 9                                           # calculates distance from 6 directions
RAY_LENGTH = 250   
CHECK_STEPS = 120       # Check if car is stuck every 120 steps (2 seconds)
MIN_DISPLACEMENT = 40.0 # Car must move at least 10 pixels in that time
TILE_SIZE = 20          # Size of the "exploration" grid tiles

# --- Checkpoint & Reward Constants ---
START_POS = (450, 373) 
FINISH_LINE_RECT = pygame.Rect(START_POS[0] - 5, START_POS[1] - 25, 10, 50) 
CHECKPOINT_1_POS = (279, 149) 
CHECKPOINT_1_RECT = pygame.Rect(CHECKPOINT_1_POS[0] - 5, CHECKPOINT_1_POS[1] - 30, 10, 60)
LAP_REWARD_BASE = 50000.0 # Base reward for a lap, divided by steps
# --- End Constants ---


pygame.init()
screen = pygame.display.set_mode(WINDOW_SIZE)
clock = pygame.time.Clock()

# load track
track_surf = pygame.image.load(TRACK).convert()
track_surf.set_colorkey((255, 255, 255))
track_mask = pygame.mask.from_surface(track_surf)

def is_on_road(x, y):
    w, h = track_surf.get_size()
    if x < 0 or x >= w or y < 0 or y >= h:
        return False

    return track_mask.get_at((int(x), int(y))) == 1 

class Car:
    def __init__(self, start_pos, start_angle = 0.0) -> None:
        self.pos = Vector2(start_pos)
        self.angle = start_angle
        self.speed = 0.0
        self.alive = True
        self.distance_travelled = 0.0
        self.steps = 0 
        self.max_speed = 4.0
        self.size = (20, 36)

        self.steps_since_last_check = 0
        self.last_check_pos = Vector2(start_pos)
        
        self.visited_tiles = set()
        start_tile = (int(self.pos.x / TILE_SIZE), int(self.pos.y / TILE_SIZE))
        self.visited_tiles.add(start_tile)
        try:
            # Load the original image
            original_image = pygame.image.load('data/car.png').convert_alpha()
            # Scale it to the desired size
            self.image = pygame.transform.scale(original_image, self.size)
        except pygame.error as e:
            print(f"Error loading 'data/car.png': {e}")
            print("Falling back to a red rectangle.")
            # Fallback: create a red surface if image fails to load
            self.image = pygame.Surface(self.size, pygame.SRCALPHA)
            self.image.fill((255, 0, 0))

        # --- Lap tracking attributes ---
        self.on_finish_line = True  # Start on the finish line
        self.lap_count = 0
        self.steps_at_lap_start = 0
        self.total_lap_bonus = 0.0
        self.passed_checkpoint_1 = False # Flag to track checkpoint
        # --- End lap tracking ---


    def update(self, steer, accel):
        if not self.alive:
            return

        # Steer, angle in range [-1, 1] -> tanh
        self.angle += steer * 2.5  # <-- REDUCED SENSITIVITY
        self.angle %= 360
        self.speed += accel * 0.2

        # clamp speed
        self.speed = max(0, min(self.speed, self.max_speed))
        # update position
        dir_vec = Vector2(math.sin(math.radians(self.angle)), -math.cos(math.radians(self.angle)))
        self.pos += dir_vec * self.speed

        self.distance_travelled += self.speed
        
        self.steps += 1 
        self.steps_since_last_check += 1

        current_tile = (int(self.pos.x / TILE_SIZE), int(self.pos.y / TILE_SIZE))
        self.visited_tiles.add(current_tile)

        # --- Lap completion logic --- 
        colliding_finish = FINISH_LINE_RECT.collidepoint(self.pos.x, self.pos.y)
        colliding_checkpoint_1 = CHECKPOINT_1_RECT.collidepoint(self.pos.x, self.pos.y)

        # 1. Check if car hits the checkpoint
        if colliding_checkpoint_1 and not self.on_finish_line:
            self.passed_checkpoint_1 = True

        # 2. Check if car hits the finish line *after* hitting the checkpoint
        if colliding_finish and not self.on_finish_line and self.passed_checkpoint_1:
            # Just crossed the finish line to complete a lap
            self.lap_count += 1
            self.on_finish_line = True
            
            # Calculate bonus
            steps_for_lap = self.steps - self.steps_at_lap_start
            if steps_for_lap > 0: # Avoid divide by zero
                bonus = LAP_REWARD_BASE / steps_for_lap
                self.total_lap_bonus += bonus

            # Reset timer for the next lap
            self.steps_at_lap_start = self.steps
            
            # Reset checkpoint for the next lap
            self.passed_checkpoint_1 = False 

        # 3. Check if car just left the finish line
        elif not colliding_finish and self.on_finish_line:
            # Just left the finish line
            self.on_finish_line = False
            # If this is the very first time leaving, set the lap start time
            if self.lap_count == 0 and self.steps > 0:
                self.steps_at_lap_start = self.steps
        # --- End lap logic ---

        if self.steps > MAX_STEPS_PER_CAR:
            self.alive = False
            
        if self.steps_since_last_check > CHECK_STEPS:
            displacement = (self.pos - self.last_check_pos).length()
            if self.alive and displacement < MIN_DISPLACEMENT:
                self.alive = False # Kill car if it's stuck or circling
            
            self.steps_since_last_check = 0
            self.last_check_pos = Vector2(self.pos.x, self.pos.y)        
            
        # collision checks -> Use bounding box for the car
        corners = [
                self.pos + Vector2(0, -9).rotate(self.angle),
                self.pos + Vector2(5, 9).rotate(self.angle),
                self.pos + Vector2(-5, 9).rotate(self.angle),
                ]

        for p in corners:
            x, y = int(p.x), int(p.y)
            if not is_on_road(x, y):
                self.alive = False
                break


    def cast_rays(self):
        readings = []
        start = self.pos
        base_angle = self.angle
        spread = 120

        for i in range(RAY_COUNT):
            ray_angle = base_angle - spread / 2 + (spread / (RAY_COUNT - 1)) * i 
            
            if RAY_COUNT == 1:
                ray_angle = base_angle
                
            rad = math.radians(ray_angle)            
            dx = math.sin(rad)
            dy = -math.cos(rad)
            dist = 0.0
            hit = False
            for d in range(1, RAY_LENGTH):
                x = int(start.x + dx * d)
                y = int(start.y + dy * d)

                if x < 0 or x >= track_surf.get_width() or y < 0 or y >= track_surf.get_height():
                    dist = d 
                    hit = True
                    break

                if not is_on_road(x, y):
                    dist = d
                    hit = True
                    break

            if not hit:
                dist = RAY_LENGTH

            readings.append(dist / RAY_LENGTH)

        return readings


    def draw(self, surf):
        # Rotate the pre-loaded image
        rotated = pygame.transform.rotate(self.image, self.angle)
        
        # Get the new bounding box, centered at the car's position
        rrect = rotated.get_rect(center = self.pos)
        
        # Blit the rotated image onto the main surface
        surf.blit(rotated, rrect.topleft)

            

# NEAT evaluation
def eval_genomes(genomes, config):
    nets = []
    cars = []
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Car(start_pos=START_POS, start_angle=0.0)) 
        genome.fitness = 0

    run = True
    while run and any(car.alive for car in cars):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        screen.blit(track_surf, (0, 0))

        pygame.draw.rect(screen, (0, 255, 0), FINISH_LINE_RECT, 2) 
        pygame.draw.rect(screen, (255, 0, 0), CHECKPOINT_1_RECT, 2) 
        
        for i, car in enumerate(cars):
            if not car.alive:
                continue
            inputs = car.cast_rays()
            inputs.append(car.speed / car.max_speed)
            outputs = nets[i].activate(inputs)
            steer, accel = outputs[0], outputs[1]
            car.update(steer, accel)
            car.draw(screen)
            # Reduce the exploration bonus. It's now a small reward to
            # encourage progress, not the main goal.
            exploration_bonus = len(car.visited_tiles) * 100
            
            # Make distance travelled a much larger part of the reward.
            # Since distance = speed * time, this rewards cars that
            # maintain a high speed for a long time.
            speed_bonus = car.distance_travelled * 2
            
            # The lap bonus is already a great speed incentive, keep it.
            lap_bonus = car.total_lap_bonus
            
            genomes[i][1].fitness = speed_bonus + exploration_bonus 

        pygame.display.flip()
        clock.tick(60)

def run(config_file):
    TOTAL_GENERATIONS = 35

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    
    # --- Check for existing checkpoints ---
    latest_checkpoint = None
    checkpoint_files = glob.glob('neat-checkpoint-*') # Find all checkpoint files
    if checkpoint_files:
        # Sort files by generation number (e.g., neat-checkpoint-9)
        checkpoint_files.sort(key=lambda f: int(f.split('-')[-1]))
        latest_checkpoint = checkpoint_files[-1]
        
    if latest_checkpoint:
        print(f"*** Resuming training from checkpoint: {latest_checkpoint} ***")
        p = Checkpointer.restore_checkpoint(latest_checkpoint)
    else:
        print("*** Starting new training session ***")
        p = neat.Population(config)
    
    # Add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    # This will save a checkpoint every 5 generations
    checkpointer = Checkpointer(generation_interval=5, 
                                time_interval_seconds=None, 
                                filename_prefix='neat-checkpoint-')
    p.add_reporter(checkpointer)
    
    # --- Define node names ONCE outside the loop ---
    node_names = {
        -1: 'Ray 1', -2: 'Ray 2', -3: 'Ray 3', 
        -4: 'Ray 4', -5: 'Ray 5', -6: 'Ray 6',
        -7: 'Ray 7', -8: 'Ray 8', -9: 'Ray 9',
        -10: 'Speed',
         0: 'Steer',  1: 'Accel'
    }
    
    # --- Create a directory for generational graphs ---
    graph_dir = "generational-graphs"
    if not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    print(f"Saving generation graphs to: {graph_dir}/")
    
    
    generations_left = TOTAL_GENERATIONS - p.generation
    
    # --- NEW: Run one generation at a time ---
    if generations_left > 0:
        print(f"--- Running for {generations_left} more generations (Current: {p.generation}, Target: {TOTAL_GENERATIONS}) ---")
        
        # We manually loop for the remaining generations
        for i in range(generations_left):
            # Run for ONE generation
            p.run(eval_genomes, 1)
            
        winner = stats.best_genome()
        print('\nBest genome (from stats):\n{!s}'.format(winner))

    else:
        print(f"--- Already trained for {p.generation} generations. ---")
        winner = stats.best_genome() # Get best from stats
        print('\nBest genome (from loaded checkpoint):\n{!s}'.format(winner))
    # --- End of new loop ---


    # --- Save the FINAL winner ---
    print("\nSaving final best network...")
    visualize.draw_net(config, winner, True, 
                            node_names=node_names, 
                            filename="winner-net.gv")
    
    # (Optional) Render the final winner as well
    try:
        import graphviz
        s = graphviz.Source.from_file("winner-net.gv")
        s.format = 'png'
        s.render(filename="winner-net", view=False, cleanup=True)
        if os.path.exists("winner-net.png"):
             print(f"Successfully rendered final network to winner-net.png")
    except Exception as e:
        print(f"Failed to render final winner: {e}")


    # Save the winner genome 
    with open('winner-genome.pkl', 'wb') as f:
        pickle.dump(winner, f)

def run_winner(config, genome_path="winner-genome.pkl"):
    """
    Load a saved genome and run it in the simulation.
    """

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config)

    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    car = Car(start_pos=START_POS, start_angle=0.0) 

    run = True
    while run and car.alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        screen.blit(track_surf, (0, 0))
        # Draw the finish line and checkpoint
        pygame.draw.rect(screen, (0, 255, 0), FINISH_LINE_RECT, 2) 
        pygame.draw.rect(screen, (255, 0, 0), CHECKPOINT_1_RECT, 2) 
        
        # Get inputs for the car
        inputs = car.cast_rays()
        inputs.append(car.speed / car.max_speed)
        
        # Activate the network
        outputs = net.activate(inputs)
        steer, accel = outputs[0], outputs[1]
        
        # Update and draw the car
        car.update(steer, accel)
        car.draw(screen)

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__": 
    local_dir = os.path.dirname(__file__) 
    config_path = os.path.join(local_dir, "config-feedforward.txt")  
    
    parser = argparse.ArgumentParser(description = "Run NEAT car training or simulation.")
    parser.add_argument(
        "--train",
        action = "store_true",  
        help = "If set, run the training process. Otherwise, run the best winner."
    )
    args = parser.parse_args()
    

    if args.train: 
        print("--- Starting in TRAINING mode ---")
        run(config_path)
    else: 
        print("--- Starting in TESTING mode ---")
        run_winner(config_path, "winner-genome.pkl")
