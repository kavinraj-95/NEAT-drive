import os, math, random, pygame, neat, pickle 
from pygame import Vector2

TRACK = 'data/track.png'
WINDOW_SIZE = (600, 400)
MAX_STEPS_PER_CAR = 2000
RAY_COUNT = 6                                           # calculates distance from 6 directions
RAY_LENGTH = 120   
CHECK_STEPS = 120       # Check if car is stuck every 120 steps (2 seconds)
MIN_DISPLACEMENT = 10.0 # Car must move at least 10 pixels in that time
TILE_SIZE = 20          # Size of the "exploration" grid tiles

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
        self.size = (10, 18)

        self.steps_since_last_check = 0
        self.last_check_pos = Vector2(start_pos)
        
        self.visited_tiles = set()
        start_tile = (int(self.pos.x / TILE_SIZE), int(self.pos.y / TILE_SIZE))
        self.visited_tiles.add(start_tile)


    def update(self, steer, accel):
        if not self.alive:
            return

        # Steer, angle in range [-1, 1] -> tanh
        self.angle += steer * 4.0
        self.speed += accel * 0.2

        # clamp speed
        self.speed = max(0.5, min(self.speed, self.max_speed))
        # update position
        dir_vec = Vector2(math.sin(math.radians(self.angle)), -math.cos(math.radians(self.angle)))
        self.pos += dir_vec * self.speed

        # self.distance_travelled += abs(self.speed) # <-- CHANGE THIS
        self.distance_travelled += self.speed        # <-- TO THIS (Punishes reversing)
        
        self.steps += 1 
        self.steps_since_last_check += 1

        current_tile = (int(self.pos.x / TILE_SIZE), int(self.pos.y / TILE_SIZE))
        self.visited_tiles.add(current_tile)

        if self.steps > MAX_STEPS_PER_CAR:
            self.alive = False
            
        if self.steps_since_last_check > CHECK_STEPS:
            displacement = (self.pos - self.last_check_pos).length()
            if self.alive and displacement < MIN_DISPLACEMENT:
                self.alive = False # Kill car if it's stuck or circling
            
            self.steps_since_last_check = 0
            self.last_check_pos = Vector2(self.pos.x, self.pos.y)        # collision checks -> Use bounding box for the car
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
            
            ray_angle = base_angle - spread / 2 + (spread / (RAY_COUNT - 1)) * i # Removing crookedness in the eye of the car
            
            # Fix for divide-by-zero if RAY_COUNT = 1
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

            readings.append(dist / RAY_LENGTH)      # Normalise it so it lies between [0, 1] and append readings

        return readings


    def draw(self, surf):
        rect = pygame.Rect(0, 0, self.size[0], self.size[1])
        rect.center = (self.pos.x, self.pos.y)
        car_surf = pygame.Surface(self.size, pygame.SRCALPHA)
        pygame.draw.rect(car_surf, (255,0,0), (0,0,self.size[0],self.size[1]))
        rotated = pygame.transform.rotate(car_surf, self.angle)
        rrect = rotated.get_rect(center = rect.center)
        surf.blit(rotated, rrect.topleft)

        for i, d in enumerate(self.cast_rays()):
            ray_angle = self.angle - 60 + (120/(RAY_COUNT-1))*i
            rad = math.radians(ray_angle)
            x2 = self.pos.x + math.sin(rad) * d * RAY_LENGTH
            y2 = self.pos.y - math.cos(rad) * d * RAY_LENGTH
            pygame.draw.line(surf, (0,255,0), (self.pos.x, self.pos.y), (x2, y2), 1)



# NEAT evaluation
def eval_genomes(genomes, config):
    nets = []
    cars = []
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Car(start_pos=(450, 373), start_angle=0.0))
        genome.fitness = 0

    run = True
    while run and any(car.alive for car in cars):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        screen.blit(track_surf, (0, 0))
        
        for i, car in enumerate(cars):
            if not car.alive:
                continue
            inputs = car.cast_rays()
            inputs.append(car.speed / car.max_speed)
            outputs = nets[i].activate(inputs)
            steer, accel = outputs[0], outputs[1]
            car.update(steer, accel)
            car.draw(screen)
            genomes[i][1].fitness = len(car.visited_tiles) * 100 + car.distance_travelled

        pygame.display.flip()
        clock.tick(60)

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(eval_genomes, 50)
    print('\nBest genome:\n{!s}'.format(winner)) 

    with open('winner-genome.pkl', 'wb') as f:
        pickle.dump(winner, f)

def run_winner(config, genome_path="winner-genome.pkl"):
    """
    Load a saved genome and run it in the simulation.
    """
    # Load the NEAT config
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config)

    # Load the saved genome
    with open(genome_path, 'rb') as f:
        genome = pickle.load(f)
    
    # Create the neural network from the genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    
    # Create one car
    car = Car(start_pos=(450, 373), start_angle=0.0)

    run = True
    while run and car.alive:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        screen.blit(track_surf, (0, 0))
        
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
    
    TRAIN = False

    if TRAIN:
        run(config_path)
    else:
        # This runs the saved winner
        run_winner(config_path, "winner-genome.pkl")
