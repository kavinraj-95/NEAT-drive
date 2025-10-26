import pygame

pygame.init()
img = pygame.image.load("data/track.png")
screen = pygame.display.set_mode(img.get_size())
screen.blit(img, (0, 0))
pygame.display.flip()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            print(f"Clicked position: ({x}, {y})")
pygame.quit()

