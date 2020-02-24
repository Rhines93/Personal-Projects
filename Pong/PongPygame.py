# Author: Robert Hines
# February 23, 2020

import pygame

pygame.init()
pygame.font.init()

# Create window
window = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Pong by Tom")
white = pygame.Color(255, 255, 255)
screenWidth = 800
screenHeight = 600
myFont = pygame.font.SysFont('Courier', 24, bold=True)
score_a = 0
score_b = 0

# Paddles
paddle_a = pygame.Rect((25, 250, 20, 100))
paddle_b = pygame.Rect((755, 250, 20, 100))

# Ball
ball = pygame.Rect((390, 290, 20, 20))
dx = 1
dy = 1

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Draw the board and game pieces

    window.fill((0,0,0))
    pygame.draw.rect(window, white, paddle_a)
    pygame.draw.rect(window, white, paddle_b)
    pygame.draw.rect(window, white, ball)

    # Control paddles vertical movement

    keys = pygame.key.get_pressed()

    if keys[pygame.K_w] and paddle_a.y >= 0:
        paddle_a.y -= 2

    if keys[pygame.K_s] and paddle_a.y <= screenHeight-paddle_a.height:
        paddle_a.y += 2

    if keys[pygame.K_UP] and paddle_b.y >= 0:
        paddle_b.y -= 2

    if keys[pygame.K_DOWN] and paddle_b.y <= screenHeight-paddle_b.height:
        paddle_b.y += 2

    # Generate ball movement

    ball.x += dx
    ball.y += dy

    # Ball horizontal boundaries

    if ball.x >= 780:
        ball.x = 390
        ball.y = 290
        score_a += 1
        dx *= -1
    
    if ball.x <= 0:
        ball.x = 390
        ball.y = 290
        score_b += 1
        dx *= -1

    # Ball vertical boundaries

    if ball.y >= 580:
        ball.y = 580
        dy *= -1
    
    if ball.y <= 0:
        ball.y = 0
        dy *= -1

    # Paddle and ball collision detection and reaction

    if ball.colliderect(paddle_a) or ball.collidepoint(paddle_a.x, paddle_a.y):
        ball.x += 10
        dx *= -1
    if ball.colliderect(paddle_b) or ball.collidepoint(paddle_b.x, paddle_b.y):
        ball.x -= 10
        dx *= -1

    text = myFont.render("Player A: {}  Player B: {}".format(score_a,score_b), False, white)
    window.blit(text,(225,25))

    # Print Victory if score is greater than 5

    if score_a >= 5:
        text = myFont.render("Player A: VICTORY!", False, white)
        text2 = myFont.render("Press SPACE to play again", False, white)
        
        window.fill((0,0,0))
        window.blit(text,(250,300))
        window.blit(text2,(200,325))

        # Press SPACE to reset scores
        if keys[pygame.K_SPACE]:
            score_a, score_b = 0, 0
            ball.x = 390
        
    elif score_b >= 5:
        text = myFont.render("Player B: VICTORY!", False, white)
        text2 = myFont.render("Press SPACE to play again", False, white)
       
        window.fill((0,0,0))
        window.blit(text,(250,300))
        window.blit(text2,(200,325))

        # Press SPACE to reset scores
        if keys[pygame.K_SPACE]:
            score_a, score_b = 0, 0
            ball.x = 390

    pygame.time.wait(2)
    pygame.display.update()

pygame.quit()
