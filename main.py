import pygame
import sys
import random, time, numpy

# Initialize pygame
pygame.init()

# Game constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
BG_COLOR = (0, 0, 0)

PREY_COLOR = (50, 0, 255)
PREY_SPEED = 5

PREDATOR_COLOR = (100, 0, 20)
PREDATOR_SPEED = 4

# Create the game window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Simulation Game")

# Create prey entity
prey_size = 10
prey_stamina = 400
prey_penalty = 0
prey_reward = 0
prey_x = random.randint(15, SCREEN_WIDTH - 15)
prey_y = random.randint(15, SCREEN_HEIGHT - 15)
preyInputNodes = 4
preyOutputNodes = 2
preyHiddenNodes = 100
preyWeightsMatrix1 = numpy.loadtxt("preyWeightsMatrix1.txt")
preyWeightsMatrix1 = numpy.array(preyWeightsMatrix1).reshape((preyInputNodes, preyHiddenNodes))
preyWeightsMatrix2 = numpy.loadtxt("preyWeightsMatrix2.txt")
preyWeightsMatrix2 = numpy.array(preyWeightsMatrix2).reshape((preyHiddenNodes, preyOutputNodes))

# preyWeightsMatrix1 = numpy.random.randn(preyInputNodes, preyHiddenNodes)
# preyWeightsMatrix2 = numpy.random.randn(preyHiddenNodes, preyOutputNodes)

# create predator entity
predator_size = 15
predator_stamina = 450
predator_penalty = 0
predator_reward = 0
predator_x = random.randint(15, SCREEN_WIDTH - 15)
predator_y = random.randint(15, SCREEN_HEIGHT - 15)
predatorInputNodes = 4
predatorOutputNodes = 2
predatorHiddenNodes = 100
predatorWeightsMatrix1 = numpy.loadtxt("predatorWeightsMatrix1.txt")
predatorWeightsMatrix1 = numpy.array(predatorWeightsMatrix1).reshape((predatorInputNodes, predatorHiddenNodes))
predatorWeightsMatrix2 = numpy.loadtxt("predatorWeightsMatrix2.txt")
predatorWeightsMatrix2 = numpy.array(predatorWeightsMatrix2).reshape((predatorHiddenNodes, predatorOutputNodes))

# predatorWeightsMatrix1 = numpy.random.randn(preyInputNodes, preyHiddenNodes)
# predatorWeightsMatrix2 = numpy.random.randn(preyHiddenNodes, preyOutputNodes)

# Initialize time variables for player inactivity and entity spawning
last_movement_time = time.time()
inactive_time_threshold = 5  # 5 seconds
spawned_entities = []

# Game loop
running = True
clock = pygame.time.Clock()

# Dangerous entity properties
dangerous_radius = 10
dangerous_x = random.randint(dangerous_radius, SCREEN_WIDTH - dangerous_radius)
dangerous_y = random.randint(dangerous_radius, SCREEN_HEIGHT - dangerous_radius)
DANGEROUS_COLOR = (255, 0, 0)

tick = 0

learningRate = 0.001

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    tick = tick + 1

    #time penalty for predator
    if tick > 60:
        predator_penalty = predator_penalty + (1 / (tick / 60))

    #time reward for prey
    if tick > 60:
        prey_reward = prey_reward + (1 / (tick / 60))

    prey_to_predator_distance = ((prey_x - predator_x) ** 2 + (prey_y - predator_y) ** 2) ** 0.5

    # Handle prey network
    preyInputData = numpy.array([prey_x, prey_y, prey_stamina, prey_to_predator_distance])
    preyHiddenValues = preyInputData.dot(preyWeightsMatrix1)

    preyHiddenRelu = numpy.maximum(preyHiddenValues, 0)
    preyOutputDataPredictions = preyHiddenRelu.dot(preyWeightsMatrix2)

    # Handle predator network
    predatorInputData = numpy.array([predator_x, predator_y, predator_stamina, prey_to_predator_distance])
    predatorHiddenValues = predatorInputData.dot(predatorWeightsMatrix1)

    predatorHiddenRelu = numpy.maximum(predatorHiddenValues, 0)
    predatorOutputDataPredictions = predatorHiddenRelu.dot(predatorWeightsMatrix2)

    if prey_stamina > 0:
        if preyOutputDataPredictions[0] > 0.6:
            prey_x += PREY_SPEED
        if preyOutputDataPredictions[0] < -0.6:
            prey_x -= PREY_SPEED
        if preyOutputDataPredictions[1] > 0.6:
            prey_y -= PREY_SPEED
        if preyOutputDataPredictions[1] < -0.6:
            prey_y += PREY_SPEED
    else:
        prey_penalty = prey_penalty + (10 / (tick / 60))
    
    # handle predator movement
    if predator_stamina > 0:
        if predatorOutputDataPredictions[0] > 0.6:
            predator_x += PREDATOR_SPEED
        if predatorOutputDataPredictions[0] < -0.6:
            predator_x -= PREDATOR_SPEED
        if predatorOutputDataPredictions[1] > 0.6:
            predator_y -= PREDATOR_SPEED
        if predatorOutputDataPredictions[1] < -0.6:
            predator_y += PREDATOR_SPEED
    else:
        predator_penalty = predator_penalty + (10 / (tick / 60))
    
    # Check for player movement
    prey_has_moved = False
    predator_has_moved = False
    keys = pygame.key.get_pressed()
    #prey
    if (preyOutputDataPredictions[0] > 0.6) or (preyOutputDataPredictions[0] <-0.6) or (preyOutputDataPredictions[1] > 0.6) or (preyOutputDataPredictions[1] < -0.6):
        if prey_stamina > 0:
            prey_stamina = prey_stamina - 1
        prey_has_moved = True
    #predator
    if (predatorOutputDataPredictions[0] > 0.6) or (predatorOutputDataPredictions[0] <-0.6) or (predatorOutputDataPredictions[1] > 0.6) or (predatorOutputDataPredictions[1] < -0.6):
        if predator_stamina > 0:
            predator_stamina = predator_stamina - 1
        predator_has_moved = True

    if prey_has_moved == False:
        if prey_stamina < 400:
            prey_stamina = prey_stamina + 1
    if predator_has_moved == False:
        if predator_stamina < 450:
            predator_stamina = predator_stamina + 1

    #keeps prey inside game window
    prey_x = max(prey_size, min(prey_x, SCREEN_WIDTH - prey_size))
    prey_y = max(prey_size, min(prey_y, SCREEN_HEIGHT - prey_size))

    #keeps predator inside game window
    predator_x = max(predator_size, min(predator_x, SCREEN_WIDTH - predator_size))
    predator_y = max(predator_size, min(predator_y, SCREEN_HEIGHT - predator_size))

    # Check for collisions
    prey_to_dangerous_distance = ((prey_x - dangerous_x) ** 2 + (prey_y - dangerous_y) ** 2) ** 0.5
    if prey_to_dangerous_distance < prey_size + dangerous_radius:
        prey_penalty = prey_penalty + (1000 / (tick / 60))
        running = False  # End the game if collision occurs
    

    #check collision between predator and prey
    if (0.5 - (prey_to_predator_distance / SCREEN_WIDTH)) > 0:
        predator_reward += (0.5 - (prey_to_predator_distance / SCREEN_WIDTH))
        prey_penalty += (0.5 - (prey_to_predator_distance / SCREEN_WIDTH))

    if prey_to_predator_distance < prey_size + predator_size or tick > (10*60):
        predator_reward = predator_reward + (1000 / (tick / 60))
        prey_penalty = prey_penalty + (1000 / (tick / 60))
        
        #reset game states
        tick = 0
        prey_reward = 0
        prey_penalty = 0
        prey_stamina = 100

        predator_penalty = 0
        predator_reward = 0
        predator_stamina = 100

        prey_x = random.randint(15, SCREEN_WIDTH - 15)
        prey_y = random.randint(15, SCREEN_HEIGHT - 15)

        predator_x = random.randint(15, SCREEN_WIDTH - 15)
        predator_y = random.randint(15, SCREEN_HEIGHT - 15)

    #back propagation for prey
    if prey_penalty != 0 or prey_reward !=0:
        preyGradientPrediction = 2 * (preyOutputDataPredictions - (preyOutputDataPredictions / (learningRate * (prey_reward - prey_penalty))))
        preyGradientWeights2 = preyHiddenRelu.reshape((-1, 1)).dot(preyGradientPrediction.reshape((1, -1)))
        preyGradientHiddenRelu = preyGradientPrediction.dot(preyGradientWeights2.T)
        preyGradientHiddenValues = preyGradientHiddenRelu.copy()
        preyGradientHiddenValues[preyHiddenValues < 0] = 0
        preyGradientWeights1 = preyInputData.T.reshape((-1,1)).dot(preyGradientHiddenValues.reshape((1, -1)))
        preyWeightsMatrix1 = preyWeightsMatrix1 - preyGradientWeights1 * 1e-4
        preyWeightsMatrix2 = preyWeightsMatrix2 - preyGradientWeights2 * 1e-4
        numpy.savetxt("preyWeightsMatrix1.txt", preyWeightsMatrix1, fmt="%e")
        numpy.savetxt("preyWeightsMatrix2.txt", preyWeightsMatrix2, fmt="%e")

    #backpropagation for predator
    if predator_reward !=0 or predator_penalty != 0:
        predatorGradientPrediction = 2 * (predatorOutputDataPredictions - (predatorOutputDataPredictions / (learningRate * (predator_penalty - predator_reward))))
        predatorGradientWeights2 = predatorHiddenRelu.reshape((-1, 1)).dot(predatorGradientPrediction.reshape((1, -1)))
        predatorGradientHiddenRelu = predatorGradientPrediction.dot(predatorGradientWeights2.T)
        predatorGradientHiddenValues = predatorGradientHiddenRelu.copy()
        predatorGradientHiddenValues[predatorHiddenValues < 0] = 0
        predatorGradientWeights1 = predatorInputData.T.reshape((-1,1)).dot(predatorGradientHiddenValues.reshape((1, -1)))
        predatorWeightsMatrix1 = predatorWeightsMatrix1 - predatorGradientWeights1 * 1e-4
        predatorWeightsMatrix2 = predatorWeightsMatrix2 - predatorGradientWeights2 * 1e-4
        numpy.savetxt("predatorWeightsMatrix1.txt", predatorWeightsMatrix1, fmt="%e")
        numpy.savetxt("predatorWeightsMatrix2.txt", predatorWeightsMatrix2, fmt="%e")


    prey_reward = 0
    prey_penalty = 0

    predator_penalty = 0
    predator_reward = 0
    # Check for inactivity and spawn new entity
    # if not has_moved and time.time() - last_movement_time >= inactive_time_threshold:
    #     new_entity = {
    #         'x': random.randint(max(prey_x - 50, 0), min(prey_x + 50, SCREEN_WIDTH)),
    #         'y': random.randint(max(prey_y - 50, 0), min(prey_y + 50, SCREEN_HEIGHT)),
    #         'radius': dangerous_radius,
    #         'color': (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)),
    #         'spawned': True
    #     }
    #     spawned_entities.append(new_entity)

    # Update screen
    screen.fill(BG_COLOR)

    # Draw and update existing entities
    for entity in spawned_entities:
        pygame.draw.circle(screen, entity['color'], (entity['x'], entity['y']), entity['radius'])


    # Draw dangerous entity
    #pygame.draw.circle(screen, DANGEROUS_COLOR, (dangerous_x, dangerous_y), dangerous_radius)

    # Draw prey
    pygame.draw.circle(screen, PREY_COLOR, (prey_x, prey_y), prey_size)

    # draw predator
    pygame.draw.circle(screen, PREDATOR_COLOR, (predator_x, predator_y), predator_size)

    #hud
    #predator
    black = (0, 0, 0)
    white = (255, 255, 255)
    font = pygame.font.Font(None, 36)
    predator_stamina_text = f'Predator Stamina: {predator_stamina}'
    predator_text_surface = font.render(predator_stamina_text, True, white)
    predator_text_rect = predator_text_surface.get_rect()
    predator_text_rect.midleft = (10, SCREEN_HEIGHT - 80)
    #clock
    clock_text = f'Clock: {tick // 60}'
    clock_text_surface = font.render(clock_text, True, white)
    clock_text_rect = clock_text_surface.get_rect()
    clock_text_rect.midleft = (10, SCREEN_HEIGHT - 110)
    #prey
    prey_stamina_text = f'Prey Stamina: {prey_stamina}'
    prey_text_surface = font.render(prey_stamina_text, True, white)
    prey_text_rect = prey_text_surface.get_rect()
    prey_text_rect.midleft = (10, SCREEN_HEIGHT - 50)
    #prey reward/penalty
    prey_reward_text = f'Prey Reward/Penaly: {int(prey_reward)} / {int(prey_penalty)}'
    prey_reward_text_surface = font.render(prey_reward_text, True, white)
    prey_reward_text_rect = prey_reward_text_surface.get_rect()
    prey_reward_text_rect.midleft = (10, SCREEN_HEIGHT - 140)
    #predator reward/penalty
    predator_reward_text = f'Predator Reward/Penaly: {int(predator_reward)} / {int(predator_penalty)}'
    predator_reward_text_surface = font.render(predator_reward_text, True, white)
    predator_reward_text_rect = predator_reward_text_surface.get_rect()
    predator_reward_text_rect.midleft = (10, SCREEN_HEIGHT - 170)
    # Blit the text surface onto the screen
    screen.blit(prey_text_surface, prey_text_rect)
    screen.blit(predator_text_surface, predator_text_rect)
    screen.blit(clock_text_surface, clock_text_rect)
    screen.blit(prey_reward_text_surface, prey_reward_text_rect)
    screen.blit(predator_reward_text_surface, predator_reward_text_rect)




    pygame.display.flip()

    clock.tick(60)

# Clean up
pygame.quit()
sys.exit()
