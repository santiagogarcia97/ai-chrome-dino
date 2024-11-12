# Este es el juego obtenido de https://github.com/MaxRohowsky/chrome-dinosaur

import pygame
import os
import random
pygame.init()

# Global Constants
SCREEN_HEIGHT = 600
SCREEN_WIDTH = 1100
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

RUNNING = [pygame.image.load(os.path.join("Assets/Dino", "DinoRun1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoRun2.png"))]
JUMPING = pygame.image.load(os.path.join("Assets/Dino", "DinoJump.png"))
DUCKING = [pygame.image.load(os.path.join("Assets/Dino", "DinoDuck1.png")),
           pygame.image.load(os.path.join("Assets/Dino", "DinoDuck2.png"))]

SMALL_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus1.png")),
                pygame.image.load(os.path.join(
                    "Assets/Cactus", "SmallCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "SmallCactus3.png"))]
LARGE_CACTUS = [pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus1.png")),
                pygame.image.load(os.path.join(
                    "Assets/Cactus", "LargeCactus2.png")),
                pygame.image.load(os.path.join("Assets/Cactus", "LargeCactus3.png"))]

BIRD = [pygame.image.load(os.path.join("Assets/Bird", "Bird1.png")),
        pygame.image.load(os.path.join("Assets/Bird", "Bird2.png"))]

CLOUD = pygame.image.load(os.path.join("Assets/Other", "Cloud.png"))

BG = pygame.image.load(os.path.join("Assets/Other", "Track.png"))


class Dinosaur:
    Y_POS = 310
    Y_POS_DUCK = 340
    JUMP_VEL = 8.5

    def __init__(self, id: int):
        self.id = id
        self.is_alive = True
        self.points = 0
        self.x_pos = random.randint(40, 140)

        self.duck_img = DUCKING
        self.run_img = RUNNING
        self.jump_img = JUMPING

        self.dino_duck = False
        self.dino_run = True
        self.dino_jump = False

        self.step_index = 0
        self.jump_vel = self.JUMP_VEL
        self.image = self.run_img[0]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.x_pos
        self.dino_rect.y = self.Y_POS

    def update(self, userInput):
        if self.dino_duck:
            self.duck()
        if self.dino_run:
            self.run()
        if self.dino_jump:
            self.jump()

        if self.step_index >= 10:
            self.step_index = 0

        if userInput == "JUMP" and not self.dino_jump:
            self.dino_duck = False
            self.dino_run = False
            self.dino_jump = True
        elif userInput == "CROUCH" and not self.dino_jump:
            self.dino_duck = True
            self.dino_run = False
            self.dino_jump = False
        elif not (self.dino_jump or userInput == "CROUCH"):
            self.dino_duck = False
            self.dino_run = True
            self.dino_jump = False

    def duck(self):
        self.image = self.duck_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.x_pos
        self.dino_rect.y = self.Y_POS_DUCK
        self.step_index += 1

    def run(self):
        self.image = self.run_img[self.step_index // 5]
        self.dino_rect = self.image.get_rect()
        self.dino_rect.x = self.x_pos
        self.dino_rect.y = self.Y_POS
        self.step_index += 1

    def jump(self):
        self.image = self.jump_img
        if self.dino_jump:
            self.dino_rect.y -= self.jump_vel * 4
            self.jump_vel -= 0.8
        if self.jump_vel < - self.JUMP_VEL:
            self.dino_jump = False
            self.jump_vel = self.JUMP_VEL

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.dino_rect.x, self.dino_rect.y))

    def get_state(self, obstacles, game_speed):
        dino_y = self.dino_rect.y

        if len(obstacles) > 0:
            obstacle_x = obstacles[0].rect.x
            obstacle_y = obstacles[0].rect.y
            obstacle_width = obstacles[0].rect.width
            obstacle_height = obstacles[0].rect.height
            obstacle_id = obstacles[0].id
        else:
            # Default values if no obstacles are present
            obstacle_x = 0
            obstacle_y = 0
            obstacle_width = 0
            obstacle_height = 0
            obstacle_id = 0

        distancia_al_obstaculo = SCREEN_WIDTH - self.dino_rect.x if obstacle_x <= 0 else obstacle_x - self.dino_rect.x

        params = [distancia_al_obstaculo, obstacle_id, SCREEN_HEIGHT - obstacle_y, obstacle_width, obstacle_height, dino_y, game_speed*10]

        #params = [distancia_al_obstaculo, obstacle_id, SCREEN_HEIGHT - obstacle_y, game_speed]

        #if self.id == 0:
            #print(params)

        return params


class Cloud:
    def __init__(self):
        self.x = SCREEN_WIDTH + random.randint(800, 1000)
        self.y = random.randint(50, 100)
        self.image = CLOUD
        self.width = self.image.get_width()

    def update(self, game_speed):
        self.x -= game_speed
        if self.x < -self.width:
            self.x = SCREEN_WIDTH + random.randint(2500, 3000)
            self.y = random.randint(50, 100)

    def draw(self, SCREEN):
        SCREEN.blit(self.image, (self.x, self.y))


class Obstacle:
    def __init__(self, image, type, id):
        self.id = id
        self.image = image
        self.type = type
        self.rect = self.image[self.type].get_rect()
        self.rect.x = SCREEN_WIDTH

    def update(self, game_speed, obstacles):
        self.rect.x -= game_speed
        if self.rect.x < -self.rect.width:
            obstacles.pop()

    def draw(self, SCREEN):
        SCREEN.blit(self.image[self.type], self.rect)


class SmallCactus(Obstacle):
    def __init__(self, image):
        self.id = 1
        self.type = random.randint(0, 2)
        super().__init__(image, self.type, self.id)
        self.rect.y = 325


class LargeCactus(Obstacle):
    def __init__(self, image):
        self.id = 2
        self.type = random.randint(0, 2)
        super().__init__(image, self.type, self.id)
        self.rect.y = 300


class Bird(Obstacle):
    def __init__(self, image):
        self.id = 3
        self.type = 0
        super().__init__(image, self.type, self.id)
        self.rect.y = random.choice([230, 250, 270])
        self.index = 0

    def draw(self, SCREEN):
        if self.index >= 9:
            self.index = 0
        SCREEN.blit(self.image[self.index//5], self.rect)
        self.index += 1


class GameInstance:
    def __init__(self, dinos_count=1):
        self.dinos_count = dinos_count
        self.dinos = []
        self.run = True
        self.clock = pygame.time.Clock()
        self.cloud = Cloud()
        self.game_speed = 20
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        self.points = 0
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.obstacles = []

        self.frame = 0

    def score(self):
        self.points += 1

        text = self.font.render("Points: " + str(self.points), True, (0, 0, 0))
        textRect = text.get_rect()
        textRect.center = (1000, 40)
        SCREEN.blit(text, textRect)

        dinos_alive = self.font.render(
            "Dinos Alive: " + str(len([dino for dino in self.dinos if dino.is_alive])), True, (0, 0, 0))
        dinosRect = dinos_alive.get_rect()
        dinosRect.bottomleft = (50, 500)
        SCREEN.blit(dinos_alive, dinosRect)

    def background(self):
        image_width = BG.get_width()
        SCREEN.blit(BG, (self.x_pos_bg, self.y_pos_bg))
        SCREEN.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))
        if self.x_pos_bg <= -image_width:
            SCREEN.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))
            self.x_pos_bg = 0
        self.x_pos_bg -= self.game_speed

    def restart(self):
        self.dinos = [Dinosaur(id=i) for i in range(self.dinos_count)]
        self.cloud = Cloud()
        self.game_speed = 20
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        self.points = 0
        self.obstacles = []

    def play_step(self, actions):
        if self.run:
            self.frame += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False
            SCREEN.fill((255, 255, 255))

            for i in range(self.dinos_count):
                if self.dinos[i].is_alive:
                    self.dinos[i].update(actions[i])
                    self.dinos[i].draw(SCREEN)

            if len(self.obstacles) == 0:
                random_obstacle = random.randint(0, 2)
                if self.frame < 500:
                    random_obstacle = random.randint(0, 3)

                if random_obstacle == 0:
                    self.obstacles.append(SmallCactus(SMALL_CACTUS))
                elif random_obstacle == 1:
                    self.obstacles.append(LargeCactus(LARGE_CACTUS))
                else:
                    self.obstacles.append(Bird(BIRD))

            for obstacle in self.obstacles:
                obstacle.draw(SCREEN)
                obstacle.update(self.game_speed, self.obstacles)
                for dinosaur in self.dinos:
                    if dinosaur.is_alive and dinosaur.dino_rect.colliderect(obstacle.rect):
                        dinosaur.is_alive = False
                        dinosaur.points = self.points

            self.background()

            self.cloud.draw(SCREEN)
            self.cloud.update(self.game_speed)

            self.score()
            self.game_speed += 0.002

            #self.clock.tick(30)
            pygame.display.update()

            return self.dinos
