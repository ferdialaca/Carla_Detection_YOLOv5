import carla
import numpy as np
import cv2
import pygame
from pygame.locals import *
import random
import torch
from utils.augmentations import letterbox
from utils.general import non_max_suppression, plot_one_box, scale_coords
from utils.plots import colors
import queue

# Carla sunucusuna bağlan
client = carla.Client('localhost', 2000)
client.set_timeout(6.0)

# Oyuncu aracı oluştur
world = client.get_world()
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_point = random.choice(world.get_map().get_spawn_points())
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Kamera oluştur
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model = model.eval()
image_queue = queue.Queue()

# Pygame penceresi oluşturma
pygame.init()
screen_width = 640
screen_height = 480
display = pygame.display.set_mode((screen_width, screen_height))
clock = pygame.time.Clock()

# Kameradan görüntü alma işlevi
def process_image(image):
    img_array = np.array(image.raw_data)
    img_rgb = img_array.reshape((image.height, image.width, 4))[:, :, :3]
    image_queue.put(img_rgb)
    if not image_queue.empty():
        image = image_queue.get()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img0 = image.copy()

        img = letterbox(img0, new_shape=(480, 480))[0]
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(torch.device('cuda'))
        img = img.float() / 255.0
        img = img.unsqueeze(0)

        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.65, 0.45, classes=None, agnostic=False)
        crosswalk_index = None
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    label = f'{model.names[c]} {conf:.2f}'
                    color = colors(c)
                    plot_one_box(xyxy, img0, label=label, color=color, line_thickness=2)
                    if model.names[c] == 'crosswalk':
                # Eğer zaten crosswalk tespiti yapılmışsa döngüden çık
                        if crosswalk_index is not None:
                            break
                        crosswalk_index = c
                if crosswalk_index is not None:
                    break
        img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
        img0 = cv2.rotate(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img0 = cv2.flip(img0, 0)
        # Resize the image to fit the pygame window
        #img0 = cv2.resize(img0, (screen_width, screen_height))

        # Display the image in pygame window
        img_surface = pygame.surfarray.make_surface(img0)
        display.blit(img_surface, (0, 0))
        pygame.display.update()

# Kamera sensörüne görüntü işleme işlevini bağlama
camera.listen(lambda image: process_image(image))

# Hareket kontrol değişkenleri
throttle = 0.0
steer = 0.0

# Simülasyon döngüsü
while True:
    clock.tick(30)
    
    # Tüm olayları işleme
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            exit()

        if event.type == KEYDOWN:
            if event.key == K_w:
                throttle = 0.5  # İleri gitmek için hızlanma
            elif event.key == K_s:
                throttle = -0.5  # Geri gitmek için hızlanma
            elif event.key == K_a:
                steer = -0.5  # Sola dönme
            elif event.key == K_d:
                steer = 0.5  # Sağa dönme
        elif event.type == KEYUP:
            if event.key == K_w or event.key == K_s:
                throttle = 0.0  # Hızlanmayı durdur
            if event.key == K_a or event.key == K_d:
                steer = 0.0  # Dönme hareketini durdur

    # Arabayı hareket ettirme
    control = carla.VehicleControl(throttle=throttle, steer=steer)
    vehicle.apply_control(control)

    # Carla simülasyon dünyasını güncelleme
    world.tick()

