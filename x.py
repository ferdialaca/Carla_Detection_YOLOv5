import cv2
import pygame
from pygame.locals import *
import carla
import numpy as np
import torch
import queue
from utils.augmentations import letterbox
from utils.general import non_max_suppression, plot_one_box, scale_coords
from utils.plots import colors
import random


def init_carla():
    client = carla.Client('localhost', 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    return client, world

def load_model(weights):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    model = model.eval()
    return model

def init_pygame(width, height):
    pygame.init()
    pygame.font.init()
    screen = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA Object Detection")
    clock = pygame.time.Clock()
    return screen, clock

def detect_objects(image, model, img_size=(640, 640), conf_threshold=0.3, iou_threshold=0.45):
    img0 = image.copy()

    img = letterbox(img0, new_shape=img_size)[0]
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(torch.device('cuda'))
    img = img.float() / 255.0
    img = img.unsqueeze(0)

    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_threshold, iou_threshold, classes=None, agnostic=False)

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{model.names[c]} {conf:.2f}'
                color = colors(c)
                plot_one_box(xyxy, img0, label=label, color=color, line_thickness=2)

    return img0


def main_loop(screen, clock, client, world, model):
    actor_list = []
    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('model3')[0]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.apply_control(carla.VehicleControl(throttle=5.0, steer=0.0))
    actor_list.append(vehicle)
    camera_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
    camera_sensor = world.spawn_actor(camera_blueprint, camera_transform, attach_to=vehicle)
    image_queue = queue.Queue()

    def camera_callback(image):
        image_queue.put(image)

    camera_sensor.listen(camera_callback)

    try:
        while True:
            clock.tick(30)
            world_snapshot = world.wait_for_tick()

            try:
                image = image_queue.get(block=False)
                image_data = np.array(image.raw_data)
                image_data = image_data.reshape((image.height, image.width, 4))
                image_data = image_data[:, :, :3]

                image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                image_data = detect_objects(image_data, model)

                image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
                image_data = np.rot90(image_data)

                pygame_img = pygame.surfarray.make_surface(image_data)
                screen.blit(pygame_img, (0, 0))
                pygame.display()

            except queue.Empty:
                continue

            for event in pygame.event.get():
                if event.type == QUIT:
                    return

    finally:
        camera_sensor.destroy()
        pygame.quit()

if __name__ == '__main__':
    weights_path = 'best.pt'  # Path to your best.pt file
    client, world = init_carla()
    model = load_model(weights_path)
    width, height = 800, 600  # Adjust the window size as needed
    screen, clock = init_pygame(width, height)
    main_loop(screen, clock, client, world, model)
