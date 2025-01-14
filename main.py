from image_generator import ImageGenerator
from detect_objects import ObjectDetector
from bvp import BVPSolver
from ball_animation import BallThrowingAnimation
from PIL import Image
import random

video_scale_factor = 0.5 # we need this now to calculate ball radius and find overlaps when shooting

# ================================= Step 1: Image =================================
# Option 1: Test on your image
image_path = r'Final-Project\Project-1_Hitting_Target\assets\test1.png'
image = Image.open(image_path)
width, height = image.size
ball_radius = int((min(width, height) // 30) * video_scale_factor) # don't change this, since animation uses same size as well

# Option 2: Generate an image
# height, width = 1008, 1008
# generator = ImageGenerator((height, width), object_path = r'Final-Project\Project-1_Hitting_Target\assets\target.png')
# image_path = generator.generate_image(6)
# print(f'Generated an image: {image_path}\n')

# =========================== Step 2: Object Detection ===========================
object_detector = ObjectDetector(image_path)
objects = object_detector.detect_objects()

targets = []

for i, obj in enumerate(objects):
    x0, y0 = obj['center']
    r = obj['radius']
    targets.append((y0, height - 1 - x0, r)) # convert pixels to 2D coordinate system for easier work

print(f'Detected {len(targets)} targets:')
for target in targets:
    print(target)
print()

# ======================== Step 3: Solving Boundary Value Problem ========================
g = 250
time = 3
bvp_solver = BVPSolver(g, time, ball_radius)

# shoot from random point
x0 = random.randint(0, width - 1)
y0 = random.randint(0, height - 1)

x0, y0 = 258, 411 # for document's examples (comment this for random start point)

print(f'Shooting from point ({x0}, {y0}):')
trajectories, sorted_targets = bvp_solver.shooting_method(targets, (x0, y0))
targets = sorted_targets
print()

# ======================== Step 4: Animation ========================
target_path = r'Final-Project\Project-1_Hitting_Target\assets\target.png'
ball_path = r'Final-Project\Project-1_Hitting_Target\assets\ball.png'
animation_name = f'throwing_ball_{time}s'

ball_animation = BallThrowingAnimation((x0, y0), width, target_path, ball_path, scale_factor = video_scale_factor)
ball_animation.animate(targets, trajectories)#, output_path = fr'Final-Project\Project-1_Hitting_Target\output\g={g}\{animation_name}.mp4')