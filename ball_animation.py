import pygame
import sys
import time
import numpy as np
import imageio
import os

class BallThrowingAnimation:
    def __init__(self, start_point, animation_width, target_path, ball_path, scale_factor = 0.5):
        pygame.init()

        self.start_point = start_point
        self.ball_path = ball_path
        self.target_path = target_path
        self.scale_factor = scale_factor

        self.img_width = animation_width

        self.screen = None  
        self.clock = pygame.time.Clock()

        self.ball_original = None
        self.target_original = None

    def animate(self, targets, trajectories, output_path = None):
        max_trajectory_height = max((max((y for (_, _, y) in traj), default = 0) for traj in trajectories), default = 0)
        max_start_y = self.start_point[1]
        overall_max_height = max(max_trajectory_height, max_start_y) + 100

        unscaled_window_width = self.img_width
        unscaled_window_height = overall_max_height

        window_width = int(unscaled_window_width * self.scale_factor)
        window_height = int(unscaled_window_height * self.scale_factor)

        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Ball Throwing Animation")

        self.ball_original = pygame.image.load(self.ball_path).convert_alpha()
        self.target_original = pygame.image.load(self.target_path).convert_alpha()

        ball_scale_width = int((unscaled_window_width // 30) * self.scale_factor)
        ball_scale_height = ball_scale_width
        self.ball_image = pygame.transform.scale(
            self.ball_original, (ball_scale_width, ball_scale_height)
        )

        processed_targets = []
        for target in targets:
            x, y, radius = target
            scaled_radius = int(radius * self.scale_factor)
            scaled_diameter = scaled_radius * 2

            scaled_target_image = pygame.transform.scale(
                self.target_original, (scaled_diameter, scaled_diameter)
            )

            pygame_x = int(x * self.scale_factor)
            pygame_y = int((unscaled_window_height - y) * self.scale_factor)

            processed_targets.append({
                'x': pygame_x,
                'y': pygame_y,
                'radius': scaled_radius,
                'image': scaled_target_image
            })

        active_targets = processed_targets.copy()
        
        writer = None
        if output_path is not None:
            video_dir = os.path.dirname(output_path)
            if video_dir and not os.path.exists(video_dir):
                os.makedirs(video_dir)
            try:
                writer = imageio.get_writer(output_path, fps=60, codec='libx264')
            except Exception as e:
                print(f"Failed to initialize video writer for '{output_path}': {e}")
                sys.exit(1)
                
        start_x, start_y = self.start_point
        start_x = int(start_x * self.scale_factor)
        start_y = int((unscaled_window_height - start_y) * self.scale_factor)

        initial_delay = 1.5
        delay_start_time = time.time()

        while time.time() - delay_start_time < initial_delay:
            self.screen.fill((220, 220, 220))
            
            for target in active_targets:
                self.screen.blit(
                    target['image'],
                    target['image'].get_rect(center=(target['x'], target['y']))
                )

            pygame.draw.circle(self.screen, (0, 0, 0), (start_x, start_y), 5)
            pygame.display.flip()

            if writer is not None:
                try:
                    frame_pixels = pygame.surfarray.array3d(self.screen)
                    frame_pixels = np.transpose(frame_pixels, (1, 0, 2))
                    writer.append_data(frame_pixels)
                except Exception as e:
                    print('Failed to append frame to video during delay:', e)

            self.clock.tick(60)
            
        paired = list(zip(trajectories, processed_targets))

        for i, (trajectory, target) in enumerate(paired):
            print(f'Animating trajectory {i+1}/{len(trajectories)}')

            start_time = time.time()
            point_i = 0
            num_points = len(trajectory)
            running = True

            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        if writer is not None:
                            writer.close()
                        pygame.quit()
                        sys.exit()

                cur_time = time.time()
                elapsed_time = cur_time - start_time

                while point_i < num_points and elapsed_time >= trajectory[point_i][0]:
                    point_i += 1

                if point_i == 0:
                    cur_pos = (trajectory[0][1], trajectory[0][2])
                elif point_i >= num_points:
                    cur_pos = (trajectory[-1][1], trajectory[-1][2])
                    running = False
                else:
                    t1, x1, y1 = trajectory[point_i - 1]
                    t2, x2, y2 = trajectory[point_i]

                    dt = (t2 - t1)
                    if dt == 0:
                        factor = 0.0
                    else:
                        factor = (elapsed_time - t1) / dt
                        factor = max(0, min(factor, 1))

                    cur_x = x1 + (x2 - x1) * factor
                    cur_y = y1 + (y2 - y1) * factor
                    cur_pos = (cur_x, cur_y)

                self.screen.fill((220, 220, 220))

                for remaining_target in active_targets:
                    self.screen.blit(
                        remaining_target['image'],
                        remaining_target['image'].get_rect(center=(remaining_target['x'], remaining_target['y']))
                    )

                pygame.draw.circle(self.screen, (0, 0, 0), (start_x, start_y), 5)

                user_x, user_y = cur_pos
                pygame_x = int(user_x * self.scale_factor)
                pygame_y = int((unscaled_window_height - user_y) * self.scale_factor)

                pygame_y = max(0, min(pygame_y, window_height))
                pygame_x = max(0, min(pygame_x, window_width))

                ball_rect = self.ball_image.get_rect(center=(pygame_x, pygame_y))
                self.screen.blit(self.ball_image, ball_rect)

                pygame.display.flip()

                if writer is not None:
                    try:
                        frame_pixels = pygame.surfarray.array3d(self.screen)
                        frame_pixels = np.transpose(frame_pixels, (1, 0, 2))
                        writer.append_data(frame_pixels)
                    except Exception as e:
                        print('Failed to append frame to video:', e)

                self.clock.tick(60)

                if not running:
                    if target in active_targets:
                        active_targets.remove(target)

                    wait_time = 0.5
                    wait_start = time.time()

                    while time.time() - wait_start < wait_time:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                if writer is not None:
                                    writer.close()
                                pygame.quit()
                                sys.exit()
                        pygame.display.flip()
                        if writer is not None:
                            try:
                                frame_pixels = pygame.surfarray.array3d(self.screen)
                                frame_pixels = np.transpose(frame_pixels, (1, 0, 2))
                                writer.append_data(frame_pixels)
                            except Exception as e:
                                print('Failed to append frame to video during wait:', e)
                        self.clock.tick(60)
                    break
                
        if writer is not None:
            try:
                writer.close()
            except Exception as e:
                print('Failed to close writer:', e)

        pygame.quit()