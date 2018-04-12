from __future__ import division, print_function

import numpy as np
import datetime
import pygame
import shutil
import os

from highway_env.road.graphics import WorldSurface, RoadGraphics
from highway_env.vehicle.graphics import VehicleGraphics


class EnvViewer(object):
    """
        A viewer to render a highway driving environment.
    """
    SCREEN_WIDTH = 600
    SCREEN_HEIGHT = 150

    # TODO: move video recording to a monitoring wrapper
    VIDEO_SPEED = 2
    OUT_FOLDER = 'out'
    TMP_FOLDER = os.path.join(OUT_FOLDER, 'tmp')

    def __init__(self, env, record_video=True):
        self.env = env
        self.record_video = record_video

        pygame.init()
        pygame.display.set_caption("Highway-env")
        panel_size = (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        self.screen = pygame.display.set_mode([self.SCREEN_WIDTH, self.SCREEN_HEIGHT])
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.clock = pygame.time.Clock()

        if self.record_video:
            self.frame = 0
            self.make_video_dir()
            self.video_name = 'highway_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))

    def handle_events(self):
        """
            Handle pygame events by forwarding them to the display and environment vehicle.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.env.close()
            self.sim_surface.handle_event(event)
            if self.env.vehicle:
                VehicleGraphics.handle_event(self.env.vehicle, event)

    def display(self):
        """
            Display the road and vehicles on a pygame window.
        """
        self.sim_surface.move_display_window_to(self.window_position())
        RoadGraphics.display(self.env.road, self.sim_surface)
        RoadGraphics.display_traffic(self.env.road, self.sim_surface)
        self.screen.blit(self.sim_surface, (0, 0))
        self.clock.tick(self.env.SIMULATION_FREQUENCY)
        pygame.display.flip()

        if self.record_video:
            self.frame += 1
            pygame.image.save(self.screen, "{}/{}_{:04d}.bmp".format(self.TMP_FOLDER,
                                                                     self.video_name,
                                                                     self.frame))

    def get_image(self):
        """
        :return: the rendered image as a rbg array
        """
        data = pygame.surfarray.array3d(self.screen)
        return np.moveaxis(data, 0, 1)

    def window_position(self):
        """
        :return: the world position of the center of the displayed window.
        """
        if self.env.vehicle:
            if False:
                return self.env.vehicle.position
            else:
                return np.array([self.env.vehicle.position[0], len(self.env.road.lanes) / 2 * 4 - 2])
        else:
            return np.array([0, len(self.env.road.lanes) / 2 * 4])

    def make_video_dir(self):
        """
            Make a temporary directory to hold the rendered images. If already existing, clear it.
        """
        if not os.path.exists(self.OUT_FOLDER):
            os.mkdir(self.OUT_FOLDER)
        self.clear_video_dir()
        os.mkdir(self.TMP_FOLDER)

    def clear_video_dir(self):
        """
            Clear the temporary directory containing the rendered images.
        """
        if os.path.exists(self.TMP_FOLDER):
            shutil.rmtree(self.TMP_FOLDER, ignore_errors=True)

    def close(self):
        """
            Close the pygame window.

            If video frames were recorded, convert them to a video/gif file
        """
        pygame.quit()
        if self.record_video:
            os.system("ffmpeg -r {3} -i {0}/{2}_%04d.bmp -vcodec libx264 -crf 25 {1}/{2}.avi"
                      .format(self.TMP_FOLDER,
                              self.OUT_FOLDER,
                              self.video_name,
                              self.VIDEO_SPEED * self.env.SIMULATION_FREQUENCY))
            delay = int(np.round(100 / (self.VIDEO_SPEED * self.env.SIMULATION_FREQUENCY)))
            os.system("convert -delay {3} -loop 0 {0}/{2}*.bmp {1}/{2}.gif"
                      .format(self.TMP_FOLDER,
                              self.OUT_FOLDER,
                              self.video_name,
                              delay))
            self.clear_video_dir()